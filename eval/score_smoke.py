#!/usr/bin/env python3
"""Score frozen Smoke predictions and optionally resolve ZoomBench with the frozen Judge."""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
import re
from typing import Any

import yaml
from openai import OpenAI

from eval.run_smoke import final_answer, parse_mcq, percentile, resolve_path, sha256_file


def score_key(item: dict[str, Any]) -> str:
    return f"{item['benchmark']}\0{item['view']}\0{item['sample_uid']}"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in sorted(rows, key=lambda x: (x["benchmark"], x["view"], x["sample_uid"])):
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def normalize_judge_output(text: str) -> bool | None:
    value = str(text or "").strip().casefold()
    if value == "yes":
        return True
    if value == "no":
        return False
    return None


SINGLE_NUMERIC_RE = re.compile(r"^[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?$")

def parse_single_numeric(value: Any) -> Decimal | None:
    """Parse only a complete standalone integer/decimal, never embedded prose."""
    normalized = str(value or "").strip()
    if not SINGLE_NUMERIC_RE.fullmatch(normalized):
        return None
    try:
        return Decimal(normalized.replace(",", ""))
    except InvalidOperation:
        return None


def base_score(item: dict[str, Any]) -> dict[str, Any]:
    score = {
        "schema_version": 1,
        "benchmark": item["benchmark"],
        "view": item["view"],
        "sample_uid": item["sample_uid"],
        "question_format": item["question_format"],
        "official_category": item["official_category"],
        "reference_answer": item["reference_answer"],
        "parsed_answer": "",
        "is_correct": False,
        "score_status": "scored",
        "score_source": "",
        "judge_raw": None,
        "judge_error": None,
        "inference_error": item.get("error"),
    }
    if item.get("error"):
        score["score_source"] = "inference_error"
        return score
    if item["question_format"] == "multiple_choice":
        parsed, source = parse_mcq(item["raw_model_answer"])
        score["parsed_answer"] = parsed
        score["score_source"] = source
        score["is_correct"] = bool(parsed and parsed == str(item["reference_answer"]).strip().upper())
        return score

    parsed = final_answer(item["raw_model_answer"])
    score["parsed_answer"] = parsed
    reference_numeric = parse_single_numeric(item["reference_answer"])
    prediction_numeric = parse_single_numeric(parsed)
    if reference_numeric is not None and prediction_numeric is not None:
        score["is_correct"] = reference_numeric == prediction_numeric
        score["score_source"] = (
            "deterministic_numeric_equal"
            if score["is_correct"]
            else "deterministic_numeric_mismatch"
        )
        return score
    try:
        from mathruler.grader import grade_answer

        if bool(grade_answer(item["reference_answer"], parsed)):
            score["is_correct"] = True
            score["score_source"] = "mathruler"
            return score
    except Exception as exc:
        score["judge_error"] = f"MathRuler {type(exc).__name__}: {exc}"
    score["score_status"] = "pending_judge"
    score["score_source"] = "fixed_base_4b_judge_required"
    return score


def call_judge(
    pending: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    config: dict[str, Any],
    api_base: str,
    api_key: str,
    model_id: str,
    workers: int,
) -> None:
    judge_cfg = config["judge"]
    template = str(judge_cfg["prompt"])
    temperature = float(judge_cfg["generation"]["temperature"])
    max_tokens = int(judge_cfg["generation"]["max_new_tokens"])
    enable_thinking = bool(judge_cfg["generation"].get("enable_thinking", False))
    system_prompt = str(judge_cfg["system_prompt"])
    thread_local = threading.local()

    def get_client() -> OpenAI:
        client = getattr(thread_local, "client", None)
        if client is None:
            client = OpenAI(api_key=api_key, base_url=api_base, timeout=600)
            thread_local.client = client
        return client

    def run_one(index: int, prediction: dict[str, Any]) -> tuple[int, str, str | None]:
        prompt = template.format(
            question=prediction["prompt"],
            reference_answer=prediction["reference_answer"],
            model_answer=final_answer(prediction["raw_model_answer"]),
        )
        for attempt in range(1, 4):
            try:
                response = get_client().chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
                )
                return index, str(response.choices[0].message.content or "").strip(), None
            except Exception as exc:
                if attempt == 3:
                    return index, "", f"{type(exc).__name__}: {exc}"
                time.sleep(attempt)
        raise AssertionError("unreachable")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(run_one, index, prediction): (index, score)
            for index, (prediction, score) in enumerate(pending)
        }
        for future in as_completed(futures):
            index, raw, error = future.result()
            score = pending[index][1]
            decision = normalize_judge_output(raw)
            score["judge_raw"] = raw
            score["judge_error"] = error
            score["score_status"] = "scored"
            score["score_source"] = "fixed_base_4b_judge" if decision is not None else "judge_failure"
            score["is_correct"] = bool(decision)
            if decision is None and not error:
                score["judge_error"] = "Judge output was not exactly Yes or No"


def summarize(
    predictions: list[dict[str, Any]],
    scores: list[dict[str, Any]],
    *,
    config_sha256: str,
    selection_sha256: str,
    judge_model: str | None,
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for score in scores:
        groups.setdefault(f"{score['benchmark']}/{score['view']}", []).append(score)
    result_groups = {}
    for name, items in sorted(groups.items()):
        pending = sum(x["score_status"] != "scored" for x in items)
        correct = sum(bool(x["is_correct"]) for x in items)
        result_groups[name] = {
            "total": len(items),
            "correct": correct,
            "incorrect": len(items) - correct - pending,
            "pending_judge": pending,
            "accuracy": correct / len(items) if not pending and items else None,
        }
    latencies = [float(x["latency_seconds"]) for x in predictions]
    zoom_full = result_groups.get("zoombench/full", {})
    zoom_crop = result_groups.get("zoombench/crop", {})
    gap = None
    if zoom_full.get("accuracy") is not None and zoom_crop.get("accuracy") is not None:
        gap = zoom_crop["accuracy"] - zoom_full["accuracy"]
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision_status": "complete" if not any(x["score_status"] != "scored" for x in scores) else "pending_judge",
        "request_count": len(predictions),
        "unique_sample_uid_count": len({(x["benchmark"], x["sample_uid"]) for x in predictions}),
        "inference_error_count": sum(bool(x.get("error")) for x in predictions),
        "judge": {
            "model": judge_model,
            "mathruler_count": sum(x["score_source"] == "mathruler" for x in scores),
            "llm_judge_count": sum(x["score_source"] == "fixed_base_4b_judge" for x in scores),
            "deterministic_numeric_count": sum(
                x["score_source"].startswith("deterministic_numeric_") for x in scores
            ),
            "judge_failure_count": sum(x["score_source"] == "judge_failure" for x in scores),
            "pending_judge_count": sum(x["score_status"] == "pending_judge" for x in scores),
        },
        "groups": result_groups,
        "zoombench_zooming_gap": gap,
        "performance": {
            "latency_mean_seconds": statistics.mean(latencies) if latencies else 0.0,
            "latency_p95_seconds": percentile(latencies, 0.95),
            "prompt_tokens": sum(int(x.get("prompt_tokens") or 0) for x in predictions),
            "completion_tokens": sum(int(x.get("completion_tokens") or 0) for x in predictions),
        },
        "config_sha256": config_sha256,
        "selection_manifest_sha256": selection_sha256,
        "resume_gate": {
            "unique_request_keys": len({score_key(x) for x in predictions}),
            "duplicate_request_keys": len(predictions) - len({score_key(x) for x in predictions}),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/benchmark_eval.yaml")
    parser.add_argument("--input-dir")
    parser.add_argument("--judge-api-base")
    parser.add_argument("--judge-api-key", default="EMPTY")
    parser.add_argument("--judge-model-id")
    parser.add_argument("--judge-workers", type=int, default=4)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    repo_root = config_path.parent.parent
    run_root = resolve_path(config["paths"]["run_root"], repo_root)
    input_dir = Path(args.input_dir).resolve() if args.input_dir else run_root / "smoke" / "base"
    predictions_path = input_dir / "predictions.jsonl"
    scores_path = input_dir / "scores.jsonl"
    summary_path = input_dir / "summary.json"
    predictions = load_jsonl(predictions_path)
    if len(predictions) != 64:
        raise ValueError(f"expected 64 Smoke request records, found {len(predictions)}")

    previous = {score_key(x): x for x in load_jsonl(scores_path)}
    scores = [base_score(item) for item in predictions]
    by_prediction = {score_key(item): item for item in predictions}
    pending: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for score in scores:
        old = previous.get(score_key(score))
        if old and old.get("score_source") == "fixed_base_4b_judge":
            score.update(old)
        if score["score_status"] == "pending_judge":
            pending.append((by_prediction[score_key(score)], score))

    if pending and args.judge_api_base:
        expected_model = str(config["judge"]["model"]["served_model_name"])
        model_id = args.judge_model_id or expected_model
        if model_id != expected_model:
            raise ValueError(f"judge model must be frozen model {expected_model!r}")
        call_judge(
            pending,
            config=config,
            api_base=args.judge_api_base,
            api_key=args.judge_api_key,
            model_id=model_id,
            workers=args.judge_workers,
        )

    write_jsonl(scores_path, scores)
    manifest_path = resolve_path(config["smoke"]["manifest"], repo_root)
    summary = summarize(
        predictions,
        scores,
        config_sha256=sha256_file(config_path),
        selection_sha256=sha256_file(manifest_path),
        judge_model=args.judge_model_id if args.judge_api_base else None,
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
