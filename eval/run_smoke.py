#!/usr/bin/env python3
"""Run the frozen Day 5 Base Smoke against an OpenAI-compatible VLM service."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import re
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from openai import OpenAI


MCQ_RE = re.compile(r"(?i)(?:^|\b)(?:answer|option)?\s*[:：\-]?\s*[\(\[]?([A-D])[\)\]]?(?:\b|$)")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(value: str | Path, repo_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (repo_root / path).resolve()


def final_answer(text: str) -> str:
    text = str(text or "")
    think_end = text.rfind("</think>")
    if think_end >= 0:
        text = text[think_end + len("</think>"):]
    matches = list(re.finditer(r"<answer>(.*?)</answer>", text, flags=re.I | re.S))
    if matches:
        return matches[-1].group(1).strip()
    return text.strip()


def parse_mcq(text: str) -> tuple[str, str]:
    answer = final_answer(text)
    exact = re.fullmatch(r"\s*[\(\[]?([A-Da-d])[\)\]]?[\s\.]*", answer)
    if exact:
        return exact.group(1).upper(), "exact_final_option"
    explicit = re.findall(
        r"(?i)(?:final\s+answer|answer|option)\s*(?:is|:|：|-)?\s*[\(\[]?([A-D])[\)\]]?",
        answer,
    )
    if explicit and len(set(value.upper() for value in explicit)) == 1:
        return explicit[-1].upper(), "explicit_final_option"
    tail = re.search(r"(?i)[\(\[]?([A-D])[\)\]]?[\s\.]*$", answer)
    if tail:
        return tail.group(1).upper(), "trailing_final_option"
    return "", "invalid_or_ambiguous"


def image_data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def load_selected_rows(
    config: dict[str, Any],
    manifest: dict[str, Any],
    repo_root: Path,
) -> list[dict[str, Any]]:
    data_root = resolve_path(config["paths"]["data_root"], repo_root)
    tasks: list[dict[str, Any]] = []
    for benchmark in ("zoombench", "mmstar", "vstar"):
        selected_uids = set(manifest["benchmarks"][benchmark]["sample_uids"])
        converted_path = data_root / "converted" / benchmark / f"{benchmark}.json"
        rows = json.loads(converted_path.read_text(encoding="utf-8"))
        selected = [row for row in rows if row["sample_uid"] in selected_uids]
        if len(selected) != 16 or {row["sample_uid"] for row in selected} != selected_uids:
            raise ValueError(f"{benchmark}: manifest does not resolve to exactly 16 rows")
        by_uid = {row["sample_uid"]: row for row in selected}
        ordered = [by_uid[uid] for uid in manifest["benchmarks"][benchmark]["sample_uids"]]
        views = ("full", "crop") if benchmark == "zoombench" else ("full",)
        for view in views:
            for row in ordered:
                image_values = row.get("crop_images") if view == "crop" else row.get("images")
                if not image_values:
                    raise ValueError(f"{benchmark}/{row['sample_uid']}/{view}: missing image")
                image_path = Path(str(image_values[0]))
                if not image_path.is_file():
                    raise FileNotFoundError(image_path)
                tasks.append({"benchmark": benchmark, "view": view, "row": row, "image_path": image_path})
    return tasks


def record_key(record: dict[str, Any]) -> str:
    return f"{record['benchmark']}\0{record['view']}\0{record['sample_uid']}"


def load_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            records[record_key(item)] = item
        except Exception:
            continue
    return records


def write_records(path: Path, records: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    ordered = sorted(records.values(), key=lambda x: (x["benchmark"], x["view"], x["sample_uid"]))
    with temporary.open("w", encoding="utf-8") as handle:
        for item in ordered:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * fraction + 0.999999)))
    return ordered[index]


def score_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scores = []
    for item in records:
        score = {
            "benchmark": item["benchmark"],
            "view": item["view"],
            "sample_uid": item["sample_uid"],
            "question_format": item["question_format"],
            "category": item["category"],
            "reference_answer": item["reference_answer"],
            "parsed_answer": "",
            "is_correct": False,
            "score_status": "scored",
            "score_source": "",
            "error": item.get("error"),
        }
        if item.get("error"):
            score["score_source"] = "inference_error"
        elif item["question_format"] == "multiple_choice":
            parsed, source = parse_mcq(item["raw_model_answer"])
            score["parsed_answer"] = parsed
            score["score_source"] = source
            score["is_correct"] = bool(parsed and parsed == str(item["reference_answer"]).strip().upper())
        else:
            score["parsed_answer"] = final_answer(item["raw_model_answer"])
            score["score_status"] = "pending_judge"
            score["score_source"] = "mathruler_or_frozen_qwen_judge_required"
        scores.append(score)

    groups: dict[str, list[dict[str, Any]]] = {}
    for item in scores:
        group = f"{item['benchmark']}/{item['view']}"
        groups.setdefault(group, []).append(item)
    summary_groups = {}
    for group, items in sorted(groups.items()):
        scored = [x for x in items if x["score_status"] == "scored"]
        correct = sum(bool(x["is_correct"]) for x in scored)
        summary_groups[group] = {
            "total": len(items),
            "scored": len(scored),
            "pending_judge": sum(x["score_status"] == "pending_judge" for x in items),
            "correct": correct,
            "incorrect_or_error": len(scored) - correct,
            "accuracy_on_scored": correct / len(scored) if scored else None,
        }
    latencies = [float(x["latency_seconds"]) for x in records if x.get("latency_seconds") is not None]
    summary = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision_status": "pending_judge" if any(x["score_status"] == "pending_judge" for x in scores) else "complete",
        "request_count": len(records),
        "unique_sample_uid_count": len({(x["benchmark"], x["sample_uid"]) for x in records}),
        "inference_error_count": sum(bool(x.get("error")) for x in records),
        "groups": summary_groups,
        "performance": {
            "latency_mean_seconds": statistics.mean(latencies) if latencies else 0.0,
            "latency_p95_seconds": percentile(latencies, 0.95),
            "prompt_tokens": sum(int(x.get("prompt_tokens") or 0) for x in records),
            "completion_tokens": sum(int(x.get("completion_tokens") or 0) for x in records),
        },
    }
    return scores, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/benchmark_eval.yaml")
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model-id", default="vision-opd-base")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    repo_root = config_path.parent.parent
    manifest_path = resolve_path(config["smoke"]["manifest"], repo_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # A scoring-only protocol amendment must not invalidate the frozen UID set.
    selection_fields = ("selection_seed", "selection_method", "selection_rank_input", "selection_rank_order")
    for field in selection_fields:
        if manifest.get(field) != config["smoke"].get(field):
            raise ValueError(f"Smoke manifest selection field changed: {field}")
    if manifest.get("experiment_id") != config["protocol"]["experiment_id"]:
        raise ValueError("Smoke manifest belongs to a different experiment")
    for benchmark, spec in config["smoke"]["selection"].items():
        frozen = manifest["benchmarks"][benchmark]
        if frozen.get("stratify_by") != spec["stratify_by"]:
            raise ValueError(f"{benchmark}: frozen stratification changed")
        if frozen.get("quotas") != spec["quotas"]:
            raise ValueError(f"{benchmark}: frozen quotas changed")
        if frozen.get("sample_count") != config["smoke"]["samples_per_benchmark"]:
            raise ValueError(f"{benchmark}: frozen sample count changed")
        if frozen.get("source_revision") != config["benchmarks"][benchmark]["dataset_revision"]:
            raise ValueError(f"{benchmark}: frozen source revision changed")
    tasks = load_selected_rows(config, manifest, repo_root)
    run_root = resolve_path(config["paths"]["run_root"], repo_root)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else run_root / "smoke" / "base"
    predictions_path = output_dir / "predictions.jsonl"
    scores_path = output_dir / "scores.jsonl"
    summary_path = output_dir / "summary.json"
    records = load_checkpoint(predictions_path)

    generation = config["generation"]
    system_prompt = config["prompt_and_image"]["system_prompt"]
    checkpoint = config["model_under_test"]["base_weight_sha256"]
    checkpoint_identity = hashlib.sha256(
        json.dumps(checkpoint, sort_keys=True).encode("utf-8")
    ).hexdigest()
    thread_local = threading.local()

    def get_client() -> OpenAI:
        client = getattr(thread_local, "client", None)
        if client is None:
            client = OpenAI(api_key=args.api_key, base_url=args.api_base, timeout=3600)
            thread_local.client = client
        return client

    def run_one(task: dict[str, Any]) -> dict[str, Any]:
        row = task["row"]
        key = f"{task['benchmark']}\0{task['view']}\0{row['sample_uid']}"
        previous = records.get(key)
        if previous and not previous.get("error") and str(previous.get("raw_model_answer", "")).strip():
            return previous
        query = str(row["query"]).replace("<image>", "").strip()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": image_data_uri(task["image_path"])}},
                {"type": "text", "text": query},
            ]},
        ]
        started = time.perf_counter()
        raw_answer = ""
        error = None
        finish_reason = None
        prompt_tokens = completion_tokens = 0
        attempts = 0
        for attempts in range(1, args.max_retries + 1):
            try:
                response = get_client().chat.completions.create(
                    model=args.model_id,
                    messages=messages,
                    seed=int(generation["seed"]),
                    temperature=float(generation["temperature"]),
                    top_p=float(generation["top_p"]),
                    presence_penalty=float(generation["presence_penalty"]),
                    max_tokens=int(generation["max_new_tokens"]),
                    extra_body={
                        "top_k": int(generation["top_k"]),
                        "repetition_penalty": float(generation["repetition_penalty"]),
                    },
                )
                choice = response.choices[0]
                raw_answer = str(choice.message.content or "").strip()
                finish_reason = choice.finish_reason
                usage = response.usage
                prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
                completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
                if not raw_answer:
                    raise ValueError("empty model response")
                break
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                if attempts < args.max_retries:
                    time.sleep(attempts)
        if raw_answer:
            error = None
        return {
            "schema_version": 1,
            "benchmark": task["benchmark"],
            "view": task["view"],
            "sample_uid": str(row["sample_uid"]),
            "source_id": str(row["source_id"]),
            "official_category": str(row.get("category", "unavailable_official")),
            "category": str(row.get("category", "unavailable_official")),
            "question_format": str(row["question_format"]),
            "dataset_revision": str(row["source_revision"]),
            "image_sha256": (
                str((row.get("crop_image_sha256") or [""])[0])
                if task["view"] == "crop"
                else str(row.get("image_sha256", ""))
            ),
            "model_id": args.model_id,
            "model_checkpoint_sha256": checkpoint,
            "model_checkpoint_identity": checkpoint_identity,
            "selection_manifest_sha256": sha256_file(manifest_path),
            "prompt": query,
            "raw_model_answer": raw_answer,
            "parsed_answer": final_answer(raw_answer),
            "reference_answer": str(row["response"]),
            "finish_reason": finish_reason,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "latency_seconds": round(time.perf_counter() - started, 6),
            "retry_count": max(0, attempts - 1),
            "error": error,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        }

    pending = [
        task for task in tasks
        if not (
            record_key({"benchmark": task["benchmark"], "view": task["view"], "sample_uid": task["row"]["sample_uid"]}) in records
            and not records[record_key({"benchmark": task["benchmark"], "view": task["view"], "sample_uid": task["row"]["sample_uid"]})].get("error")
        )
    ]
    print(f"Smoke requests: total={len(tasks)} completed={len(tasks)-len(pending)} pending={len(pending)}", flush=True)
    if pending:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(run_one, task): task for task in pending}
            completed = 0
            for future in as_completed(futures):
                item = future.result()
                records[record_key(item)] = item
                write_records(predictions_path, records)
                completed += 1
                print(
                    f"[{completed}/{len(pending)}] {item['benchmark']}/{item['view']}/"
                    f"{item['sample_uid']} error={bool(item['error'])} "
                    f"latency={item['latency_seconds']:.2f}s",
                    flush=True,
                )

    final_records = sorted(records.values(), key=lambda x: (x["benchmark"], x["view"], x["sample_uid"]))
    expected_keys = {
        f"{task['benchmark']}\0{task['view']}\0{task['row']['sample_uid']}" for task in tasks
    }
    final_records = [item for item in final_records if record_key(item) in expected_keys]
    write_records(predictions_path, {record_key(item): item for item in final_records})
    scores, summary = score_records(final_records)
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    with scores_path.open("w", encoding="utf-8") as handle:
        for item in scores:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    summary.update({
        "config_sha256": sha256_file(config_path),
        "selection_manifest_sha256": sha256_file(manifest_path),
        "model_id": args.model_id,
        "expected_request_count": len(tasks),
        "resume_gate": {
            "unique_request_keys": len({record_key(item) for item in final_records}),
            "duplicate_request_keys": len(final_records) - len({record_key(item) for item in final_records}),
        },
    })
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Summary: {summary_path}", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
