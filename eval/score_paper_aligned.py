#!/usr/bin/env python3
"""Apply official rule scoring, merge frozen Base Judge results, and summarize."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from eval.paper_aligned_common import (
    extract_answer_official,
    first_letter_match,
    judge_complete,
    load_config,
    now_utc,
    prediction_complete,
    read_jsonl_map,
    record_key,
    resolve_path,
    sha256_file,
    write_json,
    write_jsonl_map,
)


def load_mathruler() -> Callable[[Any, Any], bool]:
    try:
        from mathruler.grader import grade_answer
    except ImportError as exc:
        raise RuntimeError(
            "MathRuler is required by the frozen scoring protocol; run in the vision-opd environment"
        ) from exc
    return grade_answer


def rule_score(
    prediction: dict[str, Any],
    config: dict[str, Any],
    grader: Callable[[Any, Any], bool],
) -> dict[str, Any]:
    extracted = extract_answer_official(prediction.get("raw_model_answer"))
    result = {
        "schema_version": 1,
        "benchmark": prediction["benchmark"],
        "view": prediction.get("view", "full"),
        "sample_uid": prediction["sample_uid"],
        "source_id": prediction.get("source_id"),
        "question_format": prediction.get("question_format", "unknown"),
        "official_category": prediction.get("official_category", "unavailable_official"),
        "official_l2_category": prediction.get("official_l2_category", "unavailable_official"),
        "reference_answer": prediction["reference_answer"],
        "parsed_answer": extracted,
        "rule_score": None,
        "rule_source": None,
        "rule_error": None,
        "judge_required": False,
        "judge_raw_output": None,
        "judge_normalized_decision": None,
        "judge_source": None,
        "judge_error": None,
        "final_is_correct": False,
        "score_status": "scored",
        "inference_error": prediction.get("error"),
    }
    if not prediction_complete(prediction):
        result["rule_score"] = False
        result["rule_source"] = "inference_failure"
        return result

    math_correct = False
    try:
        math_correct = bool(grader(prediction["reference_answer"], extracted))
    except Exception as exc:
        result["rule_error"] = f"MathRuler {type(exc).__name__}: {exc}"
    if math_correct:
        result["rule_score"] = True
        result["rule_source"] = "mathruler"
        result["final_is_correct"] = True
        return result

    if prediction["benchmark"] in set(config["judge"]["first_letter_benchmarks"]):
        try:
            if first_letter_match(prediction["reference_answer"], extracted):
                result["rule_score"] = True
                result["rule_source"] = "first_letter"
                result["final_is_correct"] = True
                return result
        except Exception as exc:
            result["rule_error"] = f"first_letter {type(exc).__name__}: {exc}"

    result["rule_score"] = False
    result["rule_source"] = "llm_judge_required"
    result["judge_required"] = True
    result["score_status"] = "pending_judge"
    return result


def merge_judge(score: dict[str, Any], judge: dict[str, Any] | None) -> dict[str, Any]:
    if not score["judge_required"]:
        return score
    if not judge or not judge_complete(judge):
        return score
    decision = str(judge.get("normalized_decision") or "")
    score["judge_raw_output"] = judge.get("raw_judge_output")
    score["judge_normalized_decision"] = decision or None
    score["judge_source"] = (
        "fixed_base_4b_judge" if decision in {"Yes", "No"} and not judge.get("error")
        else "judge_failure"
    )
    score["judge_error"] = judge.get("error")
    score["final_is_correct"] = decision == "Yes" and not judge.get("error")
    score["score_status"] = "scored"
    return score


def stat(items: list[dict[str, Any]]) -> dict[str, Any]:
    pending = sum(item["score_status"] != "scored" for item in items)
    correct = sum(bool(item["final_is_correct"]) for item in items)
    incorrect = len(items) - correct - pending
    return {
        "total": len(items),
        "correct": correct,
        "incorrect": incorrect,
        "pending_judge": pending,
        "accuracy": correct / len(items) if items and pending == 0 else None,
    }


def summarize(
    predictions: list[dict[str, Any]],
    scores: list[dict[str, Any]],
    manifest: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    question_formats: dict[str, list[dict[str, Any]]] = defaultdict(list)
    categories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    l2_categories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in scores:
        group = f"{item['benchmark']}/{item['view']}"
        groups[group].append(item)
        question_formats[f"{group}/{item['question_format']}"].append(item)
        categories[f"{group}/{item['official_category']}"].append(item)
        if item["official_l2_category"] != "unavailable_official":
            l2_categories[f"{group}/{item['official_l2_category']}"].append(item)

    expected_counts = manifest["expected_requests"]
    actual_counts = {name: len(groups.get(name, [])) for name in expected_counts}
    if actual_counts != expected_counts:
        raise ValueError(f"score group counts do not match manifest: {actual_counts} != {expected_counts}")
    if (
        manifest["run_mode"] == "formal"
        and actual_counts.get("vstar/full") != int(config["reporting"]["vstar_summary_denominator"])
    ):
        raise ValueError("formal V* score denominator is not 191")

    latencies = [float(item.get("latency_seconds") or 0) for item in predictions]
    pending = sum(item["score_status"] != "scored" for item in scores)
    return {
        "schema_version": 1,
        "experiment_id": "E-PAPER-BASEJUDGE-001",
        "generated_at_utc": now_utc(),
        "run_mode": manifest["run_mode"],
        "model_role": manifest["model_role"],
        "model_id": manifest["model_id"],
        "decision_status": "complete" if pending == 0 else "pending_judge",
        "request_count": len(scores),
        "groups": {name: stat(items) for name, items in sorted(groups.items())},
        "question_format_groups": {
            name: stat(items) for name, items in sorted(question_formats.items())
        },
        "official_category_groups": {
            name: stat(items) for name, items in sorted(categories.items())
        },
        "official_l2_category_groups": {
            name: stat(items) for name, items in sorted(l2_categories.items())
        },
        "scoring_pipeline": {
            "mathruler_correct_count": sum(item["rule_source"] == "mathruler" for item in scores),
            "first_letter_correct_count": sum(item["rule_source"] == "first_letter" for item in scores),
            "llm_judge_required_count": sum(bool(item["judge_required"]) for item in scores),
            "llm_judge_completed_count": sum(item["judge_source"] == "fixed_base_4b_judge" for item in scores),
            "judge_failure_count": sum(item["judge_source"] == "judge_failure" for item in scores),
            "pending_judge_count": pending,
            "inference_failure_count": sum(item["rule_source"] == "inference_failure" for item in scores),
        },
        "generation_diagnostics": {
            "finish_reason_length_count": sum(
                str(item.get("finish_reason") or "").casefold() == "length"
                for item in predictions
            ),
            "prompt_tokens": sum(int(item.get("prompt_tokens") or 0) for item in predictions),
            "completion_tokens": sum(int(item.get("completion_tokens") or 0) for item in predictions),
            "latency_sum_seconds": sum(latencies),
            "latency_mean_seconds": statistics.mean(latencies) if latencies else 0.0,
        },
        "resume_gate": {
            "unique_prediction_keys": len({record_key(item) for item in predictions}),
            "unique_score_keys": len({record_key(item) for item in scores}),
            "duplicate_prediction_keys": len(predictions) - len({record_key(item) for item in predictions}),
            "duplicate_score_keys": len(scores) - len({record_key(item) for item in scores}),
        },
        "config_sha256_raw_bytes": manifest["config_sha256_raw_bytes"],
        "amendment_sha256_raw_bytes": manifest["amendment_sha256_raw_bytes"],
        "limitation_statement": config["reporting"]["required_limitation_statement"],
    }


def write_artifact_hashes(out: Path, config: dict[str, Any]) -> None:
    names = (
        config["paths"]["run_manifest_name"],
        config["paths"]["predictions_name"],
        config["paths"]["judge_results_name"],
        config["paths"]["scores_name"],
        config["paths"]["summary_name"],
        config["paths"]["resume_status_name"],
        config["paths"]["metrics_name"],
        config["paths"]["cost_name"],
    )
    lines = []
    for name in names:
        path = out / name
        if path.is_file():
            lines.append(f"{sha256_file(path)}  {name}")
    target = out / config["paths"]["artifact_sha256_name"]
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    temporary.replace(target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/benchmark_eval_paper_basejudge_r3_single_gpu.yaml")
    parser.add_argument("--input-dir", required=True)
    args = parser.parse_args()

    _, config = load_config(args.config)
    out = resolve_path(args.input_dir)
    manifest_path = out / config["paths"]["run_manifest_name"]
    if not manifest_path.is_file():
        raise FileNotFoundError(f"run manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("experiment_id") != "E-PAPER-BASEJUDGE-001":
        raise ValueError("input directory is not a paper-aligned run")

    predictions_map, prediction_stats = read_jsonl_map(
        out / config["paths"]["predictions_name"], complete=prediction_complete
    )
    expected = int(manifest["expected_request_count"])
    if len(predictions_map) != expected:
        raise ValueError(f"expected {expected} unique predictions, found {len(predictions_map)}")
    predictions = sorted(
        predictions_map.values(),
        key=lambda item: (item["benchmark"], item["view"], item["sample_uid"]),
    )
    grader = load_mathruler()
    scores = [rule_score(prediction, config, grader) for prediction in predictions]
    judge_map, judge_stats = read_jsonl_map(
        out / config["paths"]["judge_results_name"], complete=judge_complete
    )
    scores = [merge_judge(score, judge_map.get(record_key(score))) for score in scores]
    scores_map = {record_key(score): score for score in scores}
    write_jsonl_map(out / config["paths"]["scores_name"], scores_map)
    summary = summarize(predictions, scores, manifest, config)
    summary["input_compaction"] = {
        "prediction_malformed_lines": prediction_stats["malformed_lines"],
        "prediction_duplicate_keys": prediction_stats["duplicate_keys"],
        "judge_malformed_lines": judge_stats["malformed_lines"],
        "judge_duplicate_keys": judge_stats["duplicate_keys"],
    }
    write_json(out / config["paths"]["summary_name"], summary)
    write_artifact_hashes(out, config)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
