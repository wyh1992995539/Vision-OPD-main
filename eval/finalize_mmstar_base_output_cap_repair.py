#!/usr/bin/env python3
"""Close MMStar output-cap ambiguity while preserving all raw evidence."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "configs/mmstar_base_output_cap_terminal_closure.yaml"
FROZEN_CONFIG_SHA256 = "d84570620aab659f145a296c9a846b6f1ebb99aafbfdc5463afb165ec58354ad"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def normalized_paragraphs(text: str) -> list[str]:
    return [
        re.sub(r"\s+", " ", paragraph.strip()).casefold()
        for paragraph in re.split(r"\n\s*\n", text)
        if paragraph.strip()
    ]


def main() -> None:
    config_sha = sha256_file(CONFIG)
    if config_sha != FROZEN_CONFIG_SHA256:
        raise ValueError(f"closure config SHA256 changed: expected {FROZEN_CONFIG_SHA256}, got {config_sha}")
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    source_root = ROOT / config["source"]["root"]
    expected_hashes = config["source"]["files"]
    source_paths = {name: source_root / name for name in expected_hashes}
    source_hashes_before = {name: sha256_file(path) for name, path in source_paths.items()}
    if source_hashes_before != expected_hashes:
        raise ValueError("frozen adaptive source SHA256 mismatch")

    predictions = read_jsonl(source_paths["final_hybrid/predictions.jsonl"])
    scores = read_jsonl(source_paths["final_hybrid/scores.jsonl"])
    traces = read_jsonl(source_paths["final_hybrid/selection_trace.jsonl"])
    expected_total = int(config["completion"]["expected_total"])
    if len(predictions) != expected_total or len(scores) != expected_total:
        raise ValueError("final hybrid must contain exactly 1500 predictions and scores")
    prediction_by_uid = {row["sample_uid"]: row for row in predictions}
    score_by_uid = {row["sample_uid"]: row for row in scores}
    trace_by_uid = {row["sample_uid"]: row for row in traces}
    if len(prediction_by_uid) != expected_total or len(score_by_uid) != expected_total:
        raise ValueError("duplicate or missing sample_uid in final hybrid")

    length_uids = {
        row["sample_uid"]
        for row in predictions
        if str(row.get("finish_reason") or "").casefold() == "length"
    }
    expected_uid = config["terminal_policy"]["expected_sample_uid"]
    if length_uids != {expected_uid}:
        raise ValueError(f"expected only {expected_uid} at raw length cap, got {sorted(length_uids)}")
    terminal_prediction = prediction_by_uid[expected_uid]
    terminal_score = score_by_uid[expected_uid]
    terminal_trace = trace_by_uid[expected_uid]
    safety_cap = int(config["terminal_policy"]["safety_cap_tokens"])
    if terminal_prediction.get("completion_tokens") != safety_cap:
        raise ValueError("terminal sample did not reach the frozen safety cap")
    if terminal_trace.get("final_max_tokens") != safety_cap:
        raise ValueError("terminal trace does not end at the frozen safety cap")
    if config["terminal_policy"]["require_no_r4_option"] and terminal_score.get("mcq_predicted_option") is not None:
        raise ValueError("terminal sample unexpectedly has an R4 option")
    if config["terminal_policy"]["require_exact_judge_no"] and terminal_score.get("judge_normalized_decision") != "No":
        raise ValueError("terminal sample does not have an exact Judge No")
    if terminal_score.get("final_is_correct") is not False:
        raise ValueError("terminal sample must retain its incorrect score")

    paragraphs = normalized_paragraphs(terminal_prediction["raw_model_answer"])
    frequencies = Counter(paragraphs)
    duplicate_instances = sum(count - 1 for count in frequencies.values())
    duplicate_ratio = duplicate_instances / len(paragraphs)
    minimum_ratio = float(config["terminal_policy"]["minimum_duplicate_paragraph_ratio"])
    if duplicate_ratio < minimum_ratio:
        raise ValueError(f"duplicate paragraph ratio {duplicate_ratio:.6f} is below {minimum_ratio:.6f}")

    original_truncated_traces = [row for row in traces if row.get("attempts")]
    natural_after_regeneration = sum(
        row.get("sample_uid") != expected_uid
        and row.get("attempts", [])[-1].get("finish_reason") != "length"
        for row in original_truncated_traces
    )
    expected_natural = int(config["completion"]["expected_natural_stop_after_adaptive_regeneration"])
    if len(original_truncated_traces) != 163 or natural_after_regeneration != expected_natural:
        raise ValueError("unexpected adaptive natural-stop counts")
    correct = sum(bool(row.get("final_is_correct")) for row in scores)
    if correct != int(config["completion"]["expected_correct"]):
        raise ValueError("final score changed during closure")
    if any(row.get("judge_error") or row.get("inference_error") for row in scores):
        raise ValueError("final hybrid still contains Judge or inference errors")

    output = ROOT / config["paths"]["output"]
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(CONFIG, output / "config_snapshot.yaml")
    disposition = config["terminal_policy"]["disposition"]
    resolved_predictions = []
    for row in predictions:
        item = copy.deepcopy(row)
        item["output_cap_resolution"] = disposition if row["sample_uid"] == expected_uid else "completed_output"
        item["unresolved_output_cap"] = False
        resolved_predictions.append(item)
    resolved_scores = []
    for row in scores:
        item = copy.deepcopy(row)
        item["output_cap_resolution"] = disposition if row["sample_uid"] == expected_uid else "completed_output"
        item["unresolved_output_cap"] = False
        resolved_scores.append(item)
    resolved_traces = []
    for row in traces:
        item = copy.deepcopy(row)
        if row["sample_uid"] == expected_uid:
            item["terminal_disposition"] = disposition
            item["terminal_scoring"] = "retain_incorrect"
        resolved_traces.append(item)

    evidence = {
        "schema_version": 1,
        "sample_uid": expected_uid,
        "source_id": terminal_prediction.get("source_id"),
        "raw_finish_reason": terminal_prediction.get("finish_reason"),
        "completion_tokens": terminal_prediction.get("completion_tokens"),
        "raw_answer_characters": len(terminal_prediction["raw_model_answer"]),
        "paragraph_count": len(paragraphs),
        "unique_paragraph_count": len(frequencies),
        "duplicate_paragraph_instances": duplicate_instances,
        "duplicate_paragraph_ratio": duplicate_ratio,
        "maximum_exact_paragraph_frequency": max(frequencies.values()),
        "most_common_paragraphs": [
            {"count": count, "text": text[:240]} for text, count in frequencies.most_common(10)
        ],
        "r4_predicted_option": terminal_score.get("mcq_predicted_option"),
        "r4_parse_status": terminal_score.get("mcq_parse_status"),
        "judge_normalized_decision": terminal_score.get("judge_normalized_decision"),
        "final_is_correct": terminal_score.get("final_is_correct"),
        "disposition": disposition,
        "reason": "The response exhausted 131072 tokens and 98%+ of normalized paragraphs were repeated; no R4 option was produced and the frozen Judge returned No.",
    }
    summary = {
        "schema_version": 1,
        "experiment_id": config["experiment"]["id"],
        "parent_experiment_id": config["experiment"]["parent_experiment_id"],
        "status": "COMPLETE",
        "decision": "OUTPUT_CAP_REPAIR_COMPLETE_WITH_ONE_SCORED_GENERATION_DEGENERACY",
        "total": expected_total,
        "correct": correct,
        "incorrect": expected_total - correct,
        "accuracy": correct / expected_total,
        "original_finish_reason_length_count": 163,
        "natural_stop_after_adaptive_regeneration_count": natural_after_regeneration,
        "terminal_generation_degeneracy_count": 1,
        "raw_finish_reason_length_preserved_count": len(length_uids),
        "unresolved_output_cap_count": 0,
        "judge_error_count": 0,
        "inference_error_count": 0,
        "score_changed_by_closure": False,
        "limitation_statement": config["limitations"]["statement"],
        "updated_at_utc": now_utc(),
    }
    write_jsonl(output / "predictions_resolved.jsonl", resolved_predictions)
    write_jsonl(output / "scores_resolved.jsonl", resolved_scores)
    write_jsonl(output / "selection_trace_resolved.jsonl", resolved_traces)
    write_json(output / "terminal_evidence.json", evidence)
    write_json(output / "summary.json", summary)

    source_hashes_after = {name: sha256_file(path) for name, path in source_paths.items()}
    if source_hashes_after != source_hashes_before:
        raise ValueError("source files changed during closure")
    receipt = {
        **summary,
        "config_sha256": config_sha,
        "source_sha256": source_hashes_after,
        "source_files_unchanged": True,
        "runner_sha256": sha256_file(Path(__file__)),
    }
    write_json(output / "completion_receipt.json", receipt)
    artifact_paths = sorted(path for path in output.iterdir() if path.name != "artifact_sha256.txt")
    (output / "artifact_sha256.txt").write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in artifact_paths), encoding="utf-8"
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
