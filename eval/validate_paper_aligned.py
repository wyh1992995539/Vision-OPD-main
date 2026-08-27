#!/usr/bin/env python3
"""Validate paper-aligned config/data and optionally a completed run directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from eval.paper_aligned_common import (
    expected_counts,
    judge_complete,
    load_config,
    load_tasks,
    now_utc,
    prediction_complete,
    read_jsonl_map,
    record_key,
    require_formal_manifest_comparable_with_base,
    require_frozen_r3_config,
    resolve_path,
    sha256_file,
    write_json,
)


def validate_data(config: dict[str, Any]) -> dict[str, Any]:
    tasks = load_tasks(config)
    counts = expected_counts(config)
    actual = {
        name: sum(f"{task['benchmark']}/{task['view']}" == name for task in tasks)
        for name in counts
    }
    if actual != counts:
        raise ValueError(f"prepared data counts differ from config: {actual} != {counts}")
    verified_primary_images = 0
    verified_crop_images = 0
    for task in tasks:
        row = task["row"]
        expected_image_hash = str(row.get("image_sha256") or "")
        if not expected_image_hash or sha256_file(task["image_path"]) != expected_image_hash:
            raise ValueError(f"image hash mismatch: {task['benchmark']}/{row['sample_uid']}")
        verified_primary_images += 1
        crop_hash = str(row.get("crop_image_sha256") or "")
        if crop_hash:
            crop_paths = row.get("crop_images") or []
            if len(crop_paths) != 1 or sha256_file(Path(str(crop_paths[0]))) != crop_hash:
                raise ValueError(f"crop image hash mismatch: {task['benchmark']}/{row['sample_uid']}")
            verified_crop_images += 1
    return {
        "status": "pass",
        "expected_counts": counts,
        "actual_counts": actual,
        "total": len(tasks),
        "verified_primary_image_sha256_count": verified_primary_images,
        "verified_crop_image_sha256_count": verified_crop_images,
        "unique_request_keys": len({
            record_key({
                "benchmark": task["benchmark"],
                "view": task["view"],
                "sample_uid": task["row"]["sample_uid"],
            })
            for task in tasks
        }),
        "dataset_json_sha256": {
            name: sha256_file(resolve_path(config["benchmarks"][name]["converted_json"]))
            for name in ("zoombench", "mmstar", "vstar")
        },
    }


def validate_run(out: Path, config: dict[str, Any]) -> dict[str, Any]:
    manifest_path = out / config["paths"]["run_manifest_name"]
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("experiment_id") != "E-PAPER-BASEJUDGE-001":
        raise ValueError("wrong experiment_id in run manifest")
    comparability = require_formal_manifest_comparable_with_base(manifest, config)
    request = manifest.get("request_contract", {})
    required_request = {
        "system_prompt": None,
        "message_roles": ["user"],
        "input_image_count": 1,
        "image_view": "full",
        "enable_thinking": False,
        "temperature": 0,
        "max_tokens": 1024,
        "forbidden_parameters": config["generation"]["forbidden_request_parameters"],
    }
    if request != required_request:
        raise ValueError(f"run request contract changed: {request}")

    predictions, prediction_stats = read_jsonl_map(
        out / config["paths"]["predictions_name"], complete=prediction_complete
    )
    scores, score_stats = read_jsonl_map(out / config["paths"]["scores_name"])
    judges, judge_stats = read_jsonl_map(
        out / config["paths"]["judge_results_name"], complete=judge_complete
    )
    expected = int(manifest["expected_request_count"])
    if len(predictions) != expected or len(scores) != expected:
        raise ValueError(
            f"run requires {expected} unique predictions and scores; "
            f"found predictions={len(predictions)} scores={len(scores)}"
        )
    required_judges = {
        key for key, score in scores.items() if bool(score.get("judge_required"))
    }
    completed_judges = {
        key for key in required_judges if judge_complete(judges.get(key, {}))
    }
    pending_scores = sum(score.get("score_status") != "scored" for score in scores.values())
    if completed_judges != required_judges or pending_scores:
        raise ValueError(
            f"run is not fully scored: Judge {len(completed_judges)}/{len(required_judges)}, "
            f"pending_scores={pending_scores}"
        )
    summary_path = out / config["paths"]["summary_name"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("decision_status") != "complete":
        raise ValueError("summary is not complete")
    if manifest["run_mode"] == "formal":
        if summary["request_count"] != 2536:
            raise ValueError("formal summary request_count must be 2536")
        if summary["groups"]["vstar/full"]["total"] != 191:
            raise ValueError("formal V* denominator must be 191")
    return {
        "status": "pass",
        "run_dir": str(out),
        "run_mode": manifest["run_mode"],
        "model_role": manifest["model_role"],
        "comparability_gate": comparability,
        "expected_request_count": expected,
        "prediction_count": len(predictions),
        "score_count": len(scores),
        "required_judge_count": len(required_judges),
        "completed_judge_count": len(completed_judges),
        "compaction_stats": {
            "predictions": prediction_stats,
            "scores": score_stats,
            "judges": judge_stats,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/benchmark_eval_paper_basejudge_r3_single_gpu.yaml")
    parser.add_argument("--run-dir")
    parser.add_argument("--skip-data", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()

    config_path, config = load_config(args.config)
    require_frozen_r3_config(config_path)
    result: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": now_utc(),
        "status": "pass",
        "config_path": str(config_path),
        "config_sha256_raw_bytes": sha256_file(config_path),
        "amendment_sha256_raw_bytes": config["protocol"]["amendment_sha256"],
        "config_contract": "pass",
    }
    if not args.skip_data:
        result["data"] = validate_data(config)
    if args.run_dir:
        result["run"] = validate_run(resolve_path(args.run_dir), config)
    output = (
        resolve_path(args.output)
        if args.output
        else resolve_path(config["paths"]["run_root"]) / "preflight" / "paper_aligned_validation.json"
    )
    write_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
