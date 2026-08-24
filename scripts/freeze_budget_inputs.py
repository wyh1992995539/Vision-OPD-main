#!/usr/bin/env python3
"""Freeze the measured inputs needed to budget the full Day 6 benchmark run."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_path(value: str, repo_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (repo_root / path).resolve()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def gpu_inventory() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, text=True, capture_output=True, check=True, timeout=15)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"status": "unavailable_at_freeze_time", "error": f"{type(exc).__name__}: {exc}"}
    gpus = []
    for line in result.stdout.splitlines():
        values = [item.strip() for item in line.split(",")]
        if len(values) != 5:
            continue
        index, name, uuid, driver_version, memory_total_mib = values
        gpus.append(
            {
                "index": int(index),
                "name": name,
                "uuid": uuid,
                "driver_version": driver_version,
                "memory_total_mib": int(memory_total_mib),
            }
        )
    return {"status": "captured", "gpus": gpus}


def prediction_observations(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["benchmark"])].append(row)
    by_benchmark = {}
    for benchmark, items in sorted(grouped.items()):
        count = len(items)
        by_benchmark[benchmark] = {
            "request_count": count,
            "mean_latency_seconds": sum(float(item["latency_seconds"]) for item in items) / count,
            "prompt_tokens": sum(int(item.get("prompt_tokens") or 0) for item in items),
            "completion_tokens": sum(int(item.get("completion_tokens") or 0) for item in items),
            "mean_prompt_tokens": sum(int(item.get("prompt_tokens") or 0) for item in items) / count,
            "mean_completion_tokens": sum(int(item.get("completion_tokens") or 0) for item in items) / count,
        }
    timestamps = sorted(str(item["generated_at_utc"]) for item in rows)
    completion_span = None
    if timestamps:
        start = datetime.fromisoformat(timestamps[0].replace("Z", "+00:00"))
        end = datetime.fromisoformat(timestamps[-1].replace("Z", "+00:00"))
        completion_span = (end - start).total_seconds()
    return {
        "request_count": len(rows),
        "unique_request_key_count": len({(x["benchmark"], x["view"], x["sample_uid"]) for x in rows}),
        "inference_error_count": sum(bool(item.get("error")) for item in rows),
        "sum_request_latency_seconds": sum(float(item["latency_seconds"]) for item in rows),
        "mean_request_latency_seconds": sum(float(item["latency_seconds"]) for item in rows) / len(rows),
        "completion_timestamp_first_utc": timestamps[0] if timestamps else None,
        "completion_timestamp_last_utc": timestamps[-1] if timestamps else None,
        "completion_span_seconds": completion_span,
        "by_benchmark": by_benchmark,
    }


def full_workload(dataset_manifest: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"by_benchmark": {}}
    total_requests = 0
    max_judge_instances = 0
    for benchmark, item in sorted(dataset_manifest["benchmarks"].items()):
        converted = Path(item["converted_json"])
        rows = json.loads(converted.read_text(encoding="utf-8"))
        formats = Counter(str(row.get("question_format") or "") for row in rows)
        request_multiplier = 2 if benchmark == "zoombench" else 1
        request_count = len(rows) * request_multiplier
        open_question_instances = formats.get("open_question", 0) * request_multiplier
        result["by_benchmark"][benchmark] = {
            "converted_json": str(converted),
            "converted_json_sha256": sha256_file(converted),
            "sample_count": len(rows),
            "question_format_counts": dict(sorted(formats.items())),
            "request_multiplier": request_multiplier,
            "full_request_count": request_count,
            "maximum_semantic_judge_instances": open_question_instances,
        }
        total_requests += request_count
        max_judge_instances += open_question_instances
    result["full_request_count"] = total_requests
    result["maximum_semantic_judge_instances"] = max_judge_instances
    return result


def build_budget_inputs(config_path: Path, *, dual_gpu_hourly_cny: float | None, price_source: str | None) -> dict[str, Any]:
    config_path = config_path.resolve()
    repo_root = config_path.parent.parent
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    paths = config["paths"]
    run_root = resolve_path(paths["run_root"], repo_root)
    smoke_dir = run_root / "smoke" / "base"
    predictions_path = smoke_dir / "predictions.jsonl"
    summary_path = smoke_dir / "summary.json"
    selection_path = resolve_path(config["smoke"]["manifest"], repo_root)
    dataset_manifest_path = resolve_path(paths["dataset_manifest"], repo_root)
    prediction_rows = load_jsonl(predictions_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["decision_status"] != "complete" or summary["request_count"] != 64:
        raise ValueError("Smoke must be complete with exactly 64 requests before budgeting")

    pricing_status = "provided" if dual_gpu_hourly_cny is not None else "required_before_budget_approval"
    return {
        "schema_version": 1,
        "purpose": "day5_task6_full_external_evaluation_budget_inputs",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": config["protocol"]["experiment_id"],
        "protocol_revision": config["protocol"]["protocol_revision"],
        "inputs": {
            "config": {
                "path": str(config_path),
                "sha256": sha256_file(config_path),
                "model_under_test": config["model_under_test"],
                "serving": config["serving"],
                "judge": config["judge"],
            },
            "selection_manifest": {
                "path": str(selection_path),
                "sha256": sha256_file(selection_path),
                "sample_uid_count": sum(
                    len(value["sample_uids"]) for value in json.loads(selection_path.read_text(encoding="utf-8"))["benchmarks"].values()
                ),
            },
            "smoke_artifacts": {
                "predictions_path": str(predictions_path),
                "predictions_sha256": sha256_file(predictions_path),
                "summary_path": str(summary_path),
                "summary_sha256": sha256_file(summary_path),
                "summary": summary,
                "observations": prediction_observations(prediction_rows),
            },
            "dataset_manifest": {
                "path": str(dataset_manifest_path),
                "sha256": sha256_file(dataset_manifest_path),
                "workload": full_workload(json.loads(dataset_manifest_path.read_text(encoding="utf-8"))),
            },
            "hardware": gpu_inventory(),
            "pricing": {
                "dual_gpu_hourly_cny": dual_gpu_hourly_cny,
                "source": price_source,
                "status": pricing_status,
                "note": "Set from the AutoDL console before budget approval; do not infer a price from Smoke.",
            },
        },
    }


def write_frozen(path: Path, document: dict[str, Any], *, force: bool) -> Path:
    if path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite frozen input: {path}; use --force only for a dated amendment")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    hash_path = path.with_suffix(".sha256")
    hash_path.write_text(f"{sha256_file(path)}  {path}\n", encoding="utf-8")
    return hash_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/benchmark_eval.yaml")
    parser.add_argument("--dual-gpu-hourly-cny", type=float)
    parser.add_argument("--price-source")
    parser.add_argument("--output")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    repo_root = config_path.resolve().parent.parent
    output = Path(args.output).resolve() if args.output else resolve_path(
        config["paths"]["run_root"], repo_root
    ) / "budget_inputs.json"
    document = build_budget_inputs(
        config_path,
        dual_gpu_hourly_cny=args.dual_gpu_hourly_cny,
        price_source=args.price_source,
    )
    hash_path = write_frozen(output, document, force=args.force)
    print(json.dumps({"budget_inputs": str(output), "sha256_file": str(hash_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

