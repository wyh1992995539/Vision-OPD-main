#!/usr/bin/env python3
"""Validate and freeze the unified adaptive output-cap protocol without inference."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "configs/benchmark_adaptive_output_cap_protocol_v1.yaml"
FROZEN_CONFIG_SHA256 = "55ec71409ac0d285ab4647d982828f9277c6cfe5cbf518208c856490b7294759"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    config_sha = sha256_file(CONFIG)
    if config_sha != FROZEN_CONFIG_SHA256:
        raise ValueError(f"protocol config drift: expected {FROZEN_CONFIG_SHA256}, got {config_sha}")
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if config["protocol"]["execution_authorized_now"] is not False:
        raise ValueError("freeze must not authorize inference")
    if config["protocol"]["gpu_inference_started_by_freeze"] is not False:
        raise ValueError("freeze must record that no inference was started")

    verified_sources: dict[str, str] = {}
    for name, item in config["frozen_parent_protocols"].items():
        path = ROOT / item["path"]
        actual = sha256_file(path)
        if actual != item["sha256"]:
            raise ValueError(f"parent protocol hash mismatch: {name}")
        verified_sources[str(path.relative_to(ROOT))] = actual
    for role, identity in config["frozen_sources"].items():
        for name, item in identity.items():
            if not isinstance(item, dict) or "path" not in item or "sha256" not in item:
                continue
            path = ROOT / item["path"]
            actual = sha256_file(path)
            if actual != item["sha256"]:
                raise ValueError(f"frozen source hash mismatch: {role}/{name}")
            verified_sources[str(path.relative_to(ROOT))] = actual

    observed: dict[str, dict[str, int]] = {}
    expected_groups = {"zoombench/full", "mmstar/full", "vstar/full"}
    for role in ("base", "vision_opd", "cached_prefix"):
        pred_path = ROOT / config["frozen_sources"][role]["r4_predictions"]["path"]
        score_path = ROOT / config["frozen_sources"][role]["r4_scores"]["path"]
        predictions = read_jsonl(pred_path)
        scores = read_jsonl(score_path)
        if len(predictions) != 2536 or len(scores) != 2536:
            raise ValueError(f"unexpected R4 row count for {role}")
        if len({row["sample_uid"] + "\0" + row["benchmark"] + "\0" + row["view"] for row in predictions}) != 2536:
            raise ValueError(f"duplicate R4 prediction key for {role}")
        groups = {f"{row['benchmark']}/{row['view']}" for row in predictions}
        if groups != expected_groups:
            raise ValueError(f"unexpected benchmark groups for {role}: {groups}")
        counts = Counter(
            f"{row['benchmark']}/{row['view']}"
            for row in predictions
            if str(row.get("finish_reason") or "").casefold() == "length"
        )
        observed[role] = {group: counts[group] for group in sorted(expected_groups)}
        expected = config["observed_r4_length_counts"][role]
        if observed[role] != expected:
            raise ValueError(f"length-count drift for {role}: {observed[role]} != {expected}")

    closure_path = ROOT / config["frozen_sources"]["base"]["mmstar_terminal_closure"]["path"]
    closure = json.loads(closure_path.read_text(encoding="utf-8"))
    if closure.get("status") != "COMPLETE" or closure.get("correct") != 1091:
        raise ValueError("Base MMStar terminal closure is not complete at 1091/1500")
    if closure.get("unresolved_output_cap_count") != 0:
        raise ValueError("Base MMStar closure still has unresolved output-cap samples")

    future_root = ROOT / config["paths"]["future_run_root"]
    if future_root.exists() and any(future_root.iterdir()):
        raise ValueError("future adaptive run root already contains files; freeze cannot claim not started")
    freeze_root = ROOT / config["paths"]["freeze_root"]
    freeze_root.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(CONFIG, freeze_root / "config_snapshot.yaml")
    deferred = {
        "schema_version": 1,
        "protocol_id": config["protocol"]["id"],
        "execution_authorized_now": False,
        "inference_started": False,
        "gpu_used": False,
        "base": config["execution_state"]["base"],
        "vision_opd": config["execution_state"]["vision_opd"],
        "cached_prefix": config["execution_state"]["cached_prefix"],
        "observed_r4_length_counts": observed,
        "future_run_root": config["paths"]["future_run_root"],
    }
    write_json(freeze_root / "deferred_execution.json", deferred)
    receipt = {
        "schema_version": 1,
        "experiment_id": config["protocol"]["freeze_experiment_id"],
        "protocol_id": config["protocol"]["id"],
        "status": "PASS",
        "decision": "UNIFIED_RULE_FROZEN_PENDING_MODELS_NOT_STARTED",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(CONFIG.relative_to(ROOT)),
        "config_sha256": config_sha,
        "runner_sha256": sha256_file(Path(__file__)),
        "verified_source_sha256": verified_sources,
        "observed_r4_length_counts": observed,
        "base_mmstar_final": {"correct": 1091, "total": 1500, "accuracy": 1091 / 1500},
        "vision_opd_execution_status": "DEFERRED_BY_USER_NOT_STARTED",
        "cached_prefix_execution_status": "DEFERRED_BY_USER_NOT_STARTED",
        "inference_started": False,
        "gpu_used": False,
        "source_files_unchanged": True,
    }
    write_json(freeze_root / "protocol_freeze_receipt.json", receipt)
    artifacts = sorted(path for path in freeze_root.iterdir() if path.name != "artifact_sha256.txt")
    (freeze_root / "artifact_sha256.txt").write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in artifacts), encoding="utf-8"
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
