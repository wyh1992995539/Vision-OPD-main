#!/usr/bin/env python3
"""Verify the frozen Day 5 inputs required by the Day 6 launch gate."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml


EXPECTED = {
    "configs/benchmark_eval.yaml": "d50d420d760fa59bd8a139fa4615aed8a4b41c79ca969d5f194e95c2ad6c25b6",
    "artifacts/runs/E-D5-001/smoke_selection.json": "dc5856cf6563e5b4a341f5131fcb33785ea36efd3c4ac7f239aebb428e0a392b",
    "artifacts/runs/E-D5-001/budget_inputs.json": "9f03f8c83108319e248753442ac080f2385dd2e70015fee7acea9ab39bbefe61",
    "artifacts/runs/E-D5-001/cost.json": "13fb707e790850b6c75e549db642689dfdb837396336b503096f7dc05744456d",
}

MODEL_SHARDS = {
    "/root/autodl-tmp/models/Qwen3.5-4B/model.safetensors-00001-of-00002.safetensors":
        "26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61",
    "/root/autodl-tmp/models/Qwen3.5-4B/model.safetensors-00002-of-00002.safetensors":
        "cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check(path: Path, expected: str) -> dict:
    actual = sha256(path) if path.is_file() else None
    return {
        "path": str(path),
        "expected_sha256_raw_bytes": expected,
        "actual_sha256_raw_bytes": actual,
        "status": "pass" if actual == expected else "fail",
    }


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    checks = [check(repo / relative, expected) for relative, expected in EXPECTED.items()]
    checks.extend(check(Path(path), expected) for path, expected in MODEL_SHARDS.items())
    budget_hashes = (repo / "artifacts/runs/E-D5-001/budget_artifacts.sha256").read_text(encoding="utf-8")
    checks.append({
        "path": "artifacts/runs/E-D5-001/budget_artifacts.sha256",
        "status": "pass" if EXPECTED["artifacts/runs/E-D5-001/cost.json"] in budget_hashes else "fail",
        "requirement": "contains frozen cost.json SHA256",
    })
    budget = json.loads((repo / "artifacts/runs/E-D5-001/cost.json").read_text(encoding="utf-8"))
    checks.append({
        "path": "artifacts/runs/E-D5-001/cost.json",
        "field": "pricing.dual_gpu_hourly_cny",
        "actual": budget["pricing"]["dual_gpu_hourly_cny"],
        "expected": 11.96,
        "status": "pass" if budget["pricing"]["dual_gpu_hourly_cny"] == 11.96 else "fail",
    })
    config = yaml.safe_load((repo / "configs/benchmark_eval.yaml").read_text(encoding="utf-8"))
    checks.append({
        "path": "configs/benchmark_eval.yaml",
        "field": "protocol.protocol_revision",
        "actual": config["protocol"]["protocol_revision"],
        "expected": 4,
        "status": "pass" if config["protocol"]["protocol_revision"] == 4 else "fail",
    })
    frozen_inputs = json.loads(
        (repo / "artifacts/runs/E-D5-001/budget_inputs.json").read_text(encoding="utf-8")
    )
    for benchmark in ("zoombench", "mmstar", "vstar"):
        item = frozen_inputs["inputs"]["dataset_manifest"]["workload"]["by_benchmark"][benchmark]
        checks.append(check(Path(item["converted_json"]), item["converted_json_sha256"]))
    result = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "status": "pass" if all(item["status"] == "pass" for item in checks) else "fail",
    }
    output = repo / "artifacts/runs/E-D6-001/preflight/frozen_inputs_verification.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    raise SystemExit(0 if result["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
