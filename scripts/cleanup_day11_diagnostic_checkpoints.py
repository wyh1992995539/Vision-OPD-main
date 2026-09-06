#!/usr/bin/env python3
"""Remove only retired Day11 diagnostic model/optimizer tensor shards."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "artifacts/runs/E-D11-6K-GATE-001"
RECEIPT = RUN_ROOT / "diagnostic_checkpoint_cleanup_20260907/receipt.json"

CHECKPOINTS = (
    RUN_ROOT / "memory_optimization/fixed_validation_v1/capture/run/checkpoints/global_step_8/actor",
    RUN_ROOT / "memory_optimization/fixed_validation_v1/fixed_baseline/run/checkpoints/global_step_8/actor",
    RUN_ROOT / "memory_optimization/fixed_validation_v1/fixed_deferred/run/checkpoints/global_step_8/actor",
    RUN_ROOT / "memory_optimization/fixed_validation_v1/pressure_v2/run/checkpoints/global_step_16/actor",
)
SHARDS = (
    "model_world_size_2_rank_0.pt",
    "model_world_size_2_rank_1.pt",
    "optim_world_size_2_rank_0.pt",
    "optim_world_size_2_rank_1.pt",
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def targets() -> list[Path]:
    return [directory / name for directory in CHECKPOINTS for name in SHARDS]


def metadata(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "allocated_bytes": stat.st_blocks * 512,
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "mtime_ns": stat.st_mtime_ns,
        "nlink": stat.st_nlink,
    }


def validate(selected: list[Path]) -> list[dict]:
    if len(selected) != 16 or len(set(selected)) != 16:
        raise ValueError("Cleanup allowlist must contain exactly 16 unique files")
    allowed_parents = {path.resolve() for path in CHECKPOINTS}
    allowed_names = set(SHARDS)
    records = []
    for path in selected:
        resolved = path.resolve()
        if resolved.parent not in allowed_parents or resolved.name not in allowed_names:
            raise ValueError(f"Outside cleanup allowlist: {resolved}")
        if not resolved.is_file() or resolved.is_symlink():
            raise ValueError(f"Target is not a regular file: {resolved}")
        record = metadata(resolved)
        if record["allocated_bytes"] < 1024**3 or record["nlink"] != 1:
            raise ValueError(f"Unexpected target metadata: {record}")
        records.append(record)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    selected = targets()
    records = validate(selected)
    free_before = shutil.disk_usage(ROOT).free
    result = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "status": "VALIDATED_NOT_EXECUTED",
        "scope": "exactly 16 retired diagnostic model/optimizer shards",
        "targets": records,
        "target_allocated_bytes": sum(row["allocated_bytes"] for row in records),
        "free_before_bytes": free_before,
        "protected": [
            str(RUN_ROOT / "formal_candidate_validation_v1/run/checkpoints"),
            str(RUN_ROOT / "memory_optimization/fixed_validation_v1/capture/run/evidence/fixed_workload"),
            str(RUN_ROOT / "pilot/16/checkpoints"),
            str(RUN_ROOT / "pilot/64/checkpoints"),
        ],
        "recovery": "No backup; deleted tensor shards require rerunning their diagnostic experiment.",
    }
    if args.execute:
        if RECEIPT.exists():
            raise FileExistsError(f"Refusing to overwrite receipt: {RECEIPT}")
        for path in selected:
            os.unlink(path)
        if any(path.exists() for path in selected):
            raise RuntimeError("At least one cleanup target still exists")
        result.update(
            status="PASS",
            completed_at_utc=utc_now(),
            free_after_bytes=shutil.disk_usage(ROOT).free,
        )
        result["observed_free_bytes_increase"] = (
            result["free_after_bytes"] - result["free_before_bytes"]
        )
        RECEIPT.parent.mkdir(parents=True, exist_ok=False)
        RECEIPT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
