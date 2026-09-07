#!/usr/bin/env python3
"""Fail-closed Day 14 launcher for Cached Prefix 6,241 formal training."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.monitor_vopd_training import (
    cgroup_has_minimum_capacity, load_policy, monitor_process, query_gpus, read_cgroup,
    terminate_process_group, utc_now, write_json,
)
from scripts.promote_cached_prefix_6241 import RECEIPT, verify_receipt
from scripts.vopd_training_preflight import validate_config

CONFIG = ROOT / "configs/cached_prefix_6241.yaml"
POLICY = ROOT / "configs/cached_prefix_6241_abort_policy.yaml"
OPERATIONS = ROOT / "configs/day14_cached_operations.yaml"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def port_is_listening(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.25)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def accounting(policy: dict[str, Any], operations: dict[str, Any]) -> dict[str, Any]:
    rate = float(operations.get("hourly_dual_gpu_rate_cny", 0))
    hours = float(policy.get("runtime", {}).get("max_wall_time_hours", 0))
    valid = (
        operations.get("schema_version") == 1
        and operations.get("experiment_id") == "E-D14-6K-CACHED-001"
        and operations.get("billing_mode") == "estimate_only"
        and operations.get("require_billing_observation") is False
        and operations.get("enforce_cumulative_cost_gate") is False
        and math.isfinite(rate) and rate > 0
        and math.isfinite(hours) and hours > 0
    )
    if not valid:
        raise ValueError("Invalid Day 14 accounting contract")
    return {
        **operations,
        "maximum_guarded_hours": hours,
        "maximum_estimated_incremental_cost_cny": rate * hours,
        "is_provider_bill": False,
    }


def scheduled_checkpoints(config: dict[str, Any]) -> list[int]:
    total = int(config["training"]["total_optimizer_steps"])
    frequency = int(config["training"]["save_frequency"])
    values = list(range(frequency, total, frequency)) if frequency > 0 else []
    if not values or values[-1] != total:
        values.append(total)
    return values


def static_preflight() -> dict[str, Any]:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    policy = load_policy(POLICY)
    operations = yaml.safe_load(OPERATIONS.read_text(encoding="utf-8"))
    base = validate_config(CONFIG, ROOT)
    promotion = json.loads(RECEIPT.read_text(encoding="utf-8")) if RECEIPT.is_file() else {}
    config_hash = sha256_file(CONFIG)
    policy_hash = sha256_file(POLICY)
    checkpoints = scheduled_checkpoints(config)
    costs = accounting(policy, operations)
    checks = {
        "experiment_identity": (
            config.get("experiment", {}).get("id")
            == policy.get("experiment_id")
            == operations.get("experiment_id")
            == "E-D14-6K-CACHED-001"
        ),
        "config_status_ready": config.get("status") == "ready_after_day13_gate",
        "formal_training_authorized": config.get("promotion", {}).get("formal_training_authorized") is True,
        "promotion_receipt_valid": verify_receipt(RECEIPT, CONFIG),
        "receipt_config_hash_matches": promotion.get("promoted_formal_config", {}).get("sha256") == config_hash,
        "receipt_policy_hash_matches": promotion.get("formal_policy", {}).get("sha256") == policy_hash,
        "strict_prefix_source_ablation_pass": promotion.get("eligibility", {}).get("strict_prefix_source_ablation") == "PASS",
        "training_preflight_pass": base.get("status") == "PASS",
        "cached_prefix_active": config.get("experiment", {}).get("prefix_source") == "cached",
        "starts_from_frozen_base": config.get("paths", {}).get("model") == "/root/autodl-tmp/models/Qwen3.5-4B",
        "resume_disabled": config.get("training", {}).get("resume_mode") == "disable",
        "formal_780_step_contract": (
            config["data"]["expected_train_rows"] == 6241
            and config["training"]["expected_samples"] == 6240
            and config["training"]["padding_rows"] == 0
            and config["training"]["dropped_rows"] == 1
            and config["training"]["total_optimizer_steps"] == 780
        ),
        "checkpoint_schedule_matches": checkpoints == policy.get("checkpoint", {}).get("allowed_save_steps"),
        "checkpoint_final_step_matches": policy.get("checkpoint", {}).get("expected_final_step") == 780,
        "cached_runtime_guard_enabled": (
            policy.get("metrics", {}).get("cached_prefix_required") is True
            and policy["metrics"].get("cached_expected_records_per_step") == 8
            and policy["metrics"].get("cached_online_generation_calls_max") == 0
        ),
        "estimate_only_accounting": (
            costs["hourly_dual_gpu_rate_cny"] == 14.0
            and costs["require_billing_observation"] is False
            and costs["enforce_cumulative_cost_gate"] is False
        ),
        "disk_floor_is_130_gib": policy.get("disk", {}).get("prelaunch_required_bytes") == 130 * 1024**3,
        "cgroup_floor_is_240_gib": policy.get("memory", {}).get("prelaunch_cgroup_minimum_bytes") == 240 * 1024**3,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    return {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "status": "PASS" if not failed else "FAIL",
        "training_started": False,
        "gpu_used": False,
        "checks": checks,
        "failed_checks": failed,
        "config": str(CONFIG),
        "config_sha256": config_hash,
        "policy": str(POLICY),
        "policy_sha256": policy_hash,
        "promotion_receipt": str(RECEIPT),
        "promotion_receipt_sha256": sha256_file(RECEIPT) if RECEIPT.is_file() else None,
        "training_preflight": base,
        "accounting": costs,
        "checkpoint_schedule": checkpoints,
        "disk_required_bytes": int(policy["disk"]["prelaunch_required_bytes"]),
        "cgroup_required_bytes": int(policy["memory"]["prelaunch_cgroup_minimum_bytes"]),
    }


def output_collisions(output_dir: Path) -> list[str]:
    return sorted(
        str(path)
        for name in ("logs", "rollouts", "checkpoints", "evidence")
        for path in (output_dir / name).rglob("*")
        if path.is_file()
    )


def live_gate(result: dict[str, Any]) -> dict[str, Any]:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    policy = load_policy(POLICY)
    output_dir = Path(config["paths"]["output_dir"])
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    gpus: list[dict[str, Any]] = []
    gpu_error = None
    try:
        gpus = query_gpus()
    except Exception as exc:
        gpu_error = repr(exc)
    cgroup = read_cgroup(os.getpid())
    disk = shutil.disk_usage(output_dir)
    collisions = output_collisions(output_dir)
    expected_gpu_count = int(config["resources"]["gpus_per_node"])
    checks = {
        "static_preflight_pass": result.get("status") == "PASS",
        "promotion_still_valid": verify_receipt(RECEIPT, CONFIG),
        "storage_pass": disk.free >= int(policy["disk"]["prelaunch_required_bytes"]),
        "no_output_collision": not collisions,
        "expected_gpu_count": len(gpus) == expected_gpu_count == 2,
        "gpus_initially_free": len(gpus) == expected_gpu_count and all(
            gpu["memory_total_bytes"] > 0
            and gpu["memory_used_bytes"] / gpu["memory_total_bytes"]
            <= float(policy["memory"]["prelaunch_gpu_used_ratio_max"])
            for gpu in gpus
        ),
        "cgroup_readable": bool(cgroup.get("supported")),
        "cgroup_capacity_pass": cgroup_has_minimum_capacity(
            cgroup, int(policy["memory"]["prelaunch_cgroup_minimum_bytes"])
        ),
        "standalone_vllm_port_free": not port_is_listening(8000),
        "billing_observation_not_required": True,
        "cumulative_cost_gate_disabled": True,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    return {
        **result,
        "generated_at_utc": utc_now(),
        "status": "PASS" if not failed else "FAIL",
        "training_started": False,
        "gpu_used": False,
        "live_checks": checks,
        "failed_live_checks": failed,
        "gpus": gpus,
        "gpu_query_error": gpu_error,
        "cgroup": cgroup,
        "disk_free_bytes": disk.free,
        "collisions": collisions,
    }


def git_metadata() -> dict[str, str]:
    def command(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(ROOT), *args], check=True, capture_output=True, text=True
        ).stdout.strip()
    return {"commit": command("rev-parse", "HEAD"), "status_porcelain": command("status", "--porcelain")}


def run_training(result: dict[str, Any]) -> int:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    policy = load_policy(POLICY)
    output_dir = Path(config["paths"]["output_dir"])
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    invocation = {
        "schema_version": 1,
        "experiment_id": "E-D14-6K-CACHED-001",
        "started_at_utc": utc_now(),
        "command": ["bash", "scripts/run_cached_prefix_2gpu.sh", "--run"],
        "config": str(CONFIG),
        "config_sha256": sha256_file(CONFIG),
        "policy": str(POLICY),
        "policy_sha256": sha256_file(POLICY),
        "promotion_receipt": str(RECEIPT),
        "promotion_receipt_sha256": sha256_file(RECEIPT),
        "live_checks": result["live_checks"],
        "git": git_metadata(),
        "cuda_visible_devices": "0,1",
        "omp_num_threads": "4",
    }
    write_json(output_dir / "preflight/run_invocation.json", invocation)
    result["training_started"] = True
    result["gpu_used"] = True
    write_json(output_dir / "preflight/day14_live_launch_gate.json", result)

    env = os.environ.copy()
    env.update({
        "VOPD_GUARD_ACTIVE": "1",
        "CUDA_VISIBLE_DEVICES": "0,1",
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
    })
    command = ["bash", str(ROOT / "scripts/run_cached_prefix_2gpu.sh"), "--run"]
    started_at = utc_now()
    started = time.monotonic()
    process = subprocess.Popen(command, cwd=ROOT, env=env, start_new_session=True)
    try:
        exit_code, summary = monitor_process(
            process, output_dir, policy, output_dir / "logs/train.log"
        )
    except BaseException as exc:
        termination = terminate_process_group(
            process, float(policy["runtime"]["terminate_grace_seconds"])
        ) if process.poll() is None else None
        exit_code = 40
        summary = {
            "schema_version": 1,
            "status": "FAIL",
            "trigger": {"rule": "day14_guard_exception", "detail": repr(exc)},
            "termination": termination,
        }
        write_json(output_dir / "evidence/guard_summary.json", summary)
    elapsed = time.monotonic() - started
    rate = float(result["accounting"]["hourly_dual_gpu_rate_cny"])
    write_json(output_dir / "evidence/exit_receipt.json", {
        "schema_version": 1,
        "experiment_id": "E-D14-6K-CACHED-001",
        "guard_exit_code": exit_code,
        "training_summary": summary,
        "accounting": {
            "started_at_utc": started_at,
            "finished_at_utc": utc_now(),
            "elapsed_seconds": elapsed,
            "hourly_dual_gpu_rate_cny": rate,
            "estimated_incremental_cost_cny": elapsed / 3600 * rate,
            "is_provider_bill": False,
        },
        "operational_sources": {
            "config": {"path": str(CONFIG), "sha256": sha256_file(CONFIG)},
            "policy": {"path": str(POLICY), "sha256": sha256_file(POLICY)},
            "promotion": {"path": str(RECEIPT), "sha256": sha256_file(RECEIPT)},
            "launcher": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
        },
    })
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()

    result = static_preflight()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    output_dir = Path(config["paths"]["output_dir"])
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    write_json(output_dir / "preflight/day14_static_preflight.json", result)
    if not args.run:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print("No GPU training started.")
        return 0 if result["status"] == "PASS" else 1
    if result["status"] != "PASS":
        write_json(output_dir / "preflight/day14_live_launch_gate.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2), file=sys.stderr)
        return 41

    result = live_gate(result)
    write_json(output_dir / "preflight/day14_live_launch_gate.json", result)
    if result["status"] != "PASS":
        print(json.dumps({
            "status": result["status"],
            "failed_live_checks": result["failed_live_checks"],
            "disk_free_bytes": result["disk_free_bytes"],
            "gpu_count": len(result["gpus"]),
            "cgroup_limit_bytes": result["cgroup"].get("memory_max_bytes"),
            "training_started": False,
        }, ensure_ascii=False, indent=2), file=sys.stderr)
        return 41
    return run_training(result)


if __name__ == "__main__":
    raise SystemExit(main())
