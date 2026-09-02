#!/usr/bin/env python3
"""Fail-closed launcher for E-D12-6K-VOPD-001 after the Day 11 gate."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.monitor_vopd_training import (
    load_policy, monitor_process, query_gpus, read_cgroup,
    terminate_process_group, utc_now, write_json,
)
from scripts.vopd_training_preflight import validate_config

GATE_RELATIVE = Path("artifacts/runs/E-D11-6K-GATE-001/preflight.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def parse_time(value: str) -> dt.datetime:
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("billing timestamp must include timezone")
    return parsed.astimezone(dt.timezone.utc)


def static_preflight(config_path: Path, policy_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    policy = load_policy(policy_path)
    base = validate_config(config_path, PROJECT_ROOT)
    gate_path = PROJECT_ROOT / GATE_RELATIVE
    gate = None
    gate_error = None
    try:
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        gate_error = repr(exc)
    config_hash = sha256_file(config_path)
    checks = {
        "experiment_identity": config.get("experiment", {}).get("id")
        == policy.get("experiment_id") == "E-D12-6K-VOPD-001",
        "config_status_ready": config.get("status") == "ready_after_day11_gate",
        "training_preflight_pass": base["status"] == "PASS",
        "day11_gate_pass": gate is not None and gate.get("status") == "PASS",
        "day11_config_hash_matches": gate is not None and gate.get("config_sha256") == config_hash,
        "day11_prompt_length_pass": gate is not None and gate.get("prompt_length_status") == "PASS",
        "day11_overlap_complete": gate is not None and gate.get("overlap_status") == "PASS",
        "day11_cached_prefix_pass": gate is not None and gate.get("cached_prefix_status") == "PASS",
        "day11_pilot_pass": gate is not None and gate.get("pilot_status") == "PASS",
    }
    return {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "config": str(config_path),
        "config_sha256": config_hash,
        "policy": str(policy_path),
        "policy_sha256": sha256_file(policy_path),
        "day11_gate": str(gate_path),
        "day11_gate_error": gate_error,
        "training_preflight": base,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/vopd_6241.yaml"))
    parser.add_argument("--policy", type=Path, default=Path("configs/vopd_6241_abort_policy.yaml"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--current-autodl-cost-cny", type=float)
    parser.add_argument("--billing-observed-at-utc")
    parser.add_argument("overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    config_path = resolve(args.config)
    policy_path = resolve(args.policy)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    policy = load_policy(policy_path)
    output_dir = resolve(config["paths"]["output_dir"])
    preflight_dir = output_dir / "preflight"
    result = static_preflight(config_path, policy_path)

    if not args.run:
        write_json(preflight_dir / "guarded_launcher_preflight.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print("No GPU training started.")
        return 0 if result["status"] == "PASS" else 1

    if args.current_autodl_cost_cny is None or not args.billing_observed_at_utc:
        parser.error("--run requires fresh billing amount and timestamp")
    observed = parse_time(args.billing_observed_at_utc)
    age = (dt.datetime.now(dt.timezone.utc) - observed).total_seconds()
    projected = args.current_autodl_cost_cny + float(policy["budget"]["conservative_reservation_cny"])
    disk = shutil.disk_usage(output_dir)
    collisions = [str(path) for name in ("logs", "rollouts", "checkpoints") for path in (output_dir / name).rglob("*") if path.is_file()]
    live_checks = {
        "static_preflight_pass": result["status"] == "PASS",
        "billing_fresh": -300 <= age <= int(policy["budget"]["billing_observation_max_age_seconds"]),
        "budget_below_hard_limit": projected <= float(policy["budget"]["project_hard_limit_cny"]),
        "storage_pass": disk.free >= int(policy["disk"]["prelaunch_required_bytes"]),
        "no_output_collision": not collisions,
        "expected_gpu_count": len(query_gpus()) == int(config["resources"]["gpus_per_node"]),
        "cgroup_readable": bool(read_cgroup(os.getpid()).get("supported")),
        "git_clean": not subprocess.run(["git", "status", "--porcelain"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True).stdout.strip(),
    }
    result["live_checks"] = live_checks
    result["billing"] = {"observed_at_utc": observed.isoformat(), "age_seconds": age, "projected_total_cny": projected}
    result["disk_free_bytes"] = disk.free
    result["collisions"] = collisions
    result["status"] = "PASS" if all(live_checks.values()) else "FAIL"
    write_json(preflight_dir / "live_launch_gate.json", result)
    if result["status"] != "PASS":
        print(json.dumps(result, ensure_ascii=False, indent=2), file=sys.stderr)
        return 41

    env = os.environ.copy()
    env["VOPD_GUARD_ACTIVE"] = "1"
    command = ["bash", str(PROJECT_ROOT / "scripts/run_vopd_2gpu.sh"), "--config", str(config_path), "--run", *args.overrides]
    process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=env, start_new_session=True)
    try:
        exit_code, summary = monitor_process(process, output_dir, policy, output_dir / "logs/train.log")
    except BaseException as exc:
        termination = terminate_process_group(process, float(policy["runtime"]["terminate_grace_seconds"])) if process.poll() is None else None
        exit_code = 40
        summary = {"schema_version": 1, "status": "FAIL", "trigger": {"rule": "guard_monitor_exception", "detail": repr(exc)}, "termination": termination}
        write_json(output_dir / "evidence/guard_summary.json", summary)
    write_json(output_dir / "evidence/exit_receipt.json", {"schema_version": 1, "guard_exit_code": exit_code, "training_summary": summary})
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
