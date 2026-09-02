#!/usr/bin/env python3
"""Audited, fail-closed launcher for the E-D10-001 formal training run."""

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
    load_policy,
    monitor_process,
    query_gpus,
    read_cgroup,
    terminate_process_group,
    utc_now,
    write_json,
)


EXPECTED_CONFIG_SHA256 = "5977d0b7adda448287d7410431c9461a6f6f53c04792390b9b13d9529a00b30c"
TASK4_RELATIVE = Path("artifacts/runs/E-D10-001/preflight/task4_preflight_report.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(project_root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=project_root, check=True, capture_output=True, text=True
    ).stdout.strip()


def resolve(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def parse_observed_at(value: str) -> dt.datetime:
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("billing timestamp must include a timezone")
    return parsed.astimezone(dt.timezone.utc)


def static_preflight(project_root: Path, config_path: Path, policy_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    policy = load_policy(policy_path)
    task4_path = project_root / TASK4_RELATIVE
    task4 = json.loads(task4_path.read_text(encoding="utf-8"))
    output_dir = resolve(project_root, config["paths"]["output_dir"])
    config_hash = sha256_file(config_path)
    checks = {
        "task4_pass_to_task5": task4.get("report_status") == "PASS_TO_TASK5",
        "task4_complete": task4.get("task4_completed") is True,
        "experiment_identity": config.get("experiment", {}).get("id") == policy.get("experiment_id") == "E-D10-001",
        "formal_config_hash_frozen": config_hash == EXPECTED_CONFIG_SHA256,
        "task4_config_hash_matches": task4.get("config_hashes", {}).get("task3_freeze") == config_hash,
        "dataloader_workers_zero": config.get("data", {}).get("dataloader_num_workers") == 0,
        "expected_steps_128": config.get("training", {}).get("total_optimizer_steps") == 128,
        "policy_disk_formula_valid": int(policy["disk"]["prelaunch_required_bytes"])
        == 2 * int(policy["disk"]["checkpoint_estimate_bytes"]) + int(policy["disk"]["reserve_bytes"]),
    }
    return {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "mode": "cpu_only_static_preflight",
        "gpu_used": False,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "config_path": str(config_path),
        "config_sha256": config_hash,
        "policy_path": str(policy_path),
        "policy_sha256": sha256_file(policy_path),
        "output_dir": str(output_dir),
        "historical_billing_cny": task4["time_and_cost"]["current_autodl_cumulative_charge_cny"],
        "billing_note": "Historical only; --run requires a fresh console observation.",
    }


def live_launch_preflight(
    project_root: Path,
    config_path: Path,
    policy_path: Path,
    current_cost: float,
    observed_at: str,
) -> dict[str, Any]:
    static = static_preflight(project_root, config_path, policy_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    policy = load_policy(policy_path)
    output_dir = resolve(project_root, config["paths"]["output_dir"])
    task4 = json.loads((project_root / TASK4_RELATIVE).read_text(encoding="utf-8"))
    now = dt.datetime.now(dt.timezone.utc)
    observed = parse_observed_at(observed_at)
    age_seconds = (now - observed).total_seconds()
    projected = current_cost + float(task4["time_and_cost"]["reservation_cost_cny"])
    disk = shutil.disk_usage(output_dir)
    gpus: list[dict[str, Any]] = []
    gpu_error = None
    try:
        gpus = query_gpus()
    except Exception as exc:
        gpu_error = repr(exc)
    cgroup: dict[str, Any] = {}
    cgroup_error = None
    try:
        cgroup = read_cgroup(os.getpid())
    except Exception as exc:
        cgroup_error = repr(exc)
    collision_paths = [
        str(path.relative_to(output_dir))
        for name in ("logs", "rollouts", "checkpoints")
        for path in (output_dir / name).rglob("*")
        if path.is_file()
    ]
    status_porcelain = git(project_root, "status", "--porcelain=v1")
    checks = {
        **static["checks"],
        "static_preflight_pass": static["status"] == "PASS",
        "billing_nonnegative": current_cost >= 0,
        "billing_observation_not_future": age_seconds >= -300,
        "billing_observation_within_15_minutes": -300 <= age_seconds <= 900,
        "projected_cost_below_2000_cny": projected <= 2000,
        "git_worktree_clean": not status_porcelain,
        "no_training_output_collision": not collision_paths,
        "prelaunch_storage_pass": disk.free >= int(policy["disk"]["prelaunch_required_bytes"]),
        "expected_gpu_count": len(gpus) == int(config["resources"]["gpus_per_node"]),
        "cgroup_v2_memory_readable": bool(cgroup.get("supported")),
    }
    return {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "mode": "live_launch_preflight",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "billing": {
            "current_autodl_cumulative_charge_cny": current_cost,
            "observed_at_utc": observed.isoformat(),
            "age_seconds": age_seconds,
            "reservation_cost_cny": task4["time_and_cost"]["reservation_cost_cny"],
            "projected_total_cny": projected,
        },
        "disk": {"free_bytes": disk.free, "required_bytes": policy["disk"]["prelaunch_required_bytes"]},
        "gpus": gpus,
        "gpu_error": gpu_error,
        "cgroup": cgroup,
        "cgroup_error": cgroup_error,
        "git_commit": git(project_root, "rev-parse", "HEAD"),
        "git_status_porcelain": status_porcelain.splitlines(),
        "collision_paths": collision_paths,
        "static": static,
    }


def main() -> int:
    project_root = PROJECT_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/vopd_1024.yaml"))
    parser.add_argument("--policy", type=Path, default=Path("configs/vopd_abort_policy.yaml"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--current-autodl-cost-cny", type=float)
    parser.add_argument("--billing-observed-at-utc")
    parser.add_argument("overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    config_path = resolve(project_root, str(args.config)).resolve()
    policy_path = resolve(project_root, str(args.policy)).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir = resolve(project_root, config["paths"]["output_dir"])
    preflight_dir = output_dir / "preflight"

    if not args.run:
        result = static_preflight(project_root, config_path, policy_path)
        write_json(preflight_dir / "task5_guarded_launcher_preflight.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print("No GPU training started.")
        return 0 if result["status"] == "PASS" else 1

    if args.current_autodl_cost_cny is None or not args.billing_observed_at_utc:
        parser.error("--run requires --current-autodl-cost-cny and --billing-observed-at-utc")
    result = live_launch_preflight(
        project_root,
        config_path,
        policy_path,
        args.current_autodl_cost_cny,
        args.billing_observed_at_utc,
    )
    write_json(preflight_dir / "day10_live_launch_gate.json", result)
    if result["status"] != "PASS":
        print(json.dumps(result, ensure_ascii=False, indent=2), file=sys.stderr)
        print("Formal training was not started because launch gates failed.", file=sys.stderr)
        return 41

    env = os.environ.copy()
    env["VOPD_GUARD_ACTIVE"] = "1"
    command = [
        "bash",
        str(project_root / "scripts/run_vopd_2gpu.sh"),
        "--config",
        str(config_path),
        "--run",
        *args.overrides,
    ]
    process = subprocess.Popen(command, cwd=project_root, env=env, start_new_session=True)
    try:
        exit_code, summary = monitor_process(
            process,
            output_dir,
            load_policy(policy_path),
            output_dir / "logs/train.log",
        )
    except BaseException as exc:
        termination = terminate_process_group(
            process, float(load_policy(policy_path)["runtime"]["terminate_grace_seconds"])
        ) if process.poll() is None else None
        exit_code = 40
        summary = {
            "schema_version": 1,
            "finished_at_utc": utc_now(),
            "status": "FAIL",
            "return_code": process.poll(),
            "trigger": {"rule": "guard_monitor_exception", "detail": repr(exc)},
            "termination": termination,
            "checkpoint": None,
        }
        write_json(output_dir / "evidence/guard_summary.json", summary)
    write_json(
        output_dir / "evidence/exit_receipt.json",
        {
            "schema_version": 1,
            "finished_at_utc": utc_now(),
            "guard_exit_code": exit_code,
            "training_summary": summary,
            "launch_gate": str(preflight_dir / "day10_live_launch_gate.json"),
        },
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

