#!/usr/bin/env python3
"""Day12 launcher: preserved Day11 evidence and resource guards, optional accounting.

The original launcher and policy stay immutable because the promotion receipt
binds their hashes. This entrypoint supersedes their live billing requirements,
not their training, checkpoint, evidence or resource contracts.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_vopd_6241_guarded as frozen

CONFIG = ROOT / "configs/vopd_6241.yaml"
POLICY = ROOT / "configs/vopd_6241_abort_policy.yaml"
OPERATIONS = ROOT / "configs/day12_operations.yaml"


def cost_estimate(selected: dict, operations: dict) -> dict:
    if (operations.get("experiment_id") != "E-D12-6K-VOPD-001"
            or operations.get("billing_mode") != "estimate_only"
            or operations.get("require_billing_observation") is not False
            or operations.get("enforce_cumulative_cost_gate") is not False):
        raise ValueError("Invalid Day12 accounting amendment")
    rate = float(operations["hourly_dual_gpu_rate_cny"])
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("Dual GPU hourly rate must be finite and positive")
    result = dict(operations)
    result["is_provider_bill"] = False
    for name in ("planning", "reservation", "hard_abort_ceiling"):
        hours = float(selected[f"{name}_hours"])
        if not math.isfinite(hours) or hours <= 0:
            raise ValueError("Candidate duration must be finite and positive")
        result[f"{name}_hours"] = hours
        result[f"{name}_incremental_cost_cny"] = hours * rate
    return result


def preflight() -> dict:
    result = frozen.static_preflight(CONFIG, POLICY)
    gate_path = ROOT / frozen.GATE_RELATIVE
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    operations = yaml.safe_load(OPERATIONS.read_text(encoding="utf-8"))
    result["historical_budget_reservation_cny"] = result.pop("budget_reservation_cny")
    result["accounting"] = cost_estimate(gate["budget"]["selected"], operations)
    result["budget_reservation_cny"] = result["accounting"]["reservation_incremental_cost_cny"]
    result["operational_sources"] = {
        name: {"path": str(path), "sha256": frozen.sha256_file(path)}
        for name, path in {
            "launcher": Path(__file__).resolve(),
            "operations": OPERATIONS,
            "day11_gate": gate_path,
            "runtime_monitor": ROOT / "scripts/monitor_vopd_training.py",
        }.items()
    }
    result["billing_note"] = (
        "Day11 budget checks describe historical evidence. Day12 uses the current "
        "dual-GPU rate for estimates only; no fresh bill or cumulative-cost gate."
    )
    result["training_started"] = False
    return result


def live_gate(result: dict, config: dict, policy: dict, output_dir: Path) -> dict:
    required_disk = int(result["disk_required_bytes"])
    disk = shutil.disk_usage(output_dir)
    collisions = [str(path) for name in ("logs", "rollouts", "checkpoints", "evidence")
                  for path in (output_dir / name).rglob("*") if path.is_file()]
    cgroup = frozen.read_cgroup(os.getpid())
    gpus = frozen.query_gpus()
    checks = {
        "static_preflight_pass": result["status"] == "PASS",
        "storage_pass": disk.free >= required_disk,
        "no_output_collision": not collisions,
        "expected_gpu_count": len(gpus) == int(config["resources"]["gpus_per_node"]),
        "gpus_idle": bool(gpus) and all(
            gpu["memory_total_bytes"] > 0
            and 0 <= gpu["memory_used_bytes"] <= 0.10 * gpu["memory_total_bytes"]
            for gpu in gpus
        ),
        "cgroup_readable": bool(cgroup.get("supported")),
        "cgroup_capacity_pass": frozen.cgroup_has_minimum_capacity(
            cgroup, int(policy["memory"]["prelaunch_cgroup_minimum_bytes"])),
        "git_clean": not subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, check=True,
            capture_output=True, text=True).stdout.strip(),
    }
    return {**result, "live_checks": checks,
            "status": "PASS" if all(checks.values()) else "FAIL",
            "disk_free_bytes": disk.free, "live_disk_required_bytes": required_disk,
            "cgroup": cgroup, "gpus": gpus, "collisions": collisions}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    policy = frozen.load_policy(POLICY)
    output_dir = frozen.resolve(config["paths"]["output_dir"])
    result = preflight()
    frozen.write_json(output_dir / "preflight/day12_operations_preflight.json", result)
    if not args.run or result["status"] != "PASS":
        print(json.dumps({"status": result["status"], "checks": result["checks"],
                          "accounting": result["accounting"], "training_started": False},
                         ensure_ascii=False, indent=2))
        return (41 if args.run else 1) if result["status"] != "PASS" else 0

    result = live_gate(result, config, policy, output_dir)
    frozen.write_json(output_dir / "preflight/day12_live_launch_gate.json", result)
    if result["status"] != "PASS":
        print(json.dumps(result["live_checks"], indent=2), file=sys.stderr)
        return 41

    env = os.environ.copy()
    env["VOPD_GUARD_ACTIVE"] = "1"
    command = ["bash", str(ROOT / "scripts/run_vopd_2gpu.sh"),
               "--config", str(CONFIG), "--run"]
    started_at = frozen.utc_now()
    started = time.monotonic()
    process = subprocess.Popen(command, cwd=ROOT, env=env, start_new_session=True)
    try:
        exit_code, summary = frozen.monitor_process(
            process, output_dir, policy, output_dir / "logs/train.log")
    except BaseException as exc:
        termination = frozen.terminate_process_group(
            process, float(policy["runtime"]["terminate_grace_seconds"])
        ) if process.poll() is None else None
        exit_code = 40
        summary = {"schema_version": 1, "status": "FAIL",
                   "trigger": {"rule": "guard_monitor_exception", "detail": repr(exc)},
                   "termination": termination}
        frozen.write_json(output_dir / "evidence/guard_summary.json", summary)
    elapsed = time.monotonic() - started
    accounting = {
        "started_at_utc": started_at, "finished_at_utc": frozen.utc_now(),
        "elapsed_seconds": elapsed,
        "hourly_dual_gpu_rate_cny": result["accounting"]["hourly_dual_gpu_rate_cny"],
        "estimated_incremental_cost_cny": elapsed / 3600 * result["accounting"]["hourly_dual_gpu_rate_cny"],
        "is_provider_bill": False,
        "scope": "Launcher runtime only; excludes instance idle time and other jobs.",
    }
    frozen.write_json(output_dir / "evidence/exit_receipt.json", {
        "schema_version": 1, "guard_exit_code": exit_code,
        "training_summary": summary, "accounting": accounting,
        "operational_sources": result["operational_sources"],
    })
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
