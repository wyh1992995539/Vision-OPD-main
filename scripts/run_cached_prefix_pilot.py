#!/usr/bin/env python3
"""Fail-closed Cached Prefix Pilot launcher with estimate-only accounting."""

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

from scripts import run_vopd_6241_pilot_guarded as pilot
from scripts.monitor_vopd_training import monitor_process, terminate_process_group, write_json

POLICY = ROOT / "configs/cached_prefix_6241_pilot_abort_policy.yaml"
OPERATIONS = ROOT / "configs/cached_prefix_pilot_operations.yaml"
RELOAD_CONFIG = ROOT / "configs/cached_prefix_6241_pilot_64_reload.yaml"
CONTRACT_AUDIT_SCRIPT = ROOT / "scripts/audit_cached_prefix_contract.py"
POSTFLIGHT_SCRIPT = ROOT / "scripts/audit_cached_prefix_pilot.py"


def load_operations() -> dict:
    value = yaml.safe_load(OPERATIONS.read_text(encoding="utf-8"))
    valid = (
        value.get("schema_version") == 1
        and value.get("experiment_id") == "E-D13-6K-CACHED-PILOT-001"
        and value.get("billing_mode") == "estimate_only"
        and value.get("require_billing_observation") is False
        and value.get("enforce_cumulative_cost_gate") is False
    )
    rate = float(value.get("hourly_dual_gpu_rate_cny", 0))
    if not valid or not math.isfinite(rate) or rate <= 0:
        raise ValueError("Invalid Cached Pilot accounting amendment")
    return value


def refresh_contract_audit() -> dict:
    completed = subprocess.run(
        [sys.executable, str(CONTRACT_AUDIT_SCRIPT)], cwd=ROOT,
        check=False, capture_output=True, text=True,
    )
    gate_path = ROOT / "artifacts/runs/E-D13-6K-CACHED-PILOT-001/preflight/contract_audit.json"
    if completed.returncode != 0 or not gate_path.is_file():
        raise RuntimeError(
            "Cached contract audit failed: " + (completed.stderr or completed.stdout)[-2000:]
        )
    return json.loads(gate_path.read_text(encoding="utf-8"))


def static_preflight() -> dict:
    audit = refresh_contract_audit()
    result = pilot.static_preflight(POLICY, "64")
    operations = load_operations()
    policy, contract = pilot.load_pilot_policy(POLICY, "64")
    extra_checks = {
        "contract_audit_static_pass": audit.get("status") == "STATIC_CONTRACT_PASS",
        "contract_audit_ready_for_pilot": audit.get("ready_for_gpu_pilot") is True,
        "contract_audit_formal_not_authorized": audit.get("formal_training_authorized") is False,
        "accounting_rate_is_dual_gpu_14_cny": float(operations["hourly_dual_gpu_rate_cny"]) == 14.0,
        "billing_observation_not_required": operations["require_billing_observation"] is False,
        "cumulative_cost_gate_disabled": operations["enforce_cumulative_cost_gate"] is False,
        "policy_postflight_is_cached_specific": policy["pilot"]["postflight_script"]
        == "scripts/audit_cached_prefix_pilot.py",
        "cold_reload_required": contract["require_cold_reload"] is True,
    }
    result["checks"].update(extra_checks)
    result["failed_checks"] = sorted(name for name, passed in result["checks"].items() if not passed)
    result["status"] = "PASS" if not result["failed_checks"] else "FAIL"
    rate = float(operations["hourly_dual_gpu_rate_cny"])
    result["accounting"] = {
        **operations,
        "maximum_guarded_hours": float(contract["max_wall_time_hours"]),
        "maximum_estimated_incremental_cost_cny": float(contract["max_wall_time_hours"]) * rate,
        "is_provider_bill": False,
    }
    result["contract_audit"] = audit
    result["operational_sources"] = {
        name: {"path": str(path), "sha256": pilot.sha256_file(path)}
        for name, path in {
            "launcher": Path(__file__).resolve(),
            "policy": POLICY,
            "operations": OPERATIONS,
            "reload_config": RELOAD_CONFIG,
            "postflight": POSTFLIGHT_SCRIPT,
        }.items()
    }
    result["training_started"] = False
    return result


def live_gate(result: dict) -> dict:
    policy, contract = pilot.load_pilot_policy(POLICY, "64")
    output_dir = pilot.resolve(contract["output_dir"])
    config = yaml.safe_load(pilot.resolve(contract["config"]).read_text(encoding="utf-8"))
    gpus = []
    gpu_error = None
    try:
        gpus = pilot.query_gpus()
    except Exception as exc:
        gpu_error = repr(exc)
    disk = shutil.disk_usage(output_dir)
    collisions = pilot.output_collisions(output_dir)
    cgroup = pilot.read_cgroup(os.getpid())
    max_initial_ratio = max(
        (
            row["memory_used_bytes"] / row["memory_total_bytes"]
            for row in gpus if row.get("memory_total_bytes")
        ),
        default=1.0,
    )
    checks = {
        "static_preflight_pass": result["status"] == "PASS",
        "storage_pass": disk.free >= int(policy["disk"]["prelaunch_required_bytes"]),
        "no_output_collision": not collisions,
        "expected_gpu_count": len(gpus) == int(config["resources"]["gpus_per_node"]) == 2,
        "gpus_initially_free": max_initial_ratio <= float(policy["memory"]["prelaunch_gpu_used_ratio_max"]),
        "cgroup_readable": bool(cgroup.get("supported")),
        "cgroup_capacity_pass": pilot.cgroup_has_minimum_capacity(
            cgroup, int(policy["memory"]["prelaunch_cgroup_minimum_bytes"])
        ),
        "standalone_vllm_port_free": not pilot.port_is_listening(8000),
        "billing_observation_not_required": True,
        "cumulative_cost_gate_disabled": True,
    }
    return {
        **result,
        "live_checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "training_started": False,
        "gpu_used": False,
        "gpus": gpus,
        "gpu_query_error": gpu_error,
        "max_initial_gpu_memory_ratio": max_initial_ratio,
        "disk_free_bytes": disk.free,
        "disk_required_bytes": int(policy["disk"]["prelaunch_required_bytes"]),
        "cgroup": cgroup,
        "collisions": collisions,
    }


def run_training(result: dict) -> int:
    policy, contract = pilot.load_pilot_policy(POLICY, "64")
    output_dir = pilot.resolve(contract["output_dir"])
    env = os.environ.copy()
    env["VOPD_GUARD_ACTIVE"] = "1"
    command = [
        "bash", str(ROOT / "scripts/run_vopd_2gpu.sh"),
        "--config", str(pilot.resolve(contract["config"])), "--run",
    ]
    started_at = pilot.utc_now()
    started = time.monotonic()
    process = subprocess.Popen(command, cwd=ROOT, env=env, start_new_session=True)
    try:
        exit_code, guard_summary = monitor_process(
            process, output_dir, policy, output_dir / "logs/train.log"
        )
    except BaseException as exc:
        termination = terminate_process_group(
            process, float(policy["runtime"]["terminate_grace_seconds"])
        ) if process.poll() is None else None
        exit_code = 40
        guard_summary = {
            "schema_version": 1,
            "status": "FAIL",
            "trigger": {"rule": "cached_pilot_guard_exception", "detail": repr(exc)},
            "termination": termination,
        }
        write_json(output_dir / "evidence/guard_summary.json", guard_summary)

    reload_exit_code = None
    reload_report = output_dir / "cold_reload/reload_validation_summary.json"
    if exit_code == 0:
        with (output_dir / "evidence/cold_reload_console.log").open("w", encoding="utf-8") as stream:
            reload_process = subprocess.run(
                [
                    sys.executable, "-m", "scripts.vopd_6241_pilot_reload",
                    "--config", str(RELOAD_CONFIG), "--run",
                ],
                cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, text=True,
            )
        reload_exit_code = reload_process.returncode
        if reload_exit_code != 0:
            exit_code = 44

    postflight_exit_code = None
    postflight = None
    if (output_dir / "logs/train.log").is_file():
        audit_command = [
            sys.executable, str(POSTFLIGHT_SCRIPT), "--stage", "64",
            "--policy", str(POLICY),
        ]
        if reload_report.is_file():
            audit_command.extend(["--reload-report", str(reload_report)])
        audit_process = subprocess.run(audit_command, cwd=ROOT, text=True)
        postflight_exit_code = audit_process.returncode
        postflight_path = output_dir / "evidence/postflight.json"
        if postflight_path.is_file():
            postflight = json.loads(postflight_path.read_text(encoding="utf-8"))
        if postflight_exit_code != 0 and exit_code == 0:
            exit_code = 43

    elapsed = time.monotonic() - started
    rate = float(result["accounting"]["hourly_dual_gpu_rate_cny"])
    write_json(output_dir / "evidence/exit_receipt.json", {
        "schema_version": 1,
        "guard_exit_code": exit_code,
        "guard_summary": guard_summary,
        "reload_exit_code": reload_exit_code,
        "postflight_exit_code": postflight_exit_code,
        "postflight_status": postflight.get("status") if postflight else None,
        "strict_prefix_source_ablation": (
            postflight.get("strict_prefix_source_ablation") if postflight else "NOT_PROVEN"
        ),
        "accounting": {
            "started_at_utc": started_at,
            "finished_at_utc": pilot.utc_now(),
            "elapsed_seconds": elapsed,
            "hourly_dual_gpu_rate_cny": rate,
            "estimated_incremental_cost_cny": elapsed / 3600 * rate,
            "is_provider_bill": False,
            "scope": "Guarded training plus automatic cold reload and postflight.",
        },
        "operational_sources": result["operational_sources"],
    })
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    result = static_preflight()
    output_dir = Path(result["output_dir"])
    write_json(output_dir / "preflight/cached_pilot_static_preflight.json", result)
    if not args.run:
        print(json.dumps({
            "status": result["status"],
            "failed_checks": result["failed_checks"],
            "accounting": result["accounting"],
            "training_started": False,
        }, ensure_ascii=False, indent=2))
        return 0 if result["status"] == "PASS" else 1
    if result["status"] != "PASS":
        return 41
    result = live_gate(result)
    write_json(output_dir / "preflight/cached_pilot_live_launch_gate.json", result)
    if result["status"] != "PASS":
        print(json.dumps({
            "status": result["status"],
            "live_checks": result["live_checks"],
            "gpu_count": len(result["gpus"]),
            "cgroup_limit_bytes": result["cgroup"].get("memory_max_bytes"),
            "disk_free_bytes": result["disk_free_bytes"],
            "training_started": False,
        }, ensure_ascii=False, indent=2), file=sys.stderr)
        return 41
    result["training_started"] = True
    write_json(output_dir / "preflight/cached_pilot_live_launch_gate.json", result)
    return run_training(result)


if __name__ == "__main__":
    raise SystemExit(main())
