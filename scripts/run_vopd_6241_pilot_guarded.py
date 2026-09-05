#!/usr/bin/env python3
"""Fail-closed launcher for the 16/64-row Vision-OPD 6K GPU Pilots."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.monitor_vopd_training import (
    cgroup_has_minimum_capacity, monitor_process, query_gpus, read_cgroup,
    terminate_process_group,
    utc_now, validate_policy, write_json,
)
from scripts.vopd_training_preflight import validate_config

DEFAULT_POLICY = ROOT / "configs/vopd_6241_pilot_abort_policy.yaml"


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_time(value: str) -> dt.datetime:
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("billing timestamp must include timezone")
    return parsed.astimezone(dt.timezone.utc)


def load_pilot_policy(path: Path, stage: str) -> tuple[dict[str, Any], dict[str, Any]]:
    source = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(source, dict) or source.get("schema_version") != 1:
        raise ValueError("Pilot policy must be a schema_version=1 mapping")
    contracts = source.get("pilot", {}).get("stage_contracts", {})
    if stage not in contracts:
        raise ValueError(f"Pilot policy has no stage {stage}")
    contract = copy.deepcopy(contracts[stage])
    effective = copy.deepcopy(source)
    effective["experiment_id"] = contract["experiment_id"]
    effective["runtime"]["max_wall_time_hours"] = contract["max_wall_time_hours"]
    effective["checkpoint"]["expected_final_step"] = contract["expected_optimizer_steps"]
    validate_policy(effective)
    return effective, contract


def port_is_listening(port: int, host: str = "127.0.0.1") -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.25):
            return True
    except OSError:
        return False


def output_collisions(output_dir: Path) -> list[str]:
    collisions: list[str] = []
    for relative in ("logs", "rollouts", "checkpoints", "evidence/telemetry"):
        root = output_dir / relative
        if root.exists():
            collisions.extend(str(path) for path in root.rglob("*") if path.is_file())
    for relative in (
        "evidence/guard_summary.json", "evidence/exit_receipt.json",
        "evidence/postflight.json", "preflight/run_invocation.json",
    ):
        path = output_dir / relative
        if path.is_file():
            collisions.append(str(path))
    return sorted(set(collisions))


def static_preflight(policy_path: Path, stage: str) -> dict[str, Any]:
    policy, contract = load_pilot_policy(policy_path, stage)
    config_path = resolve(contract["config"])
    output_dir = resolve(contract["output_dir"])
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    training = config["training"]
    base = validate_config(config_path, ROOT)
    static_gate_path = resolve(policy["pilot"]["static_gate"])
    static_gate_error = None
    try:
        static_gate = json.loads(static_gate_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        static_gate = None
        static_gate_error = repr(exc)
    prerequisite_path = (
        resolve(contract["prerequisite_postflight"])
        if contract.get("prerequisite_postflight") else None
    )
    prerequisite = None
    if prerequisite_path and prerequisite_path.is_file():
        try:
            prerequisite = json.loads(prerequisite_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            prerequisite = None
    checks = {
        "policy_stage_known": True,
        "static_gate_pass_pending_pilot": (
            static_gate is not None
            and static_gate.get("status") == policy["pilot"]["required_static_gate_status"]
            and static_gate.get("ready_for_gpu_pilot") is True
            and static_gate.get("formal_training_authorized") is False
        ),
        "training_preflight_pass": base["status"] == "PASS",
        "resource_memory_policy_matches_config": (
            config["resources"].get("prelaunch_cgroup_minimum_bytes")
            == policy["memory"]["prelaunch_cgroup_minimum_bytes"]
        ),
        "static_gate_input_hashes_current": (
            static_gate is not None
            and bool(static_gate.get("inputs"))
            and all(
                resolve(entry.get("path", "")).is_file()
                and sha256_file(resolve(entry["path"])) == entry.get("sha256")
                for entry in static_gate.get("inputs", {}).values()
            )
        ),
        "experiment_id_matches_policy": config["experiment"]["id"] == contract["experiment_id"],
        "output_dir_matches_policy": resolve(config["paths"]["output_dir"]) == output_dir,
        "row_contract_matches": (
            int(config["data"]["expected_train_rows"]) == int(contract["expected_rows"])
            and base.get("train_rows") == int(contract["expected_rows"])
        ),
        "step_contract_matches": (
            int(training["total_optimizer_steps"]) == int(contract["expected_optimizer_steps"])
            == int(policy["checkpoint"]["expected_final_step"])
        ),
        "frozen_1024_token_two_gpu_profile": (
            int(config["data"]["max_response_length"]) == 1024
            and int(config["data"]["train_batch_size"]) == 8
            and int(config["rollout"]["n"]) == 1
            and int(config["resources"]["gpus_per_node"]) == 2
        ),
        "stage_prerequisite_pass": (
            prerequisite_path is None
            or (
                prerequisite is not None
                and prerequisite.get("stage") == "16"
                and prerequisite.get("stage_gate_pass") is True
            )
        ),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    return {
        "schema_version": 1, "generated_at_utc": utc_now(), "stage": stage,
        "experiment_id": contract["experiment_id"],
        "status": "PASS" if not failed else "FAIL", "gpu_used": False,
        "checks": checks, "failed_checks": failed,
        "config": str(config_path), "config_sha256": sha256_file(config_path),
        "policy": str(policy_path), "policy_sha256": sha256_file(policy_path),
        "effective_policy": policy, "output_dir": str(output_dir),
        "static_gate": str(static_gate_path), "static_gate_error": static_gate_error,
        "prerequisite_postflight": str(prerequisite_path) if prerequisite_path else None,
        "training_preflight": base,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("16", "64"), required=True)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--current-autodl-cost-cny", type=float)
    parser.add_argument("--billing-observed-at-utc")
    args = parser.parse_args()

    policy_path = resolve(args.policy)
    result = static_preflight(policy_path, args.stage)
    output_dir = Path(result["output_dir"])
    preflight_dir = output_dir / "preflight"
    write_json(preflight_dir / "pilot_guard_preflight.json", result)
    if not args.run:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print("No GPU training started.")
        return 0 if result["status"] == "PASS" else 1

    if args.current_autodl_cost_cny is None or not args.billing_observed_at_utc:
        parser.error("--run requires fresh billing amount and timestamp")
    policy, contract = load_pilot_policy(policy_path, args.stage)
    observed = parse_time(args.billing_observed_at_utc)
    age = (dt.datetime.now(dt.timezone.utc) - observed).total_seconds()
    gpus: list[dict[str, Any]] = []
    gpu_error = None
    try:
        gpus = query_gpus()
    except Exception as exc:
        gpu_error = repr(exc)
    disk = shutil.disk_usage(output_dir)
    collisions = output_collisions(output_dir)
    cgroup = read_cgroup(os.getpid())
    cgroup_capacity_pass = cgroup_has_minimum_capacity(
        cgroup, int(policy["memory"]["prelaunch_cgroup_minimum_bytes"])
    )
    max_initial_ratio = max(
        (row["memory_used_bytes"] / row["memory_total_bytes"] for row in gpus if row.get("memory_total_bytes")),
        default=1.0,
    )
    projected_total = args.current_autodl_cost_cny + float(contract["max_incremental_cost_cny"])
    live_checks = {
        "static_preflight_pass": result["status"] == "PASS",
        "billing_fresh": -300 <= age <= int(policy["budget"]["billing_observation_max_age_seconds"]),
        "projected_budget_below_hard_limit": projected_total <= float(policy["budget"]["project_hard_limit_cny"]),
        "storage_pass": disk.free >= int(policy["disk"]["prelaunch_required_bytes"]),
        "no_output_collision": not collisions,
        "expected_gpu_count": len(gpus) == 2,
        "gpus_initially_free": max_initial_ratio <= float(policy["memory"]["prelaunch_gpu_used_ratio_max"]),
        "cgroup_readable": bool(cgroup.get("supported")),
        "cgroup_capacity_pass": cgroup_capacity_pass,
        "standalone_vllm_port_free": not port_is_listening(8000)
        if policy["pilot"].get("forbid_listening_port_8000", True) else True,
    }
    result.update({
        "live_checks": live_checks, "status": "PASS" if all(live_checks.values()) else "FAIL",
        "gpu_used": bool(gpus), "gpus": gpus, "gpu_query_error": gpu_error,
        "max_initial_gpu_memory_ratio": max_initial_ratio, "disk_free_bytes": disk.free,
        "cgroup": cgroup, "collisions": collisions,
        "billing": {
            "observed_at_utc": observed.isoformat(), "age_seconds": age,
            "current_autodl_cost_cny": args.current_autodl_cost_cny,
            "stage_reservation_cny": contract["max_incremental_cost_cny"],
            "projected_total_cny": projected_total,
        },
        "git_status_porcelain": subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip(),
    })
    write_json(preflight_dir / "pilot_live_launch_gate.json", result)
    if result["status"] != "PASS":
        print(json.dumps(result, ensure_ascii=False, indent=2), file=sys.stderr)
        return 41

    env = os.environ.copy()
    env["VOPD_GUARD_ACTIVE"] = "1"
    command = ["bash", str(ROOT / "scripts/run_vopd_2gpu.sh"), "--config", result["config"], "--run"]
    process = subprocess.Popen(command, cwd=ROOT, env=env, start_new_session=True)
    try:
        exit_code, guard_summary = monitor_process(process, output_dir, policy, output_dir / "logs/train.log")
    except BaseException as exc:
        termination = terminate_process_group(process, float(policy["runtime"]["terminate_grace_seconds"])) if process.poll() is None else None
        exit_code = 40
        guard_summary = {
            "schema_version": 1, "status": "FAIL",
            "trigger": {"rule": "pilot_guard_exception", "detail": repr(exc)},
            "termination": termination,
        }
        write_json(output_dir / "evidence/guard_summary.json", guard_summary)

    postflight = None
    if exit_code == 0:
        audit_command = [
            sys.executable, str(resolve(policy["pilot"]["postflight_script"])),
            "--stage", args.stage, "--policy", str(policy_path),
        ]
        audit = subprocess.run(audit_command, cwd=ROOT, text=True)
        report_path = output_dir / "evidence/postflight.json"
        if report_path.is_file():
            postflight = json.loads(report_path.read_text(encoding="utf-8"))
        if audit.returncode != 0:
            exit_code = 43
    write_json(output_dir / "evidence/exit_receipt.json", {
        "schema_version": 1, "stage": args.stage, "guard_exit_code": exit_code,
        "guard_summary": guard_summary,
        "postflight_status": postflight.get("status") if postflight else None,
        "stage_gate_pass": postflight.get("stage_gate_pass") if postflight else False,
    })
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
