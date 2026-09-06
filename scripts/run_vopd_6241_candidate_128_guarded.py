#!/usr/bin/env python3
"""Fail-closed launcher for the 128-row, 16-step formal-candidate validation."""

from __future__ import annotations

import argparse
import datetime as dt
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
    cgroup_has_minimum_capacity,
    load_policy,
    monitor_process,
    query_gpus,
    read_cgroup,
    terminate_process_group,
    utc_now,
    write_json,
)
from scripts.vopd_training_preflight import validate_config

DEFAULT_POLICY = ROOT / "configs/vopd_6241_candidate_128_abort_policy.yaml"


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def parse_time(value: str) -> dt.datetime:
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("billing timestamp must include timezone")
    return parsed.astimezone(dt.timezone.utc)


def port_is_listening(port: int, host: str = "127.0.0.1") -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.25):
            return True
    except OSError:
        return False


def output_collisions(output_dir: Path) -> list[str]:
    collisions: list[str] = []
    for relative in ("logs", "rollouts", "checkpoints", "evidence/telemetry", "evidence/log_probs"):
        root = output_dir / relative
        if root.exists():
            collisions.extend(str(path) for path in root.rglob("*") if path.is_file())
    for relative in (
        "evidence/guard_summary.json",
        "evidence/exit_receipt.json",
        "evidence/postflight.json",
        "evidence/postflight.md",
        "evidence/postflight_sha256.txt",
        "preflight/run_invocation.json",
    ):
        path = output_dir / relative
        if path.is_file():
            collisions.append(str(path))
    return sorted(set(collisions))


def load_candidate_policy(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    policy = load_policy(path)
    contract = policy.get("candidate_validation")
    if not isinstance(contract, dict):
        raise ValueError("policy must contain candidate_validation mapping")
    required = {
        "config",
        "expected_config_sha256",
        "source_candidate",
        "expected_source_candidate_sha256",
        "selection_manifest",
        "expected_selection_sha256",
        "preparation_preflight",
        "output_dir",
        "guarded_launcher",
        "postflight_script",
        "expected_rows",
        "expected_optimizer_steps",
        "required_post_warmup_steps",
        "normal_eos_required",
        "formal_training_authorized",
    }
    missing = sorted(required - set(contract))
    if missing:
        raise ValueError(f"candidate_validation policy missing: {missing}")
    if contract["formal_training_authorized"] is not False:
        raise ValueError("candidate validation policy must not authorize formal training")
    if int(policy["checkpoint"]["expected_final_step"]) != int(contract["expected_optimizer_steps"]):
        raise ValueError("checkpoint final step does not match candidate-validation contract")
    return policy, contract


def source_bindings(policy_path: Path, contract: dict[str, Any]) -> dict[str, dict[str, str]]:
    paths = {
        "policy": policy_path,
        "config": resolve(contract["config"]),
        "source_candidate": resolve(contract["source_candidate"]),
        "selection_manifest": resolve(contract["selection_manifest"]),
        "preparation_preflight": resolve(contract["preparation_preflight"]),
        "guarded_launcher": resolve(contract["guarded_launcher"]),
        "postflight_script": resolve(contract["postflight_script"]),
    }
    return {
        name: {"path": str(path), "sha256": sha256_file(path)}
        for name, path in paths.items()
        if path.is_file()
    }


def static_preflight(policy_path: Path = DEFAULT_POLICY) -> dict[str, Any]:
    policy_path = resolve(policy_path)
    policy, contract = load_candidate_policy(policy_path)
    config_path = resolve(contract["config"])
    source_candidate_path = resolve(contract["source_candidate"])
    selection_path = resolve(contract["selection_manifest"])
    preparation_path = resolve(contract["preparation_preflight"])
    output_dir = resolve(contract["output_dir"])
    launcher_path = resolve(contract["guarded_launcher"])
    postflight_path = resolve(contract["postflight_script"])

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source_candidate = yaml.safe_load(source_candidate_path.read_text(encoding="utf-8"))
    selection = read_json(selection_path)
    preparation = read_json(preparation_path)
    base = validate_config(config_path, ROOT)
    config_hash = sha256_file(config_path)
    source_candidate_hash = sha256_file(source_candidate_path)
    selection_hash = sha256_file(selection_path)

    training = config["training"]
    validation = config["validation"]
    actor_without_path = {**config["actor"], "memory_profile_dir": None}
    source_actor_without_path = {**source_candidate["actor"], "memory_profile_dir": None}
    common_data_keys = (
        "train_batch_size",
        "image_key",
        "teacher_image_key",
        "max_prompt_length",
        "max_response_length",
        "truncation",
        "dataloader_num_workers",
        "tail_policy",
    )
    control_paths = {
        "guarded_launcher": launcher_path,
        "postflight_script": postflight_path,
        "abort_policy": policy_path,
    }
    checks = {
        "policy_path_bound": policy_path == DEFAULT_POLICY.resolve(),
        "policy_identity": (
            policy.get("experiment_id") == config["experiment"]["id"]
            == "E-D11-6K-VOPD-CANDIDATE-128"
        ),
        "config_path_bound": config_path == resolve("configs/vopd_6241_candidate_128.yaml"),
        "config_hash_bound": config_hash == contract["expected_config_sha256"],
        "source_candidate_hash_bound": (
            source_candidate_hash == contract["expected_source_candidate_sha256"]
        ),
        "selection_hash_bound": selection_hash == contract["expected_selection_sha256"],
        "config_ready_for_guarded_validation": (
            config.get("status") == "ready_for_guarded_gpu_validation"
        ),
        "training_preflight_pass": base.get("status") == "PASS",
        "preparation_preflight_current": (
            preparation.get("status") == "PASS"
            and preparation.get("experiment_id") == config["experiment"]["id"]
            and preparation.get("config_sha256") == config_hash
            and preparation.get("train_file_sha256")
            == selection.get("output", {}).get("sha256")
        ),
        "selection_identity_and_rows": (
            selection.get("experiment_id") == config["experiment"]["id"]
            and selection.get("selection", {}).get("selected_rows")
            == int(contract["expected_rows"])
            and base.get("train_rows") == int(contract["expected_rows"])
        ),
        "exact_128x16_contract": (
            int(config["data"]["expected_train_rows"]) == int(contract["expected_rows"]) == 128
            and int(training["source_samples"]) == 128
            and int(training["expected_samples"]) == 128
            and int(training["padded_samples"]) == 128
            and int(training["padding_rows"]) == 0
            and int(training["dropped_rows"]) == 0
            and int(training["total_optimizer_steps"])
            == int(contract["expected_optimizer_steps"]) == 16
            and int(config["data"]["train_batch_size"]) == 8
            and training["require_full_epoch"] is True
            and config["data"]["shuffle"] is False
        ),
        "algorithm_matches_formal_candidate": (
            actor_without_path == source_actor_without_path
            and config["rollout"] == source_candidate["rollout"]
            and config["self_distillation"] == source_candidate["self_distillation"]
            and all(config["data"][key] == source_candidate["data"][key] for key in common_data_keys)
        ),
        "deferred_and_normal_eos": (
            config["actor"]["defer_optimizer_state_load"] is True
            and config["actor"]["optimizer_offload"] is True
            and config["rollout"]["ignore_eos"] is False
            and validation["normal_eos_required"] is True
            and validation["forced_min_response_tokens"] is None
            and "diagnostic_generation" not in config
        ),
        "validation_never_authorizes_formal_training": (
            validation["formal_training_authorized"] is False
            and contract["formal_training_authorized"] is False
        ),
        "resource_policy_covers_config": (
            int(config["resources"]["prelaunch_cgroup_minimum_bytes"])
            <= int(policy["memory"]["prelaunch_cgroup_minimum_bytes"])
        ),
        "checkpoint_contract": (
            int(training["save_frequency"]) == -1
            and policy["checkpoint"]["allowed_save_steps"] == [16]
            and int(policy["checkpoint"]["expected_final_step"]) == 16
            and validation["final_checkpoint_required"] is True
        ),
        "post_warmup_contract": (
            int(config["actor"]["lr_warmup_steps"]) == 10
            and int(validation["required_post_warmup_steps"])
            == int(contract["required_post_warmup_steps"]) == 6
        ),
        "control_paths_match": (
            validation["guarded_launcher"] == contract["guarded_launcher"]
            and resolve(validation["abort_policy"]) == DEFAULT_POLICY.resolve()
            and validation["postflight_script"] == contract["postflight_script"]
        ),
        "control_files_present": all(path.is_file() for path in control_paths.values()),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    return {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "status": "PASS" if not failed else "FAIL",
        "gpu_used": False,
        "experiment_id": config["experiment"]["id"],
        "checks": checks,
        "failed_checks": failed,
        "config": str(config_path),
        "config_sha256": config_hash,
        "policy": str(policy_path),
        "policy_sha256": sha256_file(policy_path),
        "output_dir": str(output_dir),
        "source_bindings": source_bindings(policy_path, contract),
        "training_preflight": base,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--current-autodl-cost-cny", type=float)
    parser.add_argument("--billing-observed-at-utc")
    args = parser.parse_args()

    policy_path = resolve(args.policy)
    policy, contract = load_candidate_policy(policy_path)
    result = static_preflight(policy_path)
    output_dir = Path(result["output_dir"])
    preflight_dir = output_dir / "preflight"
    write_json(preflight_dir / "candidate_guard_preflight.json", result)

    if not args.run:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print("No GPU validation started.")
        return 0 if result["status"] == "PASS" else 1

    if args.current_autodl_cost_cny is None or not args.billing_observed_at_utc:
        parser.error("--run requires fresh billing amount and timestamp")

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
    max_initial_ratio = max(
        (
            row["memory_used_bytes"] / row["memory_total_bytes"]
            for row in gpus
            if row.get("memory_total_bytes")
        ),
        default=1.0,
    )
    projected_total = (
        args.current_autodl_cost_cny
        + float(policy["budget"]["conservative_reservation_cny"])
    )
    live_checks = {
        "static_preflight_pass": result["status"] == "PASS",
        "billing_fresh": (
            -300 <= age <= int(policy["budget"]["billing_observation_max_age_seconds"])
        ),
        "projected_budget_below_hard_limit": (
            projected_total <= float(policy["budget"]["project_hard_limit_cny"])
        ),
        "storage_pass": disk.free >= int(policy["disk"]["prelaunch_required_bytes"]),
        "no_output_collision": not collisions,
        "expected_gpu_count": len(gpus) == 2,
        "gpus_initially_free": (
            max_initial_ratio <= float(policy["memory"]["prelaunch_gpu_used_ratio_max"])
        ),
        "cgroup_readable": bool(cgroup.get("supported")),
        "cgroup_capacity_pass": cgroup_has_minimum_capacity(
            cgroup, int(policy["memory"]["prelaunch_cgroup_minimum_bytes"])
        ),
        "standalone_vllm_port_free": (
            not port_is_listening(8000)
            if contract.get("forbid_listening_port_8000", True)
            else True
        ),
        "source_bindings_still_current": all(
            Path(entry["path"]).is_file()
            and sha256_file(Path(entry["path"])) == entry["sha256"]
            for entry in result["source_bindings"].values()
        ),
    }
    result.update(
        {
            "generated_at_utc": utc_now(),
            "status": "PASS" if all(live_checks.values()) else "FAIL",
            "gpu_used": bool(gpus),
            "live_checks": live_checks,
            "gpus": gpus,
            "gpu_query_error": gpu_error,
            "max_initial_gpu_memory_ratio": max_initial_ratio,
            "disk_free_bytes": disk.free,
            "cgroup": cgroup,
            "collisions": collisions,
            "billing": {
                "observed_at_utc": observed.isoformat(),
                "age_seconds": age,
                "current_autodl_cost_cny": args.current_autodl_cost_cny,
                "reservation_cny": policy["budget"]["conservative_reservation_cny"],
                "projected_total_cny": projected_total,
            },
            "git_status_porcelain": subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
        }
    )
    write_json(preflight_dir / "live_launch_gate.json", result)
    if result["status"] != "PASS":
        print(json.dumps(result, ensure_ascii=False, indent=2), file=sys.stderr)
        return 41

    env = os.environ.copy()
    env["VOPD_GUARD_ACTIVE"] = "1"
    command = [
        "bash",
        str(ROOT / "scripts/run_vopd_2gpu.sh"),
        "--config",
        result["config"],
        "--run",
    ]
    process = subprocess.Popen(command, cwd=ROOT, env=env, start_new_session=True)
    try:
        exit_code, guard_summary = monitor_process(
            process, output_dir, policy, output_dir / "logs/train.log"
        )
    except BaseException as exc:
        termination = (
            terminate_process_group(
                process, float(policy["runtime"]["terminate_grace_seconds"])
            )
            if process.poll() is None
            else None
        )
        exit_code = 40
        guard_summary = {
            "schema_version": 1,
            "status": "FAIL",
            "trigger": {
                "rule": "candidate_validation_guard_exception",
                "detail": repr(exc),
            },
            "termination": termination,
        }
        write_json(output_dir / "evidence/guard_summary.json", guard_summary)

    postflight = None
    if exit_code == 0:
        audit = subprocess.run(
            [
                sys.executable,
                str(resolve(contract["postflight_script"])),
                "--policy",
                str(policy_path),
            ],
            cwd=ROOT,
            text=True,
        )
        postflight_path = output_dir / "evidence/postflight.json"
        if postflight_path.is_file():
            postflight = read_json(postflight_path)
        if audit.returncode != 0:
            exit_code = 43

    write_json(
        output_dir / "evidence/exit_receipt.json",
        {
            "schema_version": 1,
            "experiment_id": policy["experiment_id"],
            "guard_exit_code": exit_code,
            "guard_summary": guard_summary,
            "postflight_status": postflight.get("status") if postflight else None,
            "validation_gate_pass": (
                postflight.get("validation_gate_pass") if postflight else False
            ),
            "formal_training_authorized": False,
        },
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
