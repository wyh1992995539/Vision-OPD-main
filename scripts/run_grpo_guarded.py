#!/usr/bin/env python3
"""Guarded entry point for the Vision-OPD MCQ GRPO branch."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.grpo_training_preflight import atomic_write, resolve, validate_config


def training_command(report: dict, extra: list[str]) -> list[str]:
    return [
        sys.executable, "-m", "verl.trainer.main_ppo", "--config-name", "baseline_grpo",
        *report["hydra_overrides"], *extra,
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail-closed GRPO launcher")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/grpo_6241.yaml")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    args, extra = parser.parse_known_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output_dir = resolve(config["paths"]["output_dir"])
    report_path = output_dir / "preflight/preflight.json"
    report = validate_config(args.config, report_path)
    command = training_command(report, extra)
    if not args.run:
        print("PASS: GRPO configuration and static preflight")
        print(f"report: {report_path}")
        print("No GPU training started. Day18 32-prompt Pilot remains the next gate.")
        return

    authorization = config["validation"]
    if not authorization.get("pilot_training_authorized", False) and not authorization.get(
        "formal_training_authorized", False
    ):
        raise SystemExit("RUN BLOCKED: config authorizes neither Pilot nor formal GRPO training")
    if not report["runtime_resources_ready"]:
        raise SystemExit(f"RUN BLOCKED: runtime resource gates: {report['blocking_gates']}")
    collisions = []
    for name in ("checkpoints", "rollouts", "logs"):
        target = output_dir / name
        if target.exists() and any(target.iterdir()):
            collisions.append(str(target))
    if collisions:
        raise SystemExit(f"RUN BLOCKED: output collision: {collisions}")
    for name in ("checkpoints", "rollouts", "logs", "evidence"):
        (output_dir / name).mkdir(parents=True, exist_ok=True)
    invocation = {
        "schema_version": 1,
        "experiment_id": report["experiment_id"],
        "config_sha256": report["config_sha256"],
        "train_file_sha256": report["train_file_sha256"],
        "reward_function_sha256": report["reward_function_sha256"],
        "command": command,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "0,1"),
    }
    atomic_write(output_dir / "preflight/run_invocation.json", json.dumps(invocation, indent=2) + "\n")
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
    env.setdefault("OMP_NUM_THREADS", "1")
    env["PYTHONPATH"] = f"{ROOT}:{env.get('PYTHONPATH', '')}"
    env["PYTHONUNBUFFERED"] = "1"
    env["VLLM_USE_V1"] = "1"
    raise SystemExit(subprocess.run(command, cwd=ROOT, env=env, check=False).returncode)


if __name__ == "__main__":
    main()
