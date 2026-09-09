#!/usr/bin/env python3
"""Run one frozen Day15 R3 inference or Base-Judge service stage."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import urllib.error
import urllib.request
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
R3 = ROOT / "configs/benchmark_eval_paper_basejudge_r3_single_gpu.yaml"
FREEZE = ROOT / "configs/day15_final_eval_freeze.yaml"


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def wait_ready(process: subprocess.Popen, url: str, expected_id: str, log: Path) -> None:
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        if process.poll() is not None:
            tail = log.read_text(errors="replace")[-5000:] if log.exists() else ""
            raise RuntimeError(f"vLLM exited with {process.returncode}: {tail}")
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                data = json.load(response)
            if expected_id in [item["id"] for item in data.get("data", [])]:
                print(f"SERVICE_READY model={expected_id}", flush=True)
                return
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError(f"vLLM startup timed out for {expected_id}")


def stop(process: subprocess.Popen) -> int:
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            return process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
    return process.wait()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=["inference", "judge"])
    parser.add_argument("--scope", required=True, choices=["smoke", "formal"])
    parser.add_argument("--role", required=True, choices=["vision_opd", "cached_prefix", "both"])
    args = parser.parse_args()
    if args.stage == "inference" and args.role == "both":
        parser.error("inference service can evaluate only one model role at a time")

    freeze = yaml.safe_load(FREEZE.read_text())
    r3 = yaml.safe_load(R3.read_text())
    roles = ["vision_opd", "cached_prefix"] if args.role == "both" else [args.role]
    if args.stage == "inference":
        spec = freeze["models"][roles[0]]
        model_path = resolve(spec["merged_model"])
        model_id = spec["model_id"]
    else:
        model_path = resolve(r3["judge"]["model"]["path"])
        model_id = r3["judge"]["model"]["served_model_name"]

    outputs = [
        resolve(freeze["models"][role][f"{args.scope}_output"])
        for role in roles
    ]
    for output in outputs:
        if not (output / "run_manifest.json").is_file():
            raise FileNotFoundError(f"prepared run manifest missing: {output}")
    host, port = "127.0.0.1", 8000
    try:
        urllib.request.urlopen(f"http://{host}:{port}/v1/models", timeout=2)
        raise RuntimeError(f"port {port} is already occupied")
    except urllib.error.URLError:
        pass

    vllm = shutil.which("vllm") or "/root/miniconda3/envs/vision-opd/bin/vllm"
    command = [
        vllm, "serve", str(model_path),
        "--served-model-name", model_id,
        "--tensor-parallel-size", "1",
        "--gpu-memory-utilization", "0.75",
        "--trust-remote-code",
        "--additional-config", '{"gdn_prefill_backend":"triton"}',
        "--host", host, "--port", str(port),
    ]
    session_root = ROOT / "artifacts/runs/E-D15-6K-FINAL-EVAL-001/evidence"
    session_root.mkdir(parents=True, exist_ok=True)
    role_label = args.role
    log = session_root / f"{args.scope}_{role_label}_{args.stage}_server.log"
    receipt_path = session_root / f"{args.scope}_{role_label}_{args.stage}_session.json"
    if receipt_path.exists():
        raise RuntimeError(f"refusing to overwrite stage receipt: {receipt_path}")
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="8")
    started = now()
    service = None
    stream = None
    stage_results = []
    error = None
    service_exit = None
    try:
        stream = log.open("w")
        service = subprocess.Popen(
            command, cwd=ROOT, env=env, stdout=stream,
            stderr=subprocess.STDOUT, start_new_session=True,
        )
        print(f"SERVICE_START stage={args.stage} scope={args.scope} model={model_id}", flush=True)
        wait_ready(service, f"http://{host}:{port}/v1/models", model_id, log)
        for role, output in zip(roles, outputs):
            if args.stage == "inference":
                spec = freeze["models"][role]
                stage_command = [
                    "/root/miniconda3/envs/vision-opd/bin/python",
                    "eval/run_paper_aligned_eval.py",
                    "--config", str(R3),
                    "--model-role", role,
                    "--model-path", str(resolve(spec["merged_model"])),
                    "--model-id", spec["model_id"],
                    "--output-dir", str(output),
                    "--api-base", f"http://{host}:{port}/v1",
                ]
                if args.scope == "smoke":
                    stage_command.extend(["--limit-per-benchmark", "4"])
            else:
                stage_command = [
                    "/root/miniconda3/envs/vision-opd/bin/python",
                    "eval/run_paper_aligned_judge.py",
                    "--config", str(R3),
                    "--input-dir", str(output),
                    "--judge-model-id", model_id,
                    "--judge-model-path", str(model_path),
                    "--api-base", f"http://{host}:{port}/v1",
                ]
            print(f"STAGE_RUN role={role} output={output}", flush=True)
            result = subprocess.run(stage_command, cwd=ROOT, env=env)
            stage_results.append({
                "role": role, "output": str(output.resolve()),
                "command": stage_command, "exit_code": result.returncode,
            })
            if result.returncode != 0:
                raise RuntimeError(f"{args.stage} failed for {role}: exit {result.returncode}")
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if service is not None:
            service_exit = stop(service)
        if stream is not None:
            stream.close()
        receipt = {
            "schema_version": 1,
            "created_at_utc": now(),
            "status": "PASS" if error is None and all(x["exit_code"] == 0 for x in stage_results) else "FAIL",
            "stage": args.stage,
            "scope": args.scope,
            "roles": roles,
            "model_path": str(model_path.resolve()),
            "model_id": model_id,
            "service_command": command,
            "service_started_at_utc": started,
            "service_exit_code_after_controlled_shutdown": service_exit,
            "stage_results": stage_results,
            "error": error,
            "service_log": str(log.resolve()),
        }
        receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
        print(f"DAY15_R3_STAGE={receipt['status']} stage={args.stage} scope={args.scope} roles={','.join(roles)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
