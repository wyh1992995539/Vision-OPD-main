#!/usr/bin/env python3
"""Cold-load the final Day 13 Student and run five deterministic smoke samples."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request

import pyarrow.parquet as pq
import yaml

from scripts.vopd_6241_pilot_reload import resolve, runtime_env, select_samples, student_messages
from scripts.vopd_day8_reload import (
    port_is_open,
    sha256_file,
    stop_process_group,
    utc_now,
    verify_predictions,
    wait_for_server,
    write_json_atomic,
)


ROOT = Path(__file__).resolve().parents[1]


def file_stats(paths: list[Path]) -> list[dict]:
    return [
        {"path": str(path), "size_bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
        for path in paths
    ]


def verify_manifest(manifest_path: Path, base_dir: Path) -> tuple[list[Path], list[str]]:
    manifest = json.loads(manifest_path.read_text())
    errors: list[str] = []
    paths: list[Path] = []
    for record in manifest.get("files", []):
        relative = record.get("relative_path")
        if not relative:
            errors.append("manifest entry missing relative_path")
            continue
        path = base_dir / relative
        paths.append(path)
        if not path.is_file():
            errors.append(f"missing merged file: {path}")
            continue
        if path.stat().st_size != int(record["size_bytes"]):
            errors.append(f"size mismatch: {path}")
            continue
        if sha256_file(path) != record["sha256"]:
            errors.append(f"SHA256 mismatch: {path}")
    if len(paths) != int(manifest.get("file_count", -1)):
        errors.append("merged manifest file count mismatch")
    return paths, errors


def preflight(config_path: Path) -> tuple[dict, list[tuple[str, dict]], list[Path], list[Path]]:
    config = yaml.safe_load(config_path.read_text())
    source = resolve(config["source_checkpoint"])
    merged = resolve(config["merged_model_dir"])
    output = resolve(config["output_dir"])
    checkpoint_manifest_path = resolve(config["checkpoint_manifest"])
    checkpoint_receipt_path = resolve(config["checkpoint_hash_receipt"])
    merged_manifest_path = resolve(config["merged_manifest"])
    merge_receipt_path = resolve(config["merge_receipt"])
    errors: list[str] = []

    for path in (checkpoint_manifest_path, checkpoint_receipt_path, merged_manifest_path, merge_receipt_path):
        if not path.is_file():
            errors.append(f"missing evidence: {path}")
    if errors:
        raise ValueError("; ".join(errors))

    checkpoint_manifest = json.loads(checkpoint_manifest_path.read_text())
    checkpoint_receipt = json.loads(checkpoint_receipt_path.read_text())
    merged_manifest = json.loads(merged_manifest_path.read_text())
    merge_receipt = json.loads(merge_receipt_path.read_text())
    if checkpoint_manifest.get("status") not in (None, "PASS"):
        errors.append("checkpoint manifest status is not PASS")
    if checkpoint_receipt.get("status") != "PASS" or checkpoint_receipt.get("independent_verification", {}).get("status") != "PASS":
        errors.append("checkpoint SHA256 receipt is not independently verified PASS")
    if merge_receipt.get("status") != "PASS" or merge_receipt.get("independent_verification", {}).get("status") != "PASS":
        errors.append("merge receipt is not independently verified PASS")
    if merge_receipt.get("merged_manifest_sha256") != sha256_file(merged_manifest_path):
        errors.append("merge receipt does not bind the current merged manifest")
    recorded = (
        checkpoint_manifest.get("checkpoint_directory")
        or checkpoint_manifest.get("checkpoint_dir")
        or checkpoint_manifest.get("source_checkpoint")
        or checkpoint_manifest.get("source_checkpoint_dir")
    )
    if not recorded or Path(recorded).resolve() != source:
        errors.append("checkpoint manifest does not bind the configured source checkpoint")

    source_paths: list[Path] = []
    for record in checkpoint_manifest.get("files", []):
        raw = record.get("absolute_path") or record.get("path")
        path = Path(raw) if raw else source / record["relative_path"]
        source_paths.append(path)
        if not path.is_file() or path.stat().st_size != int(record["size_bytes"]):
            errors.append(f"source checkpoint stat mismatch: {path}")

    merged_paths, merged_errors = verify_manifest(merged_manifest_path, merged)
    errors.extend(merged_errors)
    data = resolve(config["input_parquet"])
    if sha256_file(data) != config["input_parquet_sha256"]:
        errors.append("training parquet SHA256 mismatch")
    rows = pq.read_table(data, columns=["extra_info", "prompt", "images"]).to_pylist()
    if len(rows) != int(config["expected_samples"]):
        errors.append(f"expected {config['expected_samples']} rows, found {len(rows)}")
    samples = select_samples(rows, int(config["reload_samples"]), int(config["seed"]))
    for _, row in samples:
        student_messages(row)

    serving = config["serving"]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(serving["cuda_visible_devices"])
    import torch

    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if gpu_count != int(serving["required_gpu_count"]):
        errors.append(f"expected {serving['required_gpu_count']} visible GPU, found {gpu_count}")
    if int(serving["tensor_parallel_size"]) != int(serving["required_gpu_count"]):
        errors.append("tensor_parallel_size must equal required_gpu_count")
    if not serving.get("enforce_eager"):
        errors.append("cold reload must use eager mode")
    if port_is_open(str(serving["host"]), int(serving["port"])):
        errors.append("serving port is occupied")
    if output.exists() and any(output.iterdir()):
        errors.append("cold reload output directory is already non-empty")
    if errors:
        raise ValueError("; ".join(errors))
    return config, samples, source_paths, merged_paths


def execute(config_path: Path) -> dict:
    config, samples, source_paths, merged_paths = preflight(config_path)
    output = resolve(config["output_dir"])
    merged = resolve(config["merged_model_dir"])
    output.mkdir(parents=True, exist_ok=False)
    source_before = file_stats(source_paths)
    merged_before = file_stats(merged_paths)
    serving = config["serving"]
    summary = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "phase": config["phase"],
        "scope": "final_student_training_samples_functional_inference_only",
        "training_resume_validated": False,
        "model_quality_evaluated": False,
        "source_checkpoint": str(resolve(config["source_checkpoint"])),
        "merged_model": str(merged),
        "started_at_utc": utc_now(),
        "config_sha256": sha256_file(config_path),
        "script_sha256": sha256_file(Path(__file__)),
        "input_parquet_sha256": config["input_parquet_sha256"],
        "checkpoint_manifest_sha256": sha256_file(resolve(config["checkpoint_manifest"])),
        "merged_manifest_sha256": sha256_file(resolve(config["merged_manifest"])),
        "merge_receipt_sha256": sha256_file(resolve(config["merge_receipt"])),
        "sample_ids": [sample_id for sample_id, _ in samples],
        "status": "RUNNING",
    }
    write_json_atomic(output / "config_snapshot.json", config)
    write_json_atomic(output / "reload_validation_summary.json", summary)

    vllm = shutil.which("vllm") or "/root/miniconda3/envs/vision-opd/bin/vllm"
    command = [
        vllm, "serve", str(merged),
        "--served-model-name", str(serving["model_name"]),
        "--host", str(serving["host"]), "--port", str(serving["port"]),
        "--trust-remote-code", "--dtype", str(serving["dtype"]),
        "--max-model-len", str(serving["max_model_len"]),
        "--max-num-seqs", str(serving["max_num_seqs"]),
        "--tensor-parallel-size", str(serving["tensor_parallel_size"]),
        "--gpu-memory-utilization", str(serving["gpu_memory_utilization"]),
        "--enforce-eager", "--limit-mm-per-prompt", '{"image":1,"video":0}',
        "--chat-template", str(resolve(config["chat_template"])),
        "--default-chat-template-kwargs", '{"enable_thinking":false}',
        "--kernel-config", '{"enable_flashinfer_autotune":false}',
        "--compilation-config", '{"pass_config":{"fuse_allreduce_rms":false}}',
    ]
    run_env = runtime_env(CUDA_VISIBLE_DEVICES=str(serving["cuda_visible_devices"]))
    write_json_atomic(output / "server_command.json", {"command": command, "cuda_visible_devices": serving["cuda_visible_devices"]})
    server = None
    stream = None
    try:
        stream = (output / "vllm_server.log").open("w")
        server = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=run_env, start_new_session=True)
        base = f"http://{serving['host']}:{serving['port']}/v1"
        models = wait_for_server(server, base + "/models", int(serving["startup_timeout_seconds"]), output / "vllm_server.log")
        if serving["model_name"] not in [item["id"] for item in models.get("data", [])]:
            raise ValueError("unexpected served model ID")
        records = []
        for index, (sample_id, row) in enumerate(samples, 1):
            record = {"sample_id": sample_id, "raw_prediction": "", "response_token_count": 0, "finish_reason": "error", "inference_error": None, "response_token_ids": None}
            try:
                payload = {
                    "model": serving["model_name"], "messages": student_messages(row),
                    "temperature": 0.0, "top_p": 1.0, "max_tokens": int(serving["max_new_tokens"]),
                    "seed": int(config["seed"]), "return_token_ids": True,
                    "chat_template_kwargs": {"enable_thinking": False},
                }
                request = urllib.request.Request(base + "/chat/completions", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(request, timeout=300) as response:
                    value = json.load(response)
                choice = value["choices"][0]
                record.update(raw_prediction=choice["message"]["content"], response_token_count=value["usage"]["completion_tokens"], finish_reason=choice["finish_reason"], response_token_ids=choice.get("token_ids"))
            except Exception as exc:
                record["inference_error"] = f"{type(exc).__name__}: {exc}"
            records.append(record)
            with (output / "predictions.jsonl").open("a") as predictions:
                predictions.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"INFERENCE {index}/{len(samples)} {sample_id} tokens={record['response_token_count']} finish={record['finish_reason']}", flush=True)
        write_json_atomic(output / "summary.json", {"total": len(records), "unique_sample_ids": len({r['sample_id'] for r in records}), "not_accuracy_evaluation": True})
        summary["verification"] = verify_predictions(output, summary["sample_ids"])
        if summary["verification"]["status"] != "PASS":
            raise ValueError(str(summary["verification"]))
        summary["status"] = "PASS"
    except BaseException as exc:
        summary["status"] = "FAIL"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if server is not None:
            summary["server_exit_code_after_controlled_shutdown"] = stop_process_group(server)
        if stream is not None:
            stream.close()
        summary["source_checkpoint_unchanged"] = source_before == file_stats(source_paths)
        summary["merged_model_unchanged"] = merged_before == file_stats(merged_paths)
        if not summary["source_checkpoint_unchanged"] or not summary["merged_model_unchanged"]:
            summary["status"] = "FAIL"
            summary["error"] = "source checkpoint or merged model changed during cold reload"
        summary["completed_at_utc"] = utc_now()
        write_json_atomic(output / "reload_validation_summary.json", summary)
        print("RELOAD_STATUS=" + summary["status"], flush=True)
    if summary["status"] != "PASS":
        raise RuntimeError(summary.get("error", "cold reload failed"))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/vopd_day13_final_reload.yaml")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.preflight_only == args.run:
        parser.error("choose exactly one of --preflight-only or --run")
    config, samples, _, _ = preflight(args.config.resolve())
    result = {
        "schema_version": 1, "created_at_utc": utc_now(), "status": "PASS",
        "config": str(args.config.resolve()), "config_sha256": sha256_file(args.config.resolve()),
        "sample_ids": [sample_id for sample_id, _ in samples],
        "gpu_count": config["serving"]["required_gpu_count"],
    }
    preflight_path = resolve(config["output_dir"]).parent / "evidence" / "cold_reload_preflight.json"
    write_json_atomic(preflight_path, result)
    print("DAY13_COLD_RELOAD_PREFLIGHT=PASS", flush=True)
    print("SAMPLE_IDS=" + json.dumps(result["sample_ids"]), flush=True)
    if args.run:
        execute(args.config.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
