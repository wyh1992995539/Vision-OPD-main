#!/usr/bin/env python3
"""Cold-reload and validate the E-D8-001 checkpoint on five frozen eval samples."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import yaml


GIB = 1024**3
SELECTION_ALGORITHM = "sha256(seed|sample_id),ascending"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_key(seed: int, sample_id: str) -> str:
    return hashlib.sha256(f"{seed}|{sample_id}".encode("utf-8")).hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    return value


def resolve(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else project_root / path).resolve()


def frozen_eval_ids(eval_path: Path, *, expected_rows: int, seed: int, count: int) -> tuple[list[str], list[str]]:
    table = pq.read_table(eval_path, columns=["images", "extra_info"])
    if table.num_rows != expected_rows:
        raise ValueError(f"expected {expected_rows} eval rows, found {table.num_rows}")
    sample_ids: list[str] = []
    missing_images: list[str] = []
    for row_index, row in enumerate(table.to_pylist()):
        try:
            sample_id = str(row["extra_info"]["provenance"]["sample_id"]).strip()
            images = row["images"]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"eval row {row_index}: invalid provenance or images") from exc
        if not sample_id:
            raise ValueError(f"eval row {row_index}: empty sample_id")
        sample_ids.append(sample_id)
        if not isinstance(images, list) or len(images) != 1 or not Path(images[0]["path"]).is_file():
            missing_images.append(sample_id)
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("eval Parquet contains duplicate sample IDs")
    selected = sorted(sample_ids, key=lambda sample_id: (stable_key(seed, sample_id), sample_id))[:count]
    return selected, missing_images


def required_checkpoint_files(config: dict[str, Any], project_root: Path) -> list[Path]:
    checkpoint = config["checkpoint"]
    actor_dir = resolve(project_root, checkpoint["actor_dir"])
    paths = [actor_dir / name for name in checkpoint["required_rank_files"]]
    paths.extend(actor_dir / "huggingface" / name for name in checkpoint["required_huggingface_files"])
    paths.append(actor_dir / "fsdp_config.json")
    paths.append(resolve(project_root, checkpoint["source_dir"]) / "data.pt")
    return paths


def file_snapshot(paths: list[Path], *, include_sha256: bool) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        stat = path.stat()
        record: dict[str, Any] = {
            "path": str(path),
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
        if include_sha256:
            record["sha256"] = sha256_file(path)
        records.append(record)
    return records


def directory_manifest(path: Path) -> list[dict[str, Any]]:
    files = sorted(item for item in path.rglob("*") if item.is_file())
    return [
        {
            "path": str(item.relative_to(path)),
            "size_bytes": item.stat().st_size,
            "sha256": sha256_file(item),
        }
        for item in files
    ]


def validate_merged_model(path: Path) -> list[str]:
    errors: list[str] = []
    for name in ("config.json", "tokenizer_config.json"):
        if not (path / name).is_file():
            errors.append(f"merged model missing {name}")
    weight_files = list(path.glob("*.safetensors"))
    if not weight_files:
        errors.append("merged model has no safetensors weights")
    if any(item.stat().st_size == 0 for item in weight_files):
        errors.append("merged model contains an empty safetensors file")
    return errors


def memory_limit_bytes() -> int:
    candidates: list[int] = []
    memory_max = Path("/sys/fs/cgroup/memory.max")
    if memory_max.is_file():
        raw = memory_max.read_text(encoding="utf-8").strip()
        if raw != "max":
            candidates.append(int(raw))
    try:
        import psutil

        candidates.append(int(psutil.virtual_memory().total))
    except ImportError:
        pass
    return min(candidates) if candidates else 0


def validate_static(config_path: Path, project_root: Path) -> dict[str, Any]:
    config = load_config(config_path)
    checkpoint = config["checkpoint"]
    evaluation = config["evaluation"]
    serving = config["serving"]
    gate = config["runtime_gate"]
    errors: list[str] = []
    warnings: list[str] = []

    source_dir = resolve(project_root, checkpoint["source_dir"])
    actor_dir = resolve(project_root, checkpoint["actor_dir"])
    merged_dir = resolve(project_root, checkpoint["merged_model_dir"])
    eval_path = resolve(project_root, evaluation["input_parquet"])
    chat_template = resolve(project_root, config["shared"]["chat_template"]["file"])
    latest_iteration = resolve(project_root, checkpoint["latest_iteration_file"])
    output_dir = resolve(project_root, evaluation["output_dir"])
    evidence_dir = resolve(project_root, config["evidence"]["directory"])

    if config["experiment"]["id"] != "E-D8-001":
        errors.append("experiment.id must be E-D8-001")
    if config["experiment"].get("phase") != "checkpoint_cold_reload":
        errors.append("experiment.phase must be checkpoint_cold_reload")
    if checkpoint.get("preserve_source_checkpoint") is not True:
        errors.append("source checkpoint preservation must be enabled")
    if actor_dir != source_dir / "actor":
        errors.append("checkpoint.actor_dir must be the source checkpoint actor directory")
    if merged_dir == source_dir or source_dir in merged_dir.parents:
        errors.append("merged model directory must be outside the source checkpoint")
    if source_dir in output_dir.parents or source_dir in evidence_dir.parents:
        errors.append("reload outputs must be outside the source checkpoint")

    required_files = required_checkpoint_files(config, project_root)
    missing_checkpoint_files = [str(path) for path in required_files if not path.is_file()]
    if missing_checkpoint_files:
        errors.append(f"missing checkpoint files: {missing_checkpoint_files}")
    empty_checkpoint_files = [str(path) for path in required_files if path.is_file() and path.stat().st_size == 0]
    if empty_checkpoint_files:
        errors.append(f"empty checkpoint files: {empty_checkpoint_files}")
    if latest_iteration.is_file():
        value = latest_iteration.read_text(encoding="utf-8").strip()
        if value != str(checkpoint["global_step"]):
            errors.append(f"latest checkpoint iteration is {value}, expected {checkpoint['global_step']}")
    else:
        errors.append(f"latest checkpoint iteration file not found: {latest_iteration}")
    fsdp_config = actor_dir / "fsdp_config.json"
    if fsdp_config.is_file():
        value = json.loads(fsdp_config.read_text(encoding="utf-8"))
        if int(value.get("world_size", -1)) != int(checkpoint["world_size"]):
            errors.append("FSDP world size does not match reload config")

    if not eval_path.is_file():
        errors.append(f"eval Parquet not found: {eval_path}")
        selected_ids: list[str] = []
    else:
        if sha256_file(eval_path) != evaluation["input_parquet_sha256"]:
            errors.append("eval Parquet SHA256 does not match reload config")
        try:
            selected_ids, missing_images = frozen_eval_ids(
                eval_path,
                expected_rows=int(evaluation["expected_samples"]),
                seed=int(config["experiment"]["seed"]),
                count=int(evaluation["reload_samples"]),
            )
            if missing_images:
                errors.append(f"eval samples contain missing images: {missing_images[:20]}")
            if selected_ids != list(evaluation["sample_ids"]):
                errors.append("configured reload sample IDs do not match deterministic selection")
        except ValueError as exc:
            selected_ids = []
            errors.append(str(exc))
    if evaluation.get("selection_algorithm") != SELECTION_ALGORITHM:
        errors.append("reload sample selection algorithm is not frozen")
    if len(set(evaluation["sample_ids"])) != int(evaluation["reload_samples"]):
        errors.append("reload sample IDs must be unique and match reload_samples")
    if not chat_template.is_file():
        errors.append(f"chat template not found: {chat_template}")
    if int(serving["tensor_parallel_size"]) != int(gate["required_gpu_count"]):
        errors.append("serving tensor parallel size must equal required GPU count")
    if int(serving["max_model_len"]) != 8192:
        errors.append("reload serving max_model_len must remain 8192")

    free_disk = shutil.disk_usage(project_root).free
    if free_disk < int(gate["minimum_free_disk_gb"]) * GIB:
        errors.append("insufficient free disk for merged checkpoint and evidence")
    memory_bytes = memory_limit_bytes()
    if memory_bytes and memory_bytes < int(gate["minimum_cpu_memory_gb"]) * GIB:
        warnings.append(
            f"current CPU/cgroup memory limit is {memory_bytes / GIB:.2f} GiB; "
            f"--run requires at least {gate['minimum_cpu_memory_gb']} GiB"
        )
    if merged_dir.exists() and any(merged_dir.iterdir()):
        warnings.append(f"merged model directory is already non-empty: {merged_dir}")

    checks = {
        "source_checkpoint_exists": source_dir.is_dir(),
        "required_checkpoint_files_complete": not missing_checkpoint_files and not empty_checkpoint_files,
        "latest_iteration_matches": latest_iteration.is_file()
        and latest_iteration.read_text(encoding="utf-8").strip() == str(checkpoint["global_step"]),
        "eval_sha256_matches": eval_path.is_file()
        and sha256_file(eval_path) == evaluation["input_parquet_sha256"],
        "frozen_sample_ids_match": selected_ids == list(evaluation["sample_ids"]),
        "source_and_merged_paths_are_separate": merged_dir != source_dir and source_dir not in merged_dir.parents,
        "chat_template_exists": chat_template.is_file(),
    }
    return {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "experiment_id": config["experiment"]["id"],
        "phase": config["experiment"]["phase"],
        "status": "PASS" if not errors else "FAIL",
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "source_checkpoint": str(source_dir),
        "merged_model": str(merged_dir),
        "eval_parquet": str(eval_path),
        "sample_ids": selected_ids,
        "free_disk_gb": round(free_disk / GIB, 3),
        "cpu_memory_limit_gb": round(memory_bytes / GIB, 3) if memory_bytes else None,
        "checks": checks,
        "warnings": warnings,
        "errors": errors,
    }


def enforce_runtime(config: dict[str, Any], project_root: Path, *, reuse_merged: bool) -> None:
    gate = config["runtime_gate"]
    errors: list[str] = []
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gate["cuda_visible_devices"])
    try:
        import torch

        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except ImportError:
        gpu_count = 0
    if gpu_count != int(gate["required_gpu_count"]):
        errors.append(f"expected {gate['required_gpu_count']} visible GPUs, found {gpu_count}")
    memory_bytes = memory_limit_bytes()
    if memory_bytes and memory_bytes < int(gate["minimum_cpu_memory_gb"]) * GIB:
        errors.append(
            f"CPU/cgroup memory limit {memory_bytes / GIB:.2f} GiB is below "
            f"{gate['minimum_cpu_memory_gb']} GiB"
        )
    merged_dir = resolve(project_root, config["checkpoint"]["merged_model_dir"])
    if merged_dir.exists() and any(merged_dir.iterdir()) and not reuse_merged:
        errors.append("merged model directory is non-empty; use --reuse-merged only after auditing it")
    if errors:
        raise RuntimeError("; ".join(errors))


def run_command(command: list[str], log_path: Path, *, env: dict[str, str] | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as stream:
        stream.write("COMMAND=" + json.dumps(command, ensure_ascii=False) + "\n")
        stream.flush()
        subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT, env=env)


def port_is_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
        connection.settimeout(0.5)
        return connection.connect_ex((host, port)) == 0


def wait_for_server(process: subprocess.Popen[Any], url: str, timeout_seconds: int, log_path: Path) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_error = "not started"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
            raise RuntimeError(f"vLLM exited before readiness with code {process.returncode}:\n{tail}")
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return payload
        except (OSError, ValueError, urllib.error.URLError) as exc:
            last_error = repr(exc)
            time.sleep(2)
    raise TimeoutError(f"vLLM did not become ready within {timeout_seconds}s: {last_error}")


def stop_process_group(process: subprocess.Popen[Any]) -> int | None:
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
    return process.returncode


def verify_predictions(output_dir: Path, expected_ids: list[str]) -> dict[str, Any]:
    predictions_path = output_dir / "predictions.jsonl"
    summary_path = output_dir / "summary.json"
    if not predictions_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError("reload inference did not produce predictions.jsonl and summary.json")
    rows = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    actual_ids = [str(row.get("sample_id", "")) for row in rows]
    errors: list[str] = []
    if actual_ids != expected_ids:
        errors.append("prediction sample order does not match frozen reload sample IDs")
    if len(rows) != len(expected_ids):
        errors.append(f"expected {len(expected_ids)} predictions, found {len(rows)}")
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        if row.get("inference_error"):
            errors.append(f"{sample_id}: inference_error={row['inference_error']}")
        if not str(row.get("raw_prediction", "")).strip():
            errors.append(f"{sample_id}: empty response")
        if row.get("finish_reason") == "error":
            errors.append(f"{sample_id}: finish_reason=error")
        if int(row.get("response_token_count", 0)) <= 0:
            errors.append(f"{sample_id}: response_token_count is not positive")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if int(summary.get("total", -1)) != len(expected_ids):
        errors.append("summary total does not match reload sample count")
    if int(summary.get("unique_sample_ids", -1)) != len(expected_ids):
        errors.append("summary unique_sample_ids does not match reload sample count")
    return {
        "status": "PASS" if not errors else "FAIL",
        "prediction_count": len(rows),
        "sample_ids": actual_ids,
        "nonempty_response_count": sum(bool(str(row.get("raw_prediction", "")).strip()) for row in rows),
        "inference_error_count": sum(bool(row.get("inference_error")) for row in rows),
        "errors": errors,
    }


def execute_reload(
    config_path: Path,
    project_root: Path,
    *,
    reuse_merged: bool,
    overwrite_results: bool,
) -> dict[str, Any]:
    config = load_config(config_path)
    checkpoint = config["checkpoint"]
    serving = config["serving"]
    evaluation = config["evaluation"]
    evidence = config["evidence"]
    enforce_runtime(config, project_root, reuse_merged=reuse_merged)

    actor_dir = resolve(project_root, checkpoint["actor_dir"])
    merged_dir = resolve(project_root, checkpoint["merged_model_dir"])
    output_dir = resolve(project_root, evaluation["output_dir"])
    evidence_dir = resolve(project_root, evidence["directory"])
    evidence_dir.mkdir(parents=True, exist_ok=True)
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite_results:
        raise RuntimeError("reload output directory is non-empty; pass --overwrite-results to replace predictions")

    source_files = required_checkpoint_files(config, project_root)
    checkpoint_manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment"]["id"],
        "generated_at_utc": utc_now(),
        "global_step": checkpoint["global_step"],
        "world_size": checkpoint["world_size"],
        "files": file_snapshot(source_files, include_sha256=True),
    }
    write_json_atomic(resolve(project_root, evidence["checkpoint_manifest"]), checkpoint_manifest)
    source_snapshot_before = checkpoint_manifest["files"]

    if reuse_merged:
        merged_errors = validate_merged_model(merged_dir)
        if merged_errors:
            raise RuntimeError("cannot reuse merged model: " + "; ".join(merged_errors))
    else:
        if merged_dir.exists() and any(merged_dir.iterdir()):
            raise RuntimeError("refusing to overwrite non-empty merged model directory")
        merged_dir.mkdir(parents=True, exist_ok=True)
        run_command(
            [
                sys.executable,
                "-m",
                "verl.model_merger",
                "merge",
                "--backend",
                str(checkpoint["merge_backend"]),
                "--local_dir",
                str(actor_dir),
                "--target_dir",
                str(merged_dir),
            ],
            evidence_dir / "merge.log",
        )
    merged_errors = validate_merged_model(merged_dir)
    if merged_errors:
        raise RuntimeError("; ".join(merged_errors))
    merged_manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment"]["id"],
        "generated_at_utc": utc_now(),
        "source_checkpoint_manifest_sha256": sha256_file(
            resolve(project_root, evidence["checkpoint_manifest"])
        ),
        "directory": str(merged_dir),
        "files": directory_manifest(merged_dir),
    }
    write_json_atomic(resolve(project_root, evidence["merged_manifest"]), merged_manifest)

    host = str(serving["host"])
    port = int(serving["port"])
    if port_is_open(host, port):
        raise RuntimeError(f"refusing cold reload because {host}:{port} is already in use")
    vllm = shutil.which("vllm")
    if not vllm:
        raise FileNotFoundError("vllm executable was not found in the active environment")
    chat_template = resolve(project_root, config["shared"]["chat_template"]["file"])
    server_command = [
        vllm,
        "serve",
        str(merged_dir),
        "--served-model-name",
        str(serving["served_model_name"]),
        "--host",
        host,
        "--port",
        str(port),
        "--trust-remote-code",
        "--dtype",
        str(serving["dtype"]),
        "--max-model-len",
        str(serving["max_model_len"]),
        "--max-num-seqs",
        str(serving["max_num_seqs"]),
        "--tensor-parallel-size",
        str(serving["tensor_parallel_size"]),
        "--distributed-executor-backend",
        str(serving["distributed_executor_backend"]),
        "--gpu-memory-utilization",
        str(serving["gpu_memory_utilization"]),
        "--limit-mm-per-prompt",
        json.dumps(serving["limit_mm_per_prompt"], separators=(",", ":")),
        "--chat-template",
        str(chat_template),
        "--default-chat-template-kwargs",
        json.dumps(
            {"enable_thinking": bool(config["shared"]["chat_template"]["enable_thinking"])},
            separators=(",", ":"),
        ),
    ]
    server_log_path = resolve(project_root, evidence["server_log"])
    server_log_path.parent.mkdir(parents=True, exist_ok=True)
    server_process: subprocess.Popen[Any] | None = None
    server_exit_code: int | None = None
    started_at = utc_now()
    verification: dict[str, Any] = {"status": "FAIL", "errors": ["inference did not run"]}
    try:
        server_stream = server_log_path.open("w", encoding="utf-8")
        server_stream.write("COMMAND=" + json.dumps(server_command, ensure_ascii=False) + "\n")
        server_stream.flush()
        run_env = os.environ.copy()
        run_env["CUDA_VISIBLE_DEVICES"] = str(config["runtime_gate"]["cuda_visible_devices"])
        server_process = subprocess.Popen(
            server_command,
            stdout=server_stream,
            stderr=subprocess.STDOUT,
            env=run_env,
            start_new_session=True,
        )
        models_payload = wait_for_server(
            server_process,
            f"http://{host}:{port}/v1/models",
            int(serving["startup_timeout_seconds"]),
            server_log_path,
        )
        served_ids = [str(item.get("id", "")) for item in models_payload.get("data", [])]
        if str(serving["served_model_name"]) not in served_ids:
            raise RuntimeError(f"vLLM readiness response has unexpected model IDs: {served_ids}")

        inference_command = [
            sys.executable,
            str(project_root / "eval/run_internal_eval.py"),
            "--config",
            str(config_path),
            "--api-base",
            f"http://{host}:{port}/v1",
            "--model-id",
            str(serving["served_model_name"]),
            "--output-dir",
            str(output_dir),
        ]
        for sample_id in evaluation["sample_ids"]:
            inference_command.extend(["--sample-id", str(sample_id)])
        if overwrite_results:
            inference_command.append("--overwrite")
        run_command(inference_command, resolve(project_root, evidence["inference_log"]), env=run_env)
        verification = verify_predictions(output_dir, list(evaluation["sample_ids"]))
        if verification["status"] != "PASS":
            raise RuntimeError("reload prediction validation failed: " + "; ".join(verification["errors"]))
    finally:
        if server_process is not None:
            server_exit_code = stop_process_group(server_process)
        if "server_stream" in locals():
            server_stream.close()

    source_snapshot_after = file_snapshot(source_files, include_sha256=True)
    source_unchanged = source_snapshot_before == source_snapshot_after
    if not source_unchanged:
        raise RuntimeError("source checkpoint size, mtime, or SHA256 changed during cold reload")
    result = {
        "schema_version": 1,
        "experiment_id": config["experiment"]["id"],
        "phase": config["experiment"]["phase"],
        "status": "PASS",
        "started_at_utc": started_at,
        "completed_at_utc": utc_now(),
        "source_checkpoint": str(resolve(project_root, checkpoint["source_dir"])),
        "source_checkpoint_unchanged": source_unchanged,
        "merged_model": str(merged_dir),
        "merged_manifest_sha256": sha256_file(resolve(project_root, evidence["merged_manifest"])),
        "served_model_name": serving["served_model_name"],
        "server_exit_code_after_controlled_shutdown": server_exit_code,
        "verification": verification,
    }
    write_json_atomic(resolve(project_root, evidence["final_summary"]), result)
    return result


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=project_root / "configs/vopd_day8_reload.yaml")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--reuse-merged", action="store_true")
    parser.add_argument("--overwrite-results", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(config_path)
    preflight = validate_static(config_path, project_root)
    preflight_path = resolve(project_root, config["evidence"]["preflight_summary"])
    write_json_atomic(preflight_path, preflight)
    print(f"DAY8_RELOAD_PREFLIGHT={preflight['status']}")
    print(f"SUMMARY={preflight_path}")
    for warning in preflight["warnings"]:
        print(f"WARNING={warning}")
    if preflight["errors"]:
        for error in preflight["errors"]:
            print(f"ERROR={error}", file=sys.stderr)
        return 1
    if args.preflight_only:
        print("No checkpoint merge or GPU inference started.")
        return 0
    try:
        result = execute_reload(
            config_path,
            project_root,
            reuse_merged=args.reuse_merged,
            overwrite_results=args.overwrite_results,
        )
    except Exception as exc:
        failure = {
            "schema_version": 1,
            "experiment_id": config["experiment"]["id"],
            "phase": config["experiment"]["phase"],
            "status": "FAIL",
            "completed_at_utc": utc_now(),
            "error": f"{type(exc).__name__}: {exc}",
        }
        write_json_atomic(resolve(project_root, config["evidence"]["final_summary"]), failure)
        raise
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
