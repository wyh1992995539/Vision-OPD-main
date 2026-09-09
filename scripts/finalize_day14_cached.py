#!/usr/bin/env python3
"""Repair, freeze, and merge the completed Day 14 Cached Prefix run."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any

import yaml
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RUN = ROOT / "artifacts/runs/E-D14-6K-CACHED-001"
CHECKPOINT = RUN / "checkpoints/global_step_780"
MERGED = RUN / "merged_hf"
POLICY = ROOT / "configs/cached_prefix_6241_abort_policy.yaml"
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
NUMBER = r"-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
PAIR_RE = re.compile(rf"(?:^| - )([^:]+):({NUMBER})")
STEP_RE = re.compile(r"step:(\d+) - ")
FATAL_RE = re.compile(
    r"CUDA out of memory|Traceback|\bnan\b|NCCL.*(?:error|abort)|"
    r"Training failed|Segmentation fault|DataLoader worker .*killed by signal",
    re.IGNORECASE,
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def parse_log() -> tuple[dict[int, dict[str, float]], list[str]]:
    rows: dict[int, dict[str, float]] = {}
    fatal: list[str] = []
    for raw_line in (RUN / "logs/train.log").open(errors="replace"):
        line = ANSI_RE.sub("", raw_line).replace("\r", "")
        match = STEP_RE.search(line)
        if match:
            rows[int(match.group(1))] = {
                key: float(value) for key, value in PAIR_RE.findall(line)
            }
        if FATAL_RE.search(line):
            fatal.append(line.strip()[-500:])
    return rows, fatal


def audit_rows(rows: dict[int, dict[str, float]], fatal: list[str]) -> dict[str, Any]:
    missing = [step for step in range(1, 781) if step not in rows]
    gate_specs = {
        "cached_prefix/resolved_records": 8.0,
        "cached_prefix/online_generation_calls": 0.0,
        "actor/policy_fallback_fraction": 0.0,
        "self_distillation/empty_target_batch": 0.0,
        "evidence/ema_update_applied": 1.0,
    }
    gate_checks: dict[str, Any] = {}
    for key, expected in gate_specs.items():
        values = [rows[step].get(key) for step in range(1, 781) if step in rows]
        gate_checks[key] = {
            "observed_steps": len(values),
            "expected_value": expected,
            "all_match": len(values) == 780 and all(value == expected for value in values),
            "min": min(values) if values and all(value is not None for value in values) else None,
            "max": max(values) if values and all(value is not None for value in values) else None,
        }
    finite_keys = ["actor/vopd_loss", "self_distillation/raw_jsd_token_mean", "actor/grad_norm"]
    finite_checks = {
        key: len([rows[s].get(key) for s in range(1, 781) if s in rows]) == 780
        and all(math.isfinite(rows[s][key]) for s in range(1, 781))
        for key in finite_keys
    }
    checks = {
        "steps_1_through_780_complete": not missing and len(rows) == 780,
        "fatal_log_matches_zero": not fatal,
        "all_runtime_gates_match": all(item["all_match"] for item in gate_checks.values()),
        "key_metrics_finite": all(finite_checks.values()),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "unique_steps": len(rows),
        "first_step": min(rows) if rows else None,
        "latest_step": max(rows) if rows else None,
        "missing_steps": missing,
        "fatal_log_matches": fatal,
        "runtime_gates": gate_checks,
        "finite_metrics": finite_checks,
    }


def backup_once(path: Path, suffix: str) -> Path:
    backup = path.with_name(path.stem + suffix + path.suffix)
    if not backup.exists():
        shutil.copy2(path, backup)
    return backup


def repair_receipts() -> dict[str, Any]:
    rows, fatal = parse_log()
    audit = audit_rows(rows, fatal)
    if audit["status"] != "PASS":
        raise RuntimeError(f"full log replay failed: {audit}")

    guard_path = RUN / "evidence/guard_summary.json"
    exit_path = RUN / "evidence/exit_receipt.json"
    original_guard = backup_once(guard_path, ".pre_final_drain")
    original_exit = backup_once(exit_path, ".pre_final_drain")
    guard = load_json(original_guard)
    receipt = load_json(original_exit)
    if guard.get("status") != "PASS" or guard.get("return_code") != 0:
        raise RuntimeError("original guard did not pass")
    if receipt.get("guard_exit_code") != 0:
        raise RuntimeError("original exit receipt did not pass")

    runtime_path = RUN / "evidence/runtime_metrics.jsonl"
    metric_rows = [json.loads(line) for line in runtime_path.read_text().splitlines() if line.strip()]
    before_steps = {int(item["step"]) for item in metric_rows}
    if 780 not in before_steps:
        from scripts.monitor_vopd_training import parse_training_metric_line
        final_line = None
        for raw_line in (RUN / "logs/train.log").open(errors="replace"):
            parsed = parse_training_metric_line(raw_line)
            if parsed and parsed["step"] == 780:
                final_line = parsed
        if final_line is None:
            raise RuntimeError("could not parse step 780 for runtime metrics repair")
        with runtime_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({
                "timestamp_utc": utc_now(),
                **final_line,
                "credential_correction": "post_exit_final_log_drain",
            }, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    after_rows = [json.loads(line) for line in runtime_path.read_text().splitlines() if line.strip()]
    after_steps = {int(item["step"]) for item in after_rows}
    if after_steps != set(range(1, 781)):
        raise RuntimeError("runtime_metrics does not contain exactly steps 1..780 after repair")

    correction = {
        "applied_at_utc": utc_now(),
        "reason": "guard exited before draining the final training log line",
        "method": "full_log_replay_and_post_exit_final_log_drain",
        "original_latest_step": guard.get("latest_step"),
        "corrected_latest_step": 780,
        "original_guard": {"path": str(original_guard), "sha256": sha256_file(original_guard)},
        "original_exit_receipt": {"path": str(original_exit), "sha256": sha256_file(original_exit)},
        "correction_script": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
        "runtime_monitor_after_fix": {
            "path": str((ROOT / "scripts/monitor_vopd_training.py").resolve()),
            "sha256": sha256_file(ROOT / "scripts/monitor_vopd_training.py"),
        },
        "runtime_metric_steps_before": len(before_steps),
        "runtime_metric_steps_after": len(after_steps),
        "full_log_replay": audit,
    }
    guard["latest_step"] = 780
    guard["latest_step_source"] = "post_exit_full_log_replay"
    guard["credential_correction"] = correction
    write_json(guard_path, guard)
    receipt["training_summary"] = guard
    receipt["credential_correction"] = correction
    write_json(exit_path, receipt)
    return correction


def checkpoint_files() -> list[Path]:
    policy = yaml.safe_load(POLICY.read_text())
    expected = [CHECKPOINT / rel for rel in policy["checkpoint"]["required_relative_files"]]
    missing = [str(path) for path in expected if not path.is_file() or path.stat().st_size <= 0]
    actual = sorted(path for path in CHECKPOINT.rglob("*") if path.is_file())
    if missing or set(actual) != set(expected):
        raise RuntimeError(f"checkpoint file contract mismatch; missing={missing}; extras={set(actual)-set(expected)}")
    return sorted(expected)


def freeze_checkpoint() -> dict[str, Any]:
    files = checkpoint_files()
    before = {path: (path.stat().st_size, path.stat().st_mtime_ns) for path in files}
    entries = []
    for path in files:
        entries.append({
            "path": str(path.resolve()),
            "relative_path": str(path.relative_to(CHECKPOINT)),
            "size_bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
            "sha256": sha256_file(path),
        })
    after = {path: (path.stat().st_size, path.stat().st_mtime_ns) for path in files}
    if before != after:
        raise RuntimeError("checkpoint files changed while hashing")
    guard = load_json(RUN / "evidence/guard_summary.json")
    marker = (RUN / "checkpoints/latest_checkpointed_iteration.txt").read_text().strip()
    manifest = {
        "schema_version": 1,
        "experiment_id": "E-D14-6K-CACHED-001",
        "generated_at_utc": utc_now(),
        "status": "PASS",
        "checkpoint_directory": str(CHECKPOINT.resolve()),
        "global_step": 780,
        "world_size": 2,
        "file_count": len(entries),
        "total_size_bytes": sum(item["size_bytes"] for item in entries),
        "source_guard_summary": {
            "path": str((RUN / "evidence/guard_summary.json").resolve()),
            "sha256": sha256_file(RUN / "evidence/guard_summary.json"),
            "status": guard.get("status"),
            "return_code": guard.get("return_code"),
            "latest_step": guard.get("latest_step"),
            "checkpoint_status": guard.get("checkpoint", {}).get("status"),
        },
        "files": entries,
    }
    manifest_path = RUN / "evidence/checkpoint_manifest.json"
    write_json(manifest_path, manifest)
    hash_path = RUN / "checkpoint_sha256.txt"
    write_text(hash_path, "".join(f"{item['sha256']}  {relative(Path(item['path']))}\n" for item in entries))
    result = subprocess.run(["sha256sum", "-c", relative(hash_path)], cwd=ROOT, text=True, capture_output=True)
    checked = sum(line.endswith(": OK") for line in result.stdout.splitlines())
    checks = {
        "source_guard_pass": guard.get("status") == "PASS" and guard.get("return_code") == 0,
        "checkpoint_guard_pass": guard.get("checkpoint", {}).get("status") == "PASS",
        "latest_step_matches": guard.get("latest_step") == 780,
        "marker_matches": marker == "780",
        "expected_file_count": len(entries) == 13,
        "all_files_nonempty": all(item["size_bytes"] > 0 for item in entries),
        "source_files_stable_while_hashing": before == after,
        "independent_sha256_verification": result.returncode == 0 and checked == len(entries),
    }
    receipt = {
        "schema_version": 1,
        "experiment_id": "E-D14-6K-CACHED-001",
        "generated_at_utc": utc_now(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "source_checkpoint": str(CHECKPOINT.resolve()),
        "global_step": 780,
        "file_count": len(entries),
        "total_size_bytes": manifest["total_size_bytes"],
        "checks": checks,
        "manifest": {"path": str(manifest_path.resolve()), "sha256": sha256_file(manifest_path)},
        "sha256_list": {"path": str(hash_path.resolve()), "sha256": sha256_file(hash_path)},
        "independent_verification": {
            "verified_at_utc": utc_now(),
            "command": f"sha256sum -c {relative(hash_path)}",
            "status": "PASS" if result.returncode == 0 and checked == len(entries) else "FAIL",
            "checked_files": checked,
            "exit_code": result.returncode,
        },
    }
    write_json(RUN / "evidence/checkpoint_hash_receipt.json", receipt)
    if receipt["status"] != "PASS":
        raise RuntimeError(f"checkpoint freeze failed: {receipt}")
    return receipt


def directory_manifest(directory: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path.resolve()),
            "relative_path": str(path.relative_to(directory)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(directory.rglob("*")) if path.is_file()
    ]


def merge_checkpoint() -> dict[str, Any]:
    checkpoint_receipt = load_json(RUN / "evidence/checkpoint_hash_receipt.json")
    if checkpoint_receipt.get("status") != "PASS":
        raise RuntimeError("checkpoint hash receipt is not PASS")
    inprogress = MERGED.with_name(MERGED.name + ".inprogress")
    if MERGED.exists() or inprogress.exists():
        raise RuntimeError("merged target already exists; refusing to overwrite")
    command = [
        sys.executable, "-m", "verl.model_merger", "merge", "--backend", "fsdp",
        "--local_dir", str(CHECKPOINT / "actor"), "--target_dir", str(inprogress),
    ]
    log_path = RUN / "evidence/merge.log"
    started = utc_now()
    with log_path.open("w", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    finished = utc_now()
    if result.returncode != 0:
        raise RuntimeError(f"merge failed with exit code {result.returncode}; see {log_path}")
    required = {
        "model.safetensors", "config.json", "generation_config.json", "tokenizer.json",
        "tokenizer_config.json", "processor_config.json", "chat_template.jinja",
    }
    actual_names = {path.name for path in inprogress.iterdir() if path.is_file()}
    if actual_names != required or any((inprogress / name).stat().st_size <= 0 for name in required):
        raise RuntimeError(f"merged file contract mismatch: {actual_names}")
    source_config = json.loads((CHECKPOINT / "actor/huggingface/config.json").read_text())
    merged_config = json.loads((inprogress / "config.json").read_text())
    tensor_count = 0
    logical_bytes = 0
    dtypes: set[str] = set()
    with safe_open(inprogress / "model.safetensors", framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
        for key in handle.keys():
            tensor_count += 1
            tensor = handle.get_slice(key)
            shape = tensor.get_shape()
            dtype = tensor.get_dtype()
            dtypes.add(dtype)
            elements = math.prod(shape)
            widths = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}
            logical_bytes += elements * widths[dtype]
    if tensor_count != 724 or dtypes != {"BF16"} or source_config != merged_config:
        raise RuntimeError(f"merged model validation failed: tensors={tensor_count}, dtypes={dtypes}, config={source_config == merged_config}")
    inprogress.replace(MERGED)
    entries = directory_manifest(MERGED)
    manifest = {
        "schema_version": 1,
        "experiment_id": "E-D14-6K-CACHED-001",
        "generated_at_utc": utc_now(),
        "status": "PASS",
        "model_role": "cached_prefix_final_student",
        "directory": str(MERGED.resolve()),
        "file_count": len(entries),
        "total_size_bytes": sum(item["size_bytes"] for item in entries),
        "source_checkpoint": str(CHECKPOINT.resolve()),
        "source_checkpoint_manifest_sha256": sha256_file(RUN / "evidence/checkpoint_manifest.json"),
        "files": entries,
    }
    manifest_path = RUN / "evidence/merged_manifest.json"
    write_json(manifest_path, manifest)
    hash_path = RUN / "merged_sha256.txt"
    write_text(hash_path, "".join(f"{item['sha256']}  {relative(Path(item['path']))}\n" for item in entries))
    verified = subprocess.run(["sha256sum", "-c", relative(hash_path)], cwd=ROOT, text=True, capture_output=True)
    checked = sum(line.endswith(": OK") for line in verified.stdout.splitlines())
    checks = {
        "merge_exit_code_zero": result.returncode == 0,
        "required_files_present": actual_names == required,
        "all_files_nonempty": all(item["size_bytes"] > 0 for item in entries),
        "expected_file_count": len(entries) == 7,
        "tensor_count_724": tensor_count == 724,
        "all_tensors_bf16": dtypes == {"BF16"},
        "config_matches_source": source_config == merged_config,
        "independent_sha256_verification": verified.returncode == 0 and checked == len(entries),
    }
    receipt = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "operation": "FSDP_to_HuggingFace_merge",
        "backend": "fsdp",
        "source_checkpoint": str((CHECKPOINT / "actor").resolve()),
        "source_checkpoint_manifest": str((RUN / "evidence/checkpoint_manifest.json").resolve()),
        "source_checkpoint_manifest_sha256": sha256_file(RUN / "evidence/checkpoint_manifest.json"),
        "target_dir": str(MERGED.resolve()),
        "command": "python -m verl.model_merger merge --backend fsdp --local_dir <checkpoint>/actor --target_dir <run>/merged_hf.inprogress",
        "merge_started_at_utc": started,
        "merge_finished_at_utc": finished,
        "merge_exit_code": result.returncode,
        "merge_log": str(log_path.resolve()),
        "merge_log_sha256": sha256_file(log_path),
        "checks": checks,
        "validation": {
            "status": "PASS" if all(checks.values()) else "FAIL",
            "merged_file_count": len(entries),
            "merged_total_size_bytes": manifest["total_size_bytes"],
            "model_file_bytes": (MERGED / "model.safetensors").stat().st_size,
            "tensor_count": tensor_count,
            "tensor_logical_bytes": logical_bytes,
            "tensor_dtype": next(iter(dtypes)) if len(dtypes) == 1 else sorted(dtypes),
            "safetensors_metadata": metadata,
            "config_matches_source": source_config == merged_config,
        },
        "merged_manifest": str(manifest_path.resolve()),
        "merged_manifest_sha256": sha256_file(manifest_path),
        "merged_sha256_list": str(hash_path.resolve()),
        "merged_sha256_list_sha256": sha256_file(hash_path),
        "independent_verification": {
            "verified_at_utc": utc_now(),
            "command": f"sha256sum -c {relative(hash_path)}",
            "status": "PASS" if verified.returncode == 0 and checked == len(entries) else "FAIL",
            "checked_files": checked,
            "exit_code": verified.returncode,
        },
    }
    write_json(RUN / "evidence/merge_receipt.json", receipt)
    if receipt["status"] != "PASS":
        raise RuntimeError(f"merge verification failed: {receipt}")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repair-receipts", action="store_true")
    parser.add_argument("--freeze-checkpoint", action="store_true")
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    if not any((args.repair_receipts, args.freeze_checkpoint, args.merge)):
        parser.error("select at least one phase")
    if args.repair_receipts:
        result = repair_receipts()
        print(f"RECEIPTS=PASS latest_step={result['corrected_latest_step']}", flush=True)
    if args.freeze_checkpoint:
        result = freeze_checkpoint()
        print(f"CHECKPOINT_SHA256=PASS files={result['file_count']} bytes={result['total_size_bytes']}", flush=True)
    if args.merge:
        result = merge_checkpoint()
        print(f"MERGE=PASS files={result['validation']['merged_file_count']} bytes={result['validation']['merged_total_size_bytes']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
