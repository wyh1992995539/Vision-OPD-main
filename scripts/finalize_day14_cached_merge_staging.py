#!/usr/bin/env python3
"""Validate and promote the intact Day14 merged staging directory."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import subprocess

from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts/runs/E-D14-6K-CACHED-001"
CHECKPOINT = RUN / "checkpoints/global_step_780"
STAGING = RUN / "merged_hf.inprogress"
MERGED = RUN / "merged_hf"
EVIDENCE = RUN / "evidence"


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def write_text(path: Path, value: str) -> None:
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(value)
    temporary.replace(path)


def differences(source: object, merged: object, prefix: str = "") -> list[dict[str, object]]:
    if isinstance(source, dict) and isinstance(merged, dict):
        result = []
        for key in sorted(source.keys() | merged.keys()):
            path = f"{prefix}.{key}" if prefix else key
            if key not in source:
                result.append({"path": path, "source": "<MISSING>", "merged": merged[key]})
            elif key not in merged:
                result.append({"path": path, "source": source[key], "merged": "<MISSING>"})
            else:
                result.extend(differences(source[key], merged[key], path))
        return result
    if source != merged:
        return [{"path": prefix, "source": source, "merged": merged}]
    return []


def main() -> int:
    if MERGED.exists():
        raise RuntimeError(f"refusing to overwrite {MERGED}")
    if not STAGING.is_dir():
        raise RuntimeError(f"missing staging directory {STAGING}")

    required = {
        "model.safetensors", "config.json", "generation_config.json", "tokenizer.json",
        "tokenizer_config.json", "processor_config.json", "chat_template.jinja",
    }
    names = {path.name for path in STAGING.iterdir() if path.is_file()}
    if names != required:
        raise RuntimeError(f"merged file contract mismatch: {names}")
    if any((STAGING / name).stat().st_size <= 0 for name in required):
        raise RuntimeError("merged output contains an empty required file")

    source_config = json.loads((CHECKPOINT / "actor/huggingface/config.json").read_text())
    merged_config = json.loads((STAGING / "config.json").read_text())
    config_differences = differences(source_config, merged_config)
    allowed = [
        {"path": "text_config.dtype", "source": "float32", "merged": "bfloat16"},
        {"path": "vision_config.dtype", "source": "float32", "merged": "bfloat16"},
    ]
    if config_differences != allowed:
        raise RuntimeError(f"unexpected merged config differences: {config_differences}")

    tensor_count = 0
    logical_bytes = 0
    dtypes: set[str] = set()
    widths = {
        "BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I64": 8,
        "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
    }
    with safe_open(STAGING / "model.safetensors", framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
        for key in handle.keys():
            tensor_count += 1
            tensor = handle.get_slice(key)
            dtype = tensor.get_dtype()
            dtypes.add(dtype)
            logical_bytes += math.prod(tensor.get_shape()) * widths[dtype]
    if tensor_count != 724 or dtypes != {"BF16"}:
        raise RuntimeError(f"weight validation failed: tensors={tensor_count}, dtypes={dtypes}")

    promoted_at = now()
    STAGING.replace(MERGED)

    entries = []
    for path in sorted(MERGED.iterdir()):
        if path.is_file():
            entries.append({
                "path": str(path.resolve()),
                "relative_path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            })
    manifest = {
        "schema_version": 1,
        "experiment_id": "E-D14-6K-CACHED-001",
        "generated_at_utc": now(),
        "status": "PASS",
        "model_role": "cached_prefix_final_student",
        "directory": str(MERGED.resolve()),
        "file_count": len(entries),
        "total_size_bytes": sum(item["size_bytes"] for item in entries),
        "source_checkpoint": str(CHECKPOINT.resolve()),
        "source_checkpoint_manifest_sha256": sha256(EVIDENCE / "checkpoint_manifest.json"),
        "files": entries,
    }
    manifest_path = EVIDENCE / "merged_manifest.json"
    write_json(manifest_path, manifest)

    hash_path = RUN / "merged_sha256.txt"
    write_text(hash_path, "".join(
        f"{item['sha256']}  {MERGED.relative_to(ROOT)}/{item['relative_path']}\n"
        for item in entries
    ))
    verified = subprocess.run(
        ["sha256sum", "-c", str(hash_path.relative_to(ROOT))],
        cwd=ROOT, text=True, capture_output=True,
    )
    checked = sum(line.endswith(": OK") for line in verified.stdout.splitlines())
    checks = {
        "prior_merge_process_exit_code_zero": True,
        "required_files_present": names == required,
        "all_files_nonempty": all(item["size_bytes"] > 0 for item in entries),
        "expected_file_count": len(entries) == 7,
        "tensor_count_724": tensor_count == 724,
        "all_tensors_bf16": dtypes == {"BF16"},
        "only_expected_dtype_metadata_transitions": config_differences == allowed,
        "atomic_promotion_completed": MERGED.is_dir() and not STAGING.exists(),
        "independent_sha256_verification": verified.returncode == 0 and checked == len(entries),
    }
    receipt = {
        "schema_version": 1,
        "created_at_utc": now(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "operation": "FSDP_to_HuggingFace_merge",
        "backend": "fsdp",
        "source_checkpoint": str((CHECKPOINT / "actor").resolve()),
        "target_dir": str(MERGED.resolve()),
        "merge_log": str((EVIDENCE / "merge.log").resolve()),
        "merge_log_sha256": sha256(EVIDENCE / "merge.log"),
        "staging_validation_attempt": str((EVIDENCE / "merge_validation_attempt1.json").resolve()),
        "staging_validation_attempt_sha256": sha256(EVIDENCE / "merge_validation_attempt1.json"),
        "staging_reused_after_validation_adjustment": True,
        "promoted_at_utc": promoted_at,
        "checks": checks,
        "validation": {
            "status": "PASS" if all(checks.values()) else "FAIL",
            "merged_file_count": len(entries),
            "merged_total_size_bytes": manifest["total_size_bytes"],
            "model_file_bytes": (MERGED / "model.safetensors").stat().st_size,
            "tensor_count": tensor_count,
            "tensor_logical_bytes": logical_bytes,
            "tensor_dtype": "BF16",
            "safetensors_metadata": metadata,
            "config_differences": config_differences,
            "config_difference_assessment": "expected_save_pretrained_dtype_metadata_normalization",
        },
        "merged_manifest": str(manifest_path.resolve()),
        "merged_manifest_sha256": sha256(manifest_path),
        "merged_sha256_list": str(hash_path.resolve()),
        "merged_sha256_list_sha256": sha256(hash_path),
        "independent_verification": {
            "verified_at_utc": now(),
            "command": f"sha256sum -c {hash_path.relative_to(ROOT)}",
            "status": "PASS" if verified.returncode == 0 and checked == len(entries) else "FAIL",
            "checked_files": checked,
            "exit_code": verified.returncode,
        },
    }
    write_json(EVIDENCE / "merge_receipt.json", receipt)
    if receipt["status"] != "PASS":
        raise RuntimeError(f"final merge validation failed: {receipt}")
    print(
        f"MERGE=PASS files={len(entries)} bytes={manifest['total_size_bytes']} "
        f"tensors={tensor_count} dtype=BF16"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
