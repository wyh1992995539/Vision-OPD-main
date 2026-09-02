"""Auditable coverage receipt for zero-weight padded training batches."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def _sample_id(extra_info: Any) -> str:
    if not isinstance(extra_info, dict):
        raise ValueError("extra_info must be a mapping")
    provenance = extra_info.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("extra_info.provenance must be a mapping")
    value = str(provenance.get("sample_id", "")).strip()
    if not value:
        raise ValueError("extra_info.provenance.sample_id is empty")
    return value


def _atomic_write(path: Path, receipt: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def update_coverage_receipt(
    receipt_path: str | Path,
    *,
    extra_infos: list[Any],
    sample_weights: Any,
    global_step: int,
    expected_unique_samples: int,
    expected_padding_rows: int,
    final_step: bool,
) -> dict[str, Any]:
    path = Path(receipt_path)
    weights = (
        sample_weights.detach().cpu().reshape(-1).tolist()
        if hasattr(sample_weights, "detach")
        else list(sample_weights)
    )
    if len(extra_infos) != len(weights):
        raise ValueError("coverage batch metadata and sample weights have different lengths")
    if any(weight not in (0.0, 1.0) for weight in weights):
        raise ValueError("sample_weight must be exactly 0 or 1")
    real_ids = [
        _sample_id(extra_info)
        for extra_info, weight in zip(extra_infos, weights, strict=True)
        if weight == 1.0
    ]
    step_record = {
        "global_step": int(global_step),
        "real_sample_ids": real_ids,
        "padding_rows": sum(weight == 0.0 for weight in weights),
    }

    if path.is_file():
        receipt = json.loads(path.read_text(encoding="utf-8"))
    else:
        receipt = {
            "schema_version": 1,
            "status": "IN_PROGRESS",
            "expected_unique_samples": expected_unique_samples,
            "expected_padding_rows": expected_padding_rows,
            "steps": [],
        }
    if receipt["expected_unique_samples"] != expected_unique_samples:
        raise ValueError("coverage receipt expected_unique_samples changed")
    if receipt["expected_padding_rows"] != expected_padding_rows:
        raise ValueError("coverage receipt expected_padding_rows changed")

    steps = receipt.setdefault("steps", [])
    prior = next((item for item in steps if item["global_step"] == global_step), None)
    if prior is not None and prior == step_record:
        return receipt
    if prior is not None and prior != step_record:
        steps[:] = [item for item in steps if item["global_step"] < global_step]
    elif steps and global_step < steps[-1]["global_step"]:
        steps[:] = [item for item in steps if item["global_step"] < global_step]
    elif steps and global_step == steps[-1]["global_step"]:
        raise ValueError("coverage receipt contains a conflicting batch for this step")
    steps.append(step_record)

    all_ids = [sample_id for item in steps for sample_id in item["real_sample_ids"]]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("a real sample appears more than once in coverage receipt")
    receipt["seen_sample_ids"] = all_ids
    receipt["unique_source_seen"] = len(set(all_ids))
    receipt["effective_train_samples"] = len(all_ids)
    receipt["padding_rows"] = sum(item["padding_rows"] for item in steps)
    receipt["last_global_step"] = max(item["global_step"] for item in steps)
    receipt["dropped_rows"] = None
    receipt["status"] = "IN_PROGRESS"

    error = None
    if final_step:
        receipt["dropped_rows"] = expected_unique_samples - receipt["unique_source_seen"]
        checks = {
            "unique_source_seen": receipt["unique_source_seen"] == expected_unique_samples,
            "effective_train_samples": receipt["effective_train_samples"] == expected_unique_samples,
            "padding_rows": receipt["padding_rows"] == expected_padding_rows,
            "dropped_rows": receipt["dropped_rows"] == 0,
        }
        receipt["checks"] = checks
        receipt["status"] = "PASS" if all(checks.values()) else "FAIL"
        if receipt["status"] != "PASS":
            error = ValueError(f"full coverage contract failed: {checks}")
    _atomic_write(path, receipt)
    if error is not None:
        raise error
    return receipt
