#!/usr/bin/env python3
"""Build a deterministic, auditable training subset for Vision-OPD stability runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


SELECTION_ALGORITHM = "sha256(seed|sample_id),ascending"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_key(seed: int, sample_id: str) -> str:
    return hashlib.sha256(f"{seed}|{sample_id}".encode("utf-8")).hexdigest()


def provenance(row: dict[str, Any], row_index: int) -> dict[str, Any]:
    extra_info = row.get("extra_info")
    if not isinstance(extra_info, dict):
        raise ValueError(f"row {row_index}: extra_info must be a struct")
    value = extra_info.get("provenance")
    if not isinstance(value, dict):
        raise ValueError(f"row {row_index}: extra_info.provenance must be a struct")
    return value


def select_row_indices(table: pa.Table, *, count: int, seed: int) -> list[int]:
    if count <= 0:
        raise ValueError("count must be positive")
    if count > table.num_rows:
        raise ValueError(f"cannot select {count} rows from a {table.num_rows}-row table")

    ranked: list[tuple[str, str, int]] = []
    seen: set[str] = set()
    for row_index, row in enumerate(table.select(["extra_info"]).to_pylist()):
        sample_id = str(provenance(row, row_index).get("sample_id", "")).strip()
        if not sample_id:
            raise ValueError(f"row {row_index}: sample_id must be non-empty")
        if sample_id in seen:
            raise ValueError(f"duplicate sample_id: {sample_id}")
        seen.add(sample_id)
        ranked.append((stable_key(seed, sample_id), sample_id, row_index))

    ranked.sort()
    return [row_index for _key, _sample_id, row_index in ranked[:count]]


def _write_parquet_atomic(table: pa.Table, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        pq.write_table(table, temporary_path, compression="snappy")
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_json_atomic(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def build_subset(
    source_path: Path,
    output_path: Path,
    manifest_path: Path,
    *,
    count: int = 64,
    seed: int = 42,
    overwrite: bool = False,
) -> dict[str, Any]:
    source_path = source_path.resolve()
    output_path = output_path.resolve()
    manifest_path = manifest_path.resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"source Parquet not found: {source_path}")
    for path in (output_path, manifest_path):
        if path.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite existing output: {path}")

    source = pq.read_table(source_path)
    if "extra_info" not in source.column_names:
        raise ValueError("source Parquet is missing extra_info")
    selected_indices = select_row_indices(source, count=count, seed=seed)
    subset = source.take(pa.array(selected_indices, type=pa.int64()))
    metadata = dict(subset.schema.metadata or {})
    metadata.update(
        {
            b"vision_opd_subset_algorithm": SELECTION_ALGORITHM.encode("utf-8"),
            b"vision_opd_subset_seed": str(seed).encode("ascii"),
            b"vision_opd_subset_count": str(count).encode("ascii"),
        }
    )
    subset = subset.replace_schema_metadata(metadata)
    _write_parquet_atomic(subset, output_path)

    selection: list[dict[str, Any]] = []
    for selection_index, (source_row_index, row) in enumerate(
        zip(selected_indices, subset.select(["extra_info"]).to_pylist(), strict=True)
    ):
        item = provenance(row, source_row_index)
        sample_id = str(item["sample_id"])
        selection.append(
            {
                "selection_index": selection_index,
                "source_row_index": source_row_index,
                "sample_id": sample_id,
                "source_id": str(item.get("source_id", "")),
                "source_row": int(item.get("source_row", -1)),
                "group_id": str(item.get("group_id", "")),
                "selection_key": stable_key(seed, sample_id),
            }
        )

    manifest = {
        "schema_version": 1,
        "experiment_id": "E-D8-001",
        "purpose": "fixed_64_training_stability_subset",
        "selection": {
            "algorithm": SELECTION_ALGORITHM,
            "seed": seed,
            "requested_rows": count,
            "selected_rows": len(selection),
            "preserve_ranked_order": True,
        },
        "source": {
            "path": str(source_path),
            "rows": source.num_rows,
            "sha256": sha256_file(source_path),
        },
        "output": {
            "path": str(output_path),
            "rows": subset.num_rows,
            "sha256": sha256_file(output_path),
        },
        "samples": selection,
    }
    _write_json_atomic(manifest, manifest_path)
    return manifest


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("/root/autodl-tmp/data/vision_opd_1024/train_1024.parquet"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/root/autodl-tmp/data/vision_opd_1024/train_day8_64.parquet"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=repository_root
        / "artifacts/runs/E-D8-001/preflight/day8_64_selection.json",
    )
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_subset(
        args.source,
        args.output,
        args.manifest,
        count=args.count,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print("DAY8_SUBSET=PASS")
    print(f"ROWS={manifest['output']['rows']}")
    print(f"SHA256={manifest['output']['sha256']}")
    print(f"MANIFEST={args.manifest.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
