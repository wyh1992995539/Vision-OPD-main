#!/usr/bin/env python3
"""Prepare ZoomBench/MMStar while preserving official source image bytes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from eval.paper_aligned_common import load_config, resolve_path, sha256_file, write_json


def image_bytes(value: Any) -> bytes:
    if isinstance(value, dict):
        raw = value.get("bytes")
        if isinstance(raw, (bytes, bytearray)):
            return bytes(raw)
        source = value.get("path")
        if source and Path(str(source)).is_file():
            return Path(str(source)).read_bytes()
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    raise ValueError("row does not contain source image bytes")


def write_source_bytes(path: Path, raw: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(raw)
    os.replace(temporary, path)
    return hashlib.sha256(raw).hexdigest()


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def prepare_zoombench(config: dict[str, Any], force: bool) -> dict[str, Any]:
    import pyarrow.parquet as pq

    benchmark = config["benchmarks"]["zoombench"]
    source = resolve_path(benchmark["source_parquet"])
    output = resolve_path(benchmark["converted_json"])
    if output.exists() and not force:
        raise FileExistsError(f"refusing to overwrite {output}; pass --force")
    parquet = pq.ParquetFile(source)
    required = {"id", "query", "response", "question_type", "image", "crop_image"}
    missing = required.difference(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"ZoomBench source missing columns: {sorted(missing)}")
    expected = int(benchmark["expected_sample_count"])
    if parquet.metadata.num_rows != expected:
        raise ValueError(f"ZoomBench expected {expected} rows, found {parquet.metadata.num_rows}")

    full_dir = output.parent / "images" / "full"
    crop_dir = output.parent / "images" / "crop"
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    columns = ["id", "query", "response", "question_type", "image", "crop_image"]
    for batch in parquet.iter_batches(batch_size=1, columns=columns, use_threads=False):
        row = batch.to_pylist()[0]
        source_id = str(row.get("id"))
        if not source_id or source_id == "None" or source_id in seen:
            raise ValueError(f"invalid or duplicate ZoomBench id: {source_id!r}")
        seen.add(source_id)
        full_path = full_dir / f"{source_id}.jpg"
        crop_path = crop_dir / f"{source_id}.jpg"
        full_hash = write_source_bytes(full_path, image_bytes(row.get("image")))
        crop_raw = image_bytes(row.get("crop_image"))
        crop_hash = write_source_bytes(crop_path, crop_raw)
        question_type = str(row.get("question_type") or "").casefold()
        rows.append(
            {
                "schema_version": 1,
                "benchmark": "zoombench",
                "sample_uid": f"zoombench:source_id:{source_id}",
                "source_id": source_id,
                "source_repo_id": benchmark["dataset_repo_id"],
                "source_revision": benchmark["dataset_revision"],
                "source_split": benchmark["split"],
                "query": str(row.get("query") or "").strip(),
                "response": str(row.get("response") or "").strip(),
                "images": [str(full_path.resolve())],
                "crop_images": [str(crop_path.resolve())],
                "question_format": "multiple_choice" if question_type == "mcq" else "open_question",
                "category": "multiple_choice" if question_type == "mcq" else "open_question",
                "l2_category": "unavailable_official",
                "image_sha256": full_hash,
                "crop_image_sha256": crop_hash,
            }
        )
        if len(rows) == 1 or len(rows) % 25 == 0:
            print(f"Prepared ZoomBench rows: {len(rows)}/{expected}", flush=True)
    write_rows(output, rows)
    return evidence("zoombench", benchmark, source, output, rows)


def prepare_mmstar(config: dict[str, Any], force: bool) -> dict[str, Any]:
    import pyarrow.parquet as pq

    benchmark = config["benchmarks"]["mmstar"]
    source = resolve_path(benchmark["source_parquet"])
    output = resolve_path(benchmark["converted_json"])
    if output.exists() and not force:
        raise FileExistsError(f"refusing to overwrite {output}; pass --force")
    parquet = pq.ParquetFile(source)
    columns = ["index", "question", "answer", "category", "l2_category", "image"]
    missing = set(columns).difference(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"MMStar source missing columns: {sorted(missing)}")
    expected = int(benchmark["expected_sample_count"])
    if parquet.metadata.num_rows != expected:
        raise ValueError(f"MMStar expected {expected} rows, found {parquet.metadata.num_rows}")

    image_dir = output.parent / "images" / "full"
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for batch in parquet.iter_batches(batch_size=1, columns=columns, use_threads=False):
        row = batch.to_pylist()[0]
        source_id = str(row.get("index"))
        if not source_id or source_id == "None" or source_id in seen:
            raise ValueError(f"invalid or duplicate MMStar index: {source_id!r}")
        seen.add(source_id)
        answer = str(row.get("answer") or "").strip().upper()
        if answer not in {"A", "B", "C", "D"}:
            raise ValueError(f"MMStar {source_id}: invalid answer {answer!r}")
        image_path = image_dir / f"{int(source_id):05d}.jpg"
        image_hash = write_source_bytes(image_path, image_bytes(row.get("image")))
        rows.append(
            {
                "schema_version": 1,
                "benchmark": "mmstar",
                "sample_uid": f"mmstar:source_id:{source_id}",
                "source_id": source_id,
                "source_repo_id": benchmark["dataset_repo_id"],
                "source_revision": benchmark["dataset_revision"],
                "source_split": benchmark["split"],
                "query": str(row.get("question") or "").strip(),
                "response": answer,
                "images": [str(image_path.resolve())],
                "crop_images": [],
                "question_format": "multiple_choice",
                "category": str(row.get("category") or "unknown").strip(),
                "l2_category": str(row.get("l2_category") or "unknown").strip(),
                "image_sha256": image_hash,
            }
        )
        if len(rows) == 1 or len(rows) % 100 == 0:
            print(f"Prepared MMStar rows: {len(rows)}/{expected}", flush=True)
    write_rows(output, rows)
    return evidence("mmstar", benchmark, source, output, rows)


def evidence(
    name: str,
    benchmark: dict[str, Any],
    source: Path,
    output: Path,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "status": "pass",
        "benchmark": name,
        "dataset_repo_id": benchmark["dataset_repo_id"],
        "dataset_revision": benchmark["dataset_revision"],
        "source_parquet": str(source),
        "source_parquet_sha256": sha256_file(source),
        "converted_json": str(output),
        "converted_json_sha256": sha256_file(output),
        "record_count": len(rows),
        "unique_sample_uid_count": len({row["sample_uid"] for row in rows}),
        "image_policy": "preserve_source_bytes",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/benchmark_eval_paper_basejudge_r3_single_gpu.yaml")
    parser.add_argument("--benchmarks", default="zoombench,mmstar")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    _, config = load_config(args.config)
    selected = [item.strip() for item in args.benchmarks.split(",") if item.strip()]
    if not selected or set(selected).difference({"zoombench", "mmstar"}):
        raise ValueError("--benchmarks accepts only zoombench,mmstar")
    results = []
    for name in selected:
        results.append(prepare_zoombench(config, args.force) if name == "zoombench" else prepare_mmstar(config, args.force))
    document = {"schema_version": 1, "status": "pass", "benchmarks": results}
    output = resolve_path(config["paths"]["run_root"]) / "preflight" / "paper_aligned_primary_data_preparation.json"
    write_json(output, document)
    print(json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
