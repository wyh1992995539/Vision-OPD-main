#!/usr/bin/env python3
"""Prepare the pinned lmms-lab V* mirror for paper-aligned evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

from PIL import Image

from eval.paper_aligned_common import (
    load_config,
    require_frozen_r3_config,
    resolve_path,
    sha256_file,
    write_json,
)


def safe_name(value: Any) -> str:
    text = str(value)
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in text)


def append_suffix(text: str, suffix: str) -> str:
    text = text.strip()
    suffix = suffix.strip()
    return text if text.endswith(suffix) else f"{text}\n{suffix}"


def image_bytes(value: Any) -> bytes:
    if isinstance(value, dict):
        raw = value.get("bytes")
        if isinstance(raw, (bytes, bytearray)):
            return bytes(raw)
        path = value.get("path")
        if path and Path(str(path)).is_file():
            return Path(str(path)).read_bytes()
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    raise ValueError("V* row does not contain decodable image bytes")


def prepare(config_path: str, source_override: str | None, force: bool) -> dict[str, Any]:
    resolved_config_path, config = load_config(config_path)
    require_frozen_r3_config(resolved_config_path)
    benchmark = config["benchmarks"]["vstar"]
    configured_source = resolve_path(benchmark["source_parquet"])
    source = resolve_path(source_override) if source_override else configured_source
    if not source.is_file():
        raise FileNotFoundError(
            f"official V* parquet not found at {source}; provide --source-parquet"
        )
    configured_source.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != configured_source.resolve():
        if configured_source.exists() and not force:
            if sha256_file(source) != sha256_file(configured_source):
                raise FileExistsError(
                    f"refusing to replace {configured_source}; pass --force after review"
                )
        else:
            temporary = configured_source.with_suffix(configured_source.suffix + ".tmp")
            shutil.copy2(source, temporary)
            os.replace(temporary, configured_source)
        source = configured_source

    output_json = resolve_path(benchmark["converted_json"])
    image_dir = output_json.parent / "images"
    if output_json.exists() and not force:
        raise FileExistsError(f"refusing to overwrite {output_json}; pass --force")
    image_dir.mkdir(parents=True, exist_ok=True)

    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(str(source))
    required = {"image", "text", "label", "question_id", "category"}
    missing = required.difference(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"official V* parquet missing columns: {sorted(missing)}")

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    suffix = str(benchmark["prompt_suffix"])
    print(f"Reading official V* parquet: {source}", flush=True)
    for batch in parquet.iter_batches(
        batch_size=1,
        columns=["image", "text", "label", "question_id", "category"],
    ):
        row = batch.to_pylist()[0]
        source_id = str(row.get("question_id"))
        if not source_id or source_id == "None" or source_id in seen:
            raise ValueError(f"invalid or duplicate V* question_id: {source_id!r}")
        seen.add(source_id)
        raw = image_bytes(row.get("image"))
        image_path = image_dir / f"{safe_name(source_id)}.jpg"
        temporary = image_path.with_suffix(".jpg.tmp")
        temporary.write_bytes(raw)
        os.replace(temporary, image_path)
        answer = str(row.get("label") or "").strip().upper()
        if answer not in {"A", "B", "C", "D"}:
            raise ValueError(f"V* {source_id}: invalid answer {answer!r}")
        query = append_suffix(str(row.get("text") or ""), suffix)
        category = str(row.get("category") or "unknown")
        rows.append(
            {
                "schema_version": 1,
                "benchmark": "vstar",
                "sample_uid": f"vstar:source_id:{source_id}",
                "source_id": source_id,
                "source_repo_id": benchmark["dataset_repo_id"],
                "source_revision": benchmark["dataset_revision"],
                "source_split": benchmark["split"],
                "query": query,
                "response": answer,
                "images": [str(image_path.resolve())],
                "crop_images": [],
                "question_format": "multiple_choice",
                "category": category,
                "l2_category": "unavailable_official",
                "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
            }
        )
        if len(rows) == 1 or len(rows) % 10 == 0:
            print(f"Prepared V* rows: {len(rows)}/{benchmark['expected_sample_count']}", flush=True)

    expected = int(benchmark["expected_sample_count"])
    if len(rows) != expected:
        raise ValueError(f"official V* mirror expected {expected} rows, found {len(rows)}")
    temporary_json = output_json.with_suffix(".json.tmp")
    temporary_json.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_json, output_json)

    category_counts: dict[str, int] = {}
    for row in rows:
        category_counts[row["category"]] = category_counts.get(row["category"], 0) + 1
    evidence = {
        "schema_version": 1,
        "status": "pass",
        "dataset_repo_id": benchmark["dataset_repo_id"],
        "dataset_revision": benchmark["dataset_revision"],
        "source_parquet": str(source),
        "source_parquet_sha256": sha256_file(source),
        "converted_json": str(output_json),
        "converted_json_sha256": sha256_file(output_json),
        "record_count": len(rows),
        "unique_sample_uid_count": len({row["sample_uid"] for row in rows}),
        "category_counts": dict(sorted(category_counts.items())),
    }
    evidence_path = (
        resolve_path(config["paths"]["run_root"])
        / "preflight"
        / "vstar_official_mirror_preparation.json"
    )
    write_json(evidence_path, evidence)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/benchmark_eval_paper_basejudge_r3_single_gpu.yaml")
    parser.add_argument("--source-parquet")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    evidence = prepare(args.config, args.source_parquet, args.force)
    print(json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
