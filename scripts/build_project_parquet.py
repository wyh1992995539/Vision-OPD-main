"""Build the frozen Vision-OPD project Parquet files on the Linux server.

This script converts the already selected 1024/128/64 JSONL manifests.  It
does not download data, extract archives, or resample the dataset.

Example (run on the server from the repository root):

    python scripts/build_project_parquet.py \
        --data-root /root/autodl-tmp/data/vision_opd_1024
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


DATA_SOURCE = "zwz_rl_vqa_bbox_teacher"
ABILITY = "visual_question_answering"
REMOVE_HINT = (
    "Only focus on the objects inside the red bounding box in the image "
    "to answer this question."
)
DEFAULT_SPLITS = {
    "train": ("train_1024.jsonl", "train_1024.parquet", 1024),
    "eval": ("eval_128.jsonl", "eval_128.parquet", 128),
    "retention": ("retention_64.jsonl", "retention_64.parquet", 64),
}
REQUIRED_FIELDS = {
    "sample_id",
    "source_id",
    "split",
    "group_id",
    "full_image_path",
    "crop_image_path",
    "bbox",
    "problem",
    "question",
    "answer",
    "question_type",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build frozen train/eval/retention Parquet files from project manifests."
    )
    parser.add_argument(
        "--data-root",
        default="/root/autodl-tmp/data/vision_opd_1024",
        help="Root containing images/, teacher_images/, and manifests/.",
    )
    parser.add_argument(
        "--manifest-dir",
        help="Manifest directory; defaults to <data-root>/manifests.",
    )
    parser.add_argument(
        "--output-dir",
        help="Parquet output directory; defaults to <data-root>.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing Parquet/report files atomically.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_question(problem: str) -> str:
    text = problem.replace("<image>", "").strip()
    text = text.replace(f"\n\n{REMOVE_HINT}", "")
    text = text.replace(REMOVE_HINT, "")
    return text.strip()


def validate_relative_path(value: Any, expected_root: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    value = value.strip()
    if "\\" in value or re.match(r"^[A-Za-z]:", value):
        raise ValueError(f"{field} contains a Windows path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field} must be a safe relative path: {value!r}")
    if not path.parts or path.parts[0] != expected_root:
        raise ValueError(f"{field} must start with {expected_root}/: {value!r}")
    return path.as_posix()


def resolve_image(data_root: Path, relative_path: str, field: str) -> Path:
    candidate = (data_root / Path(*PurePosixPath(relative_path).parts)).resolve()
    if not candidate.is_relative_to(data_root):
        raise ValueError(f"{field} escapes data root: {relative_path!r}")
    if not candidate.is_file():
        raise FileNotFoundError(f"{field} not found: {candidate}")
    return candidate


def validate_bbox(value: Any) -> list[int | float]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("bbox must contain exactly four numbers")
    if not all(isinstance(number, (int, float)) and not isinstance(number, bool) for number in value):
        raise ValueError("bbox must contain exactly four numbers")
    x1, y1, x2, y2 = value
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"bbox has non-positive area: {value!r}")
    return value


def load_manifest(path: Path, split: str, expected_count: int) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank line")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: row must be a JSON object")
            missing = sorted(REQUIRED_FIELDS - record.keys())
            if missing:
                raise ValueError(f"{path}:{line_number}: missing fields: {missing}")
            if record["split"] != split:
                raise ValueError(
                    f"{path}:{line_number}: expected split {split!r}, got {record['split']!r}"
                )
            records.append(record)

    if len(records) != expected_count:
        raise ValueError(f"{path}: expected {expected_count} rows, found {len(records)}")
    return records


def build_training_record(item: dict[str, Any], data_root: Path) -> dict[str, Any]:
    sample_id = item["sample_id"]
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise ValueError("sample_id must be a non-empty string")
    source_id = item["source_id"]
    if not isinstance(source_id, str) or not source_id.strip():
        raise ValueError(f"{sample_id}: source_id must be a non-empty string")
    group_id = item["group_id"]
    if not isinstance(group_id, str) or not group_id.strip():
        raise ValueError(f"{sample_id}: group_id must be a non-empty string")

    problem = item["problem"]
    answer = item["answer"]
    if not isinstance(problem, str) or problem.count("<image>") != 1:
        raise ValueError(f"{sample_id}: problem must contain exactly one <image>")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError(f"{sample_id}: answer must be a non-empty string")

    full_relative = validate_relative_path(item["full_image_path"], "images", "full_image_path")
    crop_relative = validate_relative_path(
        item["crop_image_path"], "teacher_images", "crop_image_path"
    )
    full_path = resolve_image(data_root, full_relative, "full_image_path")
    crop_path = resolve_image(data_root, crop_relative, "crop_image_path")
    bbox = validate_bbox(item["bbox"])

    question = item.get("question")
    if not isinstance(question, str) or not question.strip():
        question = clean_question(problem)

    provenance = {
        "sample_id": sample_id,
        "source_id": source_id,
        "group_id": group_id,
        "split": item["split"],
        "question_type": str(item["question_type"]),
        "bbox": bbox,
        "full_image_path": full_relative,
        "crop_image_path": crop_relative,
        "original_image_path": str(item.get("original_image_path", "")),
        "source_row": int(item.get("source_row", -1)),
    }

    return {
        "data_source": DATA_SOURCE,
        "prompt": [{"role": "user", "content": problem}],
        "images": [{"path": full_path.as_posix()}],
        "bbox_images": [{"path": crop_path.as_posix()}],
        "ability": ABILITY,
        "reward_model": {"style": "none", "ground_truth": answer},
        "extra_info": {
            "answer": answer,
            "question": question,
            "provenance": provenance,
        },
    }


def validate_no_overlap(records_by_split: dict[str, list[dict[str, Any]]]) -> None:
    seen_sample_ids: dict[str, str] = {}
    seen_group_ids: dict[str, str] = {}
    seen_full_paths: dict[str, str] = {}
    seen_crop_paths: dict[str, str] = {}

    for split, records in records_by_split.items():
        for record in records:
            identifiers = (
                ("sample_id", str(record["sample_id"]), seen_sample_ids),
                ("group_id", str(record["group_id"]), seen_group_ids),
                ("full_image_path", str(record["full_image_path"]), seen_full_paths),
                ("crop_image_path", str(record["crop_image_path"]), seen_crop_paths),
            )
            for field, value, seen in identifiers:
                previous = seen.get(value)
                if previous is not None:
                    raise ValueError(
                        f"Duplicate {field} across records: {value!r} in {previous!r} and {split!r}"
                    )
                seen[value] = split


def atomic_write_parquet(records: list[dict[str, Any]], output_path: Path) -> None:
    table = pa.Table.from_pylist(records)
    with tempfile.NamedTemporaryFile(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary_path = Path(stream.name)
    try:
        pq.write_table(table, temporary_path, compression="zstd")
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def atomic_write_text(path: Path, content: str) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary_path = Path(stream.name)
        stream.write(content)
    try:
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def build_project_parquets(
    manifest_dir: Path,
    data_root: Path,
    output_dir: Path,
    *,
    expected_splits: dict[str, tuple[str, str, int]] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    manifest_dir = manifest_dir.resolve()
    data_root = data_root.resolve()
    output_dir = output_dir.resolve()
    splits = expected_splits or DEFAULT_SPLITS

    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root not found: {data_root}")
    if not manifest_dir.is_dir():
        raise FileNotFoundError(f"Manifest directory not found: {manifest_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    records_by_split = {
        split: load_manifest(manifest_dir / manifest_name, split, expected_count)
        for split, (manifest_name, _output_name, expected_count) in splits.items()
    }
    validate_no_overlap(records_by_split)

    output_paths = {
        split: output_dir / output_name
        for split, (_manifest_name, output_name, _expected_count) in splits.items()
    }
    report_path = manifest_dir / "parquet_build_report.json"
    hash_path = manifest_dir / "parquet_sha256.txt"
    targets = [*output_paths.values(), report_path, hash_path]
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        formatted = "\n".join(f"  - {path}" for path in existing)
        raise FileExistsError(f"Output already exists; use --overwrite to replace:\n{formatted}")

    outputs: dict[str, Any] = {}
    hashes: list[str] = []
    for split, source_records in records_by_split.items():
        training_records = [build_training_record(item, data_root) for item in source_records]
        output_path = output_paths[split]
        atomic_write_parquet(training_records, output_path)

        with output_path.open("rb") as parquet_stream:
            parquet_file = pq.ParquetFile(parquet_stream)
            readback_rows = parquet_file.metadata.num_rows
            actual_columns = set(parquet_file.schema_arrow.names)

        expected_count = splits[split][2]
        if readback_rows != expected_count:
            raise AssertionError(
                f"Read-back row mismatch for {output_path}: "
                f"expected {expected_count}, got {readback_rows}"
            )
        required_columns = {
            "data_source",
            "prompt",
            "images",
            "bbox_images",
            "ability",
            "reward_model",
            "extra_info",
        }
        if not required_columns.issubset(actual_columns):
            raise AssertionError(
                f"Read-back schema missing columns for {output_path}: "
                f"{sorted(required_columns - actual_columns)}"
            )

        digest = sha256_file(output_path)
        outputs[split] = {
            "path": output_path.as_posix(),
            "rows": expected_count,
            "byte_size": output_path.stat().st_size,
            "sha256": digest,
        }
        hashes.append(f"{digest}  {output_path.name}")

    report = {
        "schema_version": 1,
        "status": "PASS",
        "data_root": data_root.as_posix(),
        "manifest_dir": manifest_dir.as_posix(),
        "output_dir": output_dir.as_posix(),
        "total_rows": sum(item["rows"] for item in outputs.values()),
        "outputs": outputs,
        "checks": {
            "expected_split_counts": True,
            "required_manifest_fields": True,
            "portable_relative_manifest_paths": True,
            "all_image_paths_readable": True,
            "no_sample_group_or_image_overlap": True,
            "parquet_readback_schema_and_rows": True,
        },
    }
    atomic_write_text(hash_path, "\n".join(hashes) + "\n")
    atomic_write_text(report_path, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def main() -> None:
    args = parse_args()
    if os.name == "nt":
        raise SystemExit(
            "This command must run on the Linux server so Parquet image paths use /root/... paths."
        )

    data_root = Path(args.data_root).resolve()
    manifest_dir = Path(args.manifest_dir).resolve() if args.manifest_dir else data_root / "manifests"
    output_dir = Path(args.output_dir).resolve() if args.output_dir else data_root
    report = build_project_parquets(
        manifest_dir,
        data_root,
        output_dir,
        overwrite=args.overwrite,
    )

    print("PASS: project Parquet build completed")
    for split, item in report["outputs"].items():
        print(f"  {split}: {item['rows']} rows -> {item['path']}")
    print(f"  report: {manifest_dir / 'parquet_build_report.json'}")
    print(f"  hashes: {manifest_dir / 'parquet_sha256.txt'}")


if __name__ == "__main__":
    main()
