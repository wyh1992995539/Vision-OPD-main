#!/usr/bin/env python3
"""Create a deterministic, stratified 30-sample manual-QA worklist."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.validate_project_data import EXPECTED_SPLITS, load_manifests


DEFAULT_QUOTAS = {"train": 20, "eval": 5, "retention": 5}
BUCKETS = ("small", "medium", "large")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Select 30 frozen samples for manual visual-semantic QA."
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=repo_root / "artifacts" / "data",
    )
    parser.add_argument("--subset-root", type=Path, required=True)
    parser.add_argument(
        "--validation-report",
        type=Path,
        default=repo_root / "artifacts" / "data" / "data_validation.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "artifacts" / "data" / "manual_qa_30.jsonl",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing worklist. Never use this after review has started.",
    )
    return parser.parse_args()


def stable_key(seed: int, sample_id: str) -> str:
    return hashlib.sha256(f"{seed}|{sample_id}".encode("utf-8")).hexdigest()


def bucket_quota(total: int) -> dict[str, int]:
    base, remainder = divmod(total, 3)
    values = [base, base, base]
    if remainder == 1:
        values[1] += 1
    elif remainder == 2:
        values[0] += 1
        values[2] += 1
    return dict(zip(BUCKETS, values, strict=True))


def add_area_ratio(records: list[dict[str, object]], subset_root: Path) -> None:
    for record in records:
        full_path = subset_root / Path(str(record["full_image_path"]))
        if not full_path.is_file():
            raise FileNotFoundError(f"Missing Student image: {full_path}")
        with Image.open(full_path) as image:
            width, height = image.size
        bbox = record.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"Invalid bbox for {record.get('sample_id')}: {bbox!r}")
        x1, y1, x2, y2 = bbox
        record["_bbox_area_ratio"] = ((x2 - x1) * (y2 - y1)) / (width * height)


def select_records(
    records: list[dict[str, object]],
    *,
    seed: int,
    quotas: dict[str, int] = DEFAULT_QUOTAS,
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for split in ("train", "eval", "retention"):
        split_records = [record for record in records if record.get("split") == split]
        requested = quotas.get(split, 0)
        if len(split_records) < requested:
            raise ValueError(
                f"Split {split!r} has {len(split_records)} records; cannot select {requested}"
            )

        ranked = sorted(
            split_records,
            key=lambda record: (
                float(record["_bbox_area_ratio"]),
                stable_key(seed, str(record["sample_id"])),
            ),
        )
        buckets: dict[str, list[dict[str, object]]] = {name: [] for name in BUCKETS}
        for index, record in enumerate(ranked):
            bucket_index = min(2, (index * 3) // len(ranked))
            record["_area_bucket"] = BUCKETS[bucket_index]
            buckets[BUCKETS[bucket_index]].append(record)

        for bucket, count in bucket_quota(requested).items():
            candidates = sorted(
                buckets[bucket],
                key=lambda record: stable_key(seed, str(record["sample_id"])),
            )
            if len(candidates) < count:
                raise ValueError(
                    f"Split {split!r} bucket {bucket!r} has {len(candidates)} "
                    f"records; cannot select {count}"
                )
            selected.extend(candidates[:count])

    return selected


def make_review_record(
    record: dict[str, object], *, qa_index: int, seed: int
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "qa_index": qa_index,
        "selection_seed": seed,
        "selection_strategy": "split_quota_plus_per_split_bbox_area_tertiles",
        "area_bucket": record["_area_bucket"],
        "bbox_area_ratio": round(float(record["_bbox_area_ratio"]), 8),
        "sample_id": record.get("sample_id"),
        "source_id": record.get("source_id"),
        "group_id": record.get("group_id"),
        "split": record.get("split"),
        "full_image_path": record.get("full_image_path"),
        "crop_image_path": record.get("crop_image_path"),
        "bbox": record.get("bbox"),
        "problem": record.get("problem"),
        "question": record.get("question"),
        "answer": record.get("answer"),
        "question_type": record.get("question_type"),
        "full_image_has_box": None,
        "crop_matches_box": None,
        "question_matches_region": None,
        "answer_matches_image": None,
        "status": "pending",
        "note": "",
    }


def write_jsonl_atomic(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing manual QA worklist: {output}. "
            "Use --overwrite only before review starts."
        )

    validation = json.loads(args.validation_report.read_text(encoding="utf-8"))
    if validation.get("status") != "PASS" or validation.get("issue_count") != 0:
        raise ValueError("Automatic data validation must PASS with zero issues first")

    records, issues = load_manifests(args.manifest_dir.resolve(), EXPECTED_SPLITS)
    if issues:
        raise ValueError(f"Manifest validation produced {len(issues)} issue(s)")
    add_area_ratio(records, args.subset_root.resolve())
    selected = select_records(records, seed=args.seed)
    review_records = [
        make_review_record(record, qa_index=index, seed=args.seed)
        for index, record in enumerate(selected, start=1)
    ]
    write_jsonl_atomic(output, review_records)

    split_counts = Counter(str(record["split"]) for record in review_records)
    bucket_counts = Counter(
        f"{record['split']}:{record['area_bucket']}" for record in review_records
    )
    print(f"Manual QA worklist: {output}")
    print(f"Selected: {len(review_records)}")
    print(f"Split counts: {dict(split_counts)}")
    print(f"Bucket counts: {dict(bucket_counts)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
