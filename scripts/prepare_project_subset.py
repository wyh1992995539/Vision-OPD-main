"""Create the deterministic Day 2 Vision-OPD project split.

This script intentionally operates on metadata only. It does not download or
decode images and it does not create training Parquet files. Image extraction
and image-level QA belong to Day 3.

Example:
    python scripts/prepare_project_subset.py --config configs/project_1024.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import yaml


SCHEMA_VERSION = 1
ALGORITHM_VERSION = "original-image-one-sample-v1"
SOURCE_ID_KEYS = ("source_id", "sample_id", "question_id", "id")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build deterministic train/eval/retention metadata manifests."
    )
    parser.add_argument(
        "--config",
        default="configs/project_1024.yaml",
        help="Project YAML, relative to the repository root by default.",
    )
    parser.add_argument(
        "--input-jsonl",
        help="Override data.source_metadata.local_path from the project YAML.",
    )
    parser.add_argument(
        "--output-dir",
        help="Override all Day 2 output paths, useful for reproducibility tests.",
    )
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFC", str(value))
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in text.split("\n")).strip()


def sha256_text(*parts: Any) -> str:
    payload = "\x1f".join(normalize_text(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def one_relative_path(value: Any, field: str, issues: list[str]) -> str:
    if not isinstance(value, list) or len(value) != 1:
        issues.append(f"{field}_must_contain_exactly_one_path")
        return ""
    path = normalize_text(value[0]).replace("\\", "/")
    if not path:
        issues.append(f"empty_{field}_path")
        return ""
    posix_path = PurePosixPath(path)
    if posix_path.is_absolute() or ".." in posix_path.parts:
        issues.append(f"non_portable_{field}_path")
    return posix_path.as_posix()


def original_source_id(item: dict[str, Any]) -> tuple[str | None, str]:
    for key in SOURCE_ID_KEYS:
        value = normalize_text(item.get(key))
        if value:
            return value, f"top_level.{key}"
    extra_info = item.get("extra_info")
    if isinstance(extra_info, dict):
        for key in SOURCE_ID_KEYS:
            value = normalize_text(extra_info.get(key))
            if value:
                return value, f"extra_info.{key}"
    return None, "missing"


def classify_question(problem: str, answer: str) -> str:
    if re.search(r"(?m)^\s*[A-D][.)]\s+", problem) and re.fullmatch(
        r"[A-D]", answer.strip(), re.IGNORECASE
    ):
        return "multiple_choice"
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?(?:\s*[%a-zA-Z]+)?", answer.strip()):
        return "numeric_or_unit"
    return "short_answer"


def parse_bbox(value: Any, issues: list[str]) -> list[int | float]:
    if not isinstance(value, list) or len(value) != 4:
        issues.append("bbox_must_have_four_values")
        return []
    if not all(isinstance(number, (int, float)) and math.isfinite(number) for number in value):
        issues.append("bbox_contains_non_finite_number")
        return []
    x1, y1, x2, y2 = value
    if x2 <= x1 or y2 <= y1:
        issues.append("bbox_has_non_positive_area")
    return value


def make_record(
    item: dict[str, Any],
    source_row: int,
    source_name: str,
    source_revision: str,
) -> dict[str, Any]:
    issues: list[str] = []
    full_image_path = one_relative_path(item.get("images"), "full_image", issues)
    crop_image_path = one_relative_path(item.get("teacher_images"), "crop_image", issues)
    original_image_path = one_relative_path(
        item.get("original_images"), "original_image", issues
    )

    problem = normalize_text(item.get("problem"))
    answer = normalize_text(item.get("answer"))
    extra_info = item.get("extra_info") if isinstance(item.get("extra_info"), dict) else {}
    question = normalize_text(extra_info.get("question")) or problem

    if not problem:
        issues.append("empty_problem")
    if not answer:
        issues.append("empty_answer")
    if problem.count("<image>") != 1:
        issues.append("problem_must_contain_one_image_placeholder")

    extra_answer = normalize_text(extra_info.get("answer"))
    if extra_answer and extra_answer != answer:
        issues.append("answer_mismatch_with_extra_info")

    bbox = parse_bbox(item.get("bbox"), issues)
    raw_source_id, raw_source_id_field = original_source_id(item)
    fallback_source_id = f"{source_revision}:row:{source_row:06d}"
    source_id = raw_source_id or fallback_source_id
    source_id_kind = raw_source_id_field if raw_source_id else "frozen_row_fallback"

    group_id = (
        f"img_{sha256_text('original_image', original_image_path)[:24]}"
        if original_image_path
        else ""
    )
    sample_id = "vopd_" + sha256_text(
        source_name,
        source_revision,
        original_image_path,
        full_image_path,
        crop_image_path,
        problem,
        answer,
    )[:24]

    return {
        "schema_version": SCHEMA_VERSION,
        "sample_id": sample_id,
        "source_id": source_id,
        "source_id_kind": source_id_kind,
        "source_row": source_row,
        "group_id": group_id,
        "original_image_path": original_image_path,
        "full_image_path": full_image_path,
        "crop_image_path": crop_image_path,
        "bbox": bbox,
        "problem": problem,
        "question": question,
        "answer": answer,
        "question_type": classify_question(problem, answer),
        "valid": not issues,
        "issues": sorted(set(issues)),
        "selection_eligible": False,
        "selection_status": "not_evaluated",
        "split": None,
    }


def read_records(
    input_path: Path, source_name: str, source_revision: str
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as stream:
        for source_row, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"Blank JSONL line at source row {source_row}")
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at source row {source_row}: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"Source row {source_row} is not a JSON object")
            records.append(make_record(item, source_row, source_name, source_revision))
    return records


def mark_exact_duplicates(records: list[dict[str, Any]]) -> None:
    sample_id_counts = Counter(record["sample_id"] for record in records)
    for record in records:
        if sample_id_counts[record["sample_id"]] > 1:
            record["issues"] = sorted(set(record["issues"] + ["duplicate_sample_id"]))
            record["valid"] = False


def select_one_record_per_group(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record["valid"]:
            groups[record["group_id"]].append(record)

    representatives: list[dict[str, Any]] = []
    for group_id in sorted(groups):
        group_records = sorted(
            groups[group_id], key=lambda record: (record["sample_id"], record["source_row"])
        )
        representative = group_records[0]
        representative["selection_eligible"] = True
        representative["selection_status"] = "group_representative"
        representatives.append(representative)
        for duplicate in group_records[1:]:
            duplicate["selection_status"] = "excluded_duplicate_original_image_group"

    for record in records:
        if not record["valid"]:
            record["selection_status"] = "excluded_invalid_metadata"
    return representatives


def assign_splits(
    representatives: list[dict[str, Any]], seed: int, split_sizes: dict[str, int]
) -> dict[str, list[dict[str, Any]]]:
    total_required = sum(split_sizes.values())
    if len(representatives) < total_required:
        raise ValueError(
            "Not enough valid unique original-image groups: "
            f"need {total_required}, found {len(representatives)}"
        )

    ranked = sorted(
        representatives,
        key=lambda record: (
            sha256_text(seed, record["group_id"]),
            record["group_id"],
            record["sample_id"],
        ),
    )

    assigned: dict[str, list[dict[str, Any]]] = {}
    cursor = 0
    for split_name in ("train", "eval", "retention"):
        size = split_sizes[split_name]
        split_records = ranked[cursor : cursor + size]
        for record in split_records:
            record["split"] = split_name
            record["selection_status"] = "selected"
        assigned[split_name] = split_records
        cursor += size

    for record in ranked[cursor:]:
        record["selection_status"] = "eligible_not_selected"
    return assigned


def selected_record(record: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "schema_version",
        "sample_id",
        "source_id",
        "source_id_kind",
        "source_row",
        "split",
        "group_id",
        "original_image_path",
        "full_image_path",
        "crop_image_path",
        "bbox",
        "problem",
        "question",
        "answer",
        "question_type",
    )
    return {field: record[field] for field in fields}


def ensure_no_group_overlap(assigned: dict[str, list[dict[str, Any]]]) -> None:
    group_sets = {
        split: {record["group_id"] for record in records}
        for split, records in assigned.items()
    }
    split_names = tuple(group_sets)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            overlap = group_sets[left] & group_sets[right]
            if overlap:
                raise AssertionError(
                    f"Group leakage between {left} and {right}: {sorted(overlap)[:5]}"
                )


def json_line(record: dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    os.replace(temporary_path, path)


def jsonl_content(records: Iterable[dict[str, Any]]) -> str:
    lines = [json_line(record) for record in records]
    return "\n".join(lines) + ("\n" if lines else "")


def resolve_config_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def output_paths(
    config: dict[str, Any], project_root: Path, output_dir: str | None
) -> dict[str, Path]:
    if output_dir:
        root = Path(output_dir).resolve()
        return {
            "candidate": root / "candidate_manifest.jsonl",
            "split": root / "split_manifest.json",
            "statistics": root / "candidate_stats.json",
            "train": root / "train_1024.jsonl",
            "eval": root / "eval_128.jsonl",
            "retention": root / "retention_64.jsonl",
        }

    manifests = config["data"]["manifests"]
    return {
        "candidate": resolve_config_path(project_root, manifests["candidate"]),
        "split": resolve_config_path(project_root, manifests["split"]),
        "statistics": resolve_config_path(
            project_root, manifests["candidate_statistics"]
        ),
        "train": resolve_config_path(project_root, manifests["train_jsonl"]),
        "eval": resolve_config_path(project_root, manifests["eval_jsonl"]),
        "retention": resolve_config_path(project_root, manifests["retention_jsonl"]),
    }


def build_statistics(
    records: list[dict[str, Any]],
    representatives: list[dict[str, Any]],
    assigned: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    issue_counts = Counter(issue for record in records for issue in record["issues"])
    group_counts = Counter(record["group_id"] for record in records if record["group_id"])
    return {
        "schema_version": SCHEMA_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "total_records": len(records),
        "valid_records": sum(record["valid"] for record in records),
        "invalid_records": sum(not record["valid"] for record in records),
        "unique_original_image_groups": len(group_counts),
        "groups_with_multiple_records": sum(count > 1 for count in group_counts.values()),
        "eligible_group_representatives": len(representatives),
        "selected_counts": {split: len(items) for split, items in assigned.items()},
        "issue_counts": dict(sorted(issue_counts.items())),
        "question_type_counts_selected": dict(
            sorted(
                Counter(
                    record["question_type"]
                    for split_records in assigned.values()
                    for record in split_records
                ).items()
            )
        ),
        "cross_split_group_overlap_count": 0,
    }


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    project_root = config_path.parent.parent
    data_config = config["data"]
    source_metadata = data_config["source_metadata"]
    input_path = Path(args.input_jsonl or source_metadata["local_path"]).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input metadata not found: {input_path}")

    actual_sha256 = sha256_file(input_path)
    expected_sha256 = normalize_text(source_metadata["sha256"]).lower()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"Input SHA256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    actual_byte_size = input_path.stat().st_size
    expected_byte_size = int(source_metadata["byte_size"])
    if actual_byte_size != expected_byte_size:
        raise ValueError(
            "Input byte-size mismatch: "
            f"expected {expected_byte_size}, got {actual_byte_size}"
        )

    records = read_records(
        input_path,
        source_name=data_config["source"],
        source_revision=data_config["source_revision"],
    )
    expected_count = int(source_metadata["record_count"])
    if len(records) != expected_count:
        raise ValueError(
            f"Input row count mismatch: expected {expected_count}, got {len(records)}"
        )

    mark_exact_duplicates(records)
    representatives = select_one_record_per_group(records)
    split_sizes = {
        name: int(data_config["splits"][name]["size"])
        for name in ("train", "eval", "retention")
    }
    seed = int(config["reproducibility"]["data_split_seed"])
    assigned = assign_splits(representatives, seed, split_sizes)
    ensure_no_group_overlap(assigned)

    statistics = build_statistics(records, representatives, assigned)
    split_manifest = {
        "schema_version": SCHEMA_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "source": data_config["source"],
        "source_revision": data_config["source_revision"],
        "source_metadata_sha256": actual_sha256,
        "source_record_count": len(records),
        "seed": seed,
        "grouping_rule": "one deterministic representative per original_images[0]",
        "splits": {
            split: [selected_record(record) for record in split_records]
            for split, split_records in assigned.items()
        },
    }

    paths = output_paths(config, project_root, args.output_dir)
    atomic_write_text(paths["candidate"], jsonl_content(records))
    atomic_write_text(
        paths["split"],
        json.dumps(split_manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    )
    atomic_write_text(
        paths["statistics"],
        json.dumps(statistics, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    )
    for split_name in ("train", "eval", "retention"):
        atomic_write_text(
            paths[split_name],
            jsonl_content(selected_record(record) for record in assigned[split_name]),
        )

    print("Day 2 deterministic subset preparation: PASS")
    print(f"source_revision={data_config['source_revision']}")
    print(f"source_sha256={actual_sha256}")
    print(f"source_records={len(records)}")
    print(f"valid_unique_groups={len(representatives)}")
    for split_name in ("train", "eval", "retention"):
        print(f"{split_name}={len(assigned[split_name])}")
    for name, path in paths.items():
        print(f"{name}_path={path}")
        print(f"{name}_sha256={sha256_file(path)}")


if __name__ == "__main__":
    main()
