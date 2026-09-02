#!/usr/bin/env python3
"""Validate the frozen Vision-OPD image subset and its manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Iterable

from PIL import Image, UnidentifiedImageError
import yaml


EXPECTED_SPLITS = {
    "train_1024.jsonl": ("train", 1024),
    "eval_128.jsonl": ("eval", 128),
    "retention_64.jsonl": ("retention", 64),
}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Validate frozen manifests and all extracted Student/Teacher images."
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Project YAML defining active split manifests and expected counts.",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=repo_root / "artifacts" / "data",
        help="Directory containing train_1024/eval_128/retention_64 JSONL files.",
    )
    parser.add_argument(
        "--subset-root",
        type=Path,
        required=True,
        help="Subset root containing images/ and teacher_images/.",
    )
    parser.add_argument(
        "--validation-report",
        type=Path,
        default=repo_root / "artifacts" / "data" / "data_validation.json",
    )
    parser.add_argument(
        "--statistics-report",
        type=Path,
        default=repo_root / "artifacts" / "data" / "data_stats.json",
    )
    parser.add_argument(
        "--sha256-report",
        type=Path,
        default=repo_root / "artifacts" / "data" / "data_sha256.txt",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel image readers. Use 1 for a slow mechanical disk.",
    )
    return parser.parse_args()


def configured_splits(config_path: Path) -> dict[str, tuple[str, int]]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data = config["data"]
    result: dict[str, tuple[str, int]] = {}
    for name, split in data["splits"].items():
        if bool(split.get("historical_only", False)):
            continue
        filename = Path(data["manifests"][f"{name}_jsonl"]).name
        result[filename] = (name, int(split["size"]))
    if not result:
        raise ValueError("Project config contains no active data splits")
    return result


def add_issue(
    issues: list[dict[str, object]],
    kind: str,
    detail: str,
    *,
    sample_id: str | None = None,
    path: str | None = None,
) -> None:
    issue: dict[str, object] = {"kind": kind, "detail": detail}
    if sample_id is not None:
        issue["sample_id"] = sample_id
    if path is not None:
        issue["path"] = path
    issues.append(issue)


def normalize_relative_image_path(
    value: object,
    expected_directory: str,
    *,
    source: str,
) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{source}: image path is not a string")
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{source}: unsafe image path {value!r}")
    if len(path.parts) != 2 or path.parts[0] != expected_directory or not path.name:
        raise ValueError(
            f"{source}: expected {expected_directory}/<file>, got {value!r}"
        )
    return path.as_posix()


def load_manifests(
    manifest_dir: Path,
    expected_splits: dict[str, tuple[str, int]] = EXPECTED_SPLITS,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    records: list[dict[str, object]] = []
    issues: list[dict[str, object]] = []
    sample_ids: set[str] = set()
    full_paths: set[str] = set()
    crop_paths: set[str] = set()
    group_splits: dict[str, set[str]] = defaultdict(set)

    for filename, (expected_split, expected_count) in expected_splits.items():
        manifest_path = manifest_dir / filename
        if not manifest_path.is_file():
            add_issue(issues, "missing_manifest", "Manifest file does not exist", path=str(manifest_path))
            continue

        split_count = 0
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                source = f"{manifest_path}:{line_number}"
                split_count += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    add_issue(issues, "invalid_json", str(error), path=source)
                    continue

                sample_id = record.get("sample_id")
                if not isinstance(sample_id, str) or not sample_id:
                    add_issue(issues, "invalid_sample_id", "sample_id is empty", path=source)
                    continue
                if sample_id in sample_ids:
                    add_issue(
                        issues,
                        "duplicate_sample_id",
                        "sample_id appears more than once",
                        sample_id=sample_id,
                        path=source,
                    )
                sample_ids.add(sample_id)

                if record.get("split") != expected_split:
                    add_issue(
                        issues,
                        "split_mismatch",
                        f"Expected split {expected_split!r}, got {record.get('split')!r}",
                        sample_id=sample_id,
                        path=source,
                    )

                try:
                    full_path = normalize_relative_image_path(
                        record.get("full_image_path"), "images", source=source
                    )
                    crop_path = normalize_relative_image_path(
                        record.get("crop_image_path"), "teacher_images", source=source
                    )
                except ValueError as error:
                    add_issue(
                        issues,
                        "invalid_image_path",
                        str(error),
                        sample_id=sample_id,
                        path=source,
                    )
                    continue

                if full_path in full_paths:
                    add_issue(
                        issues,
                        "duplicate_full_image_reference",
                        "Student image is referenced by multiple samples",
                        sample_id=sample_id,
                        path=full_path,
                    )
                if crop_path in crop_paths:
                    add_issue(
                        issues,
                        "duplicate_crop_image_reference",
                        "Teacher image is referenced by multiple samples",
                        sample_id=sample_id,
                        path=crop_path,
                    )
                full_paths.add(full_path)
                crop_paths.add(crop_path)

                full_stem = Path(full_path).stem
                crop_stem = Path(crop_path).stem
                if not crop_stem.endswith(f"_{full_stem}"):
                    add_issue(
                        issues,
                        "full_crop_id_mismatch",
                        "Teacher filename does not end with the Student image identifier",
                        sample_id=sample_id,
                    )

                problem = record.get("problem")
                answer = record.get("answer")
                if not isinstance(problem, str) or not problem.strip():
                    add_issue(
                        issues,
                        "empty_problem",
                        "problem is empty",
                        sample_id=sample_id,
                    )
                elif problem.count("<image>") != 1:
                    add_issue(
                        issues,
                        "image_placeholder_count",
                        f"Expected one <image> placeholder, found {problem.count('<image>')}",
                        sample_id=sample_id,
                    )
                if not isinstance(answer, str) or not answer.strip():
                    add_issue(
                        issues,
                        "empty_answer",
                        "answer is empty",
                        sample_id=sample_id,
                    )

                group_id = record.get("group_id")
                if isinstance(group_id, str) and group_id:
                    group_splits[group_id].add(expected_split)

                record["full_image_path"] = full_path
                record["crop_image_path"] = crop_path
                records.append(record)

        if split_count != expected_count:
            add_issue(
                issues,
                "split_count_mismatch",
                f"Expected {expected_count} records, found {split_count}",
                path=str(manifest_path),
            )

    for group_id, splits in sorted(group_splits.items()):
        if len(splits) > 1:
            add_issue(
                issues,
                "cross_split_group_leakage",
                f"group_id appears in splits {sorted(splits)}",
                path=group_id,
            )

    return records, issues


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_image(task: tuple[str, Path]) -> dict[str, object]:
    relative_path, absolute_path = task
    result: dict[str, object] = {
        "relative_path": relative_path,
        "absolute_path": str(absolute_path),
        "exists": absolute_path.is_file(),
    }
    if not absolute_path.is_file():
        result["error"] = "file does not exist"
        return result
    try:
        result["byte_size"] = absolute_path.stat().st_size
        if result["byte_size"] <= 0:
            result["error"] = "file is empty"
            return result
        result["sha256"] = sha256_file(absolute_path)
        with Image.open(absolute_path) as image:
            result["format"] = image.format
            result["mode"] = image.mode
            result["width"], result["height"] = image.size
            image.load()
        if result["width"] <= 0 or result["height"] <= 0:
            result["error"] = "image dimensions are not positive"
    except (OSError, ValueError, UnidentifiedImageError) as error:
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def describe(values: Iterable[float | int]) -> dict[str, float | int | None]:
    data = sorted(values)
    if not data:
        return {"min": None, "median": None, "max": None}
    return {
        "min": data[0],
        "median": statistics.median(data),
        "max": data[-1],
    }


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        temporary.write_text(value, encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_dataset(
    manifest_dir: Path,
    subset_root: Path,
    *,
    workers: int,
    expected_splits: dict[str, tuple[str, int]] = EXPECTED_SPLITS,
) -> tuple[dict[str, object], dict[str, object], list[str]]:
    if workers < 1:
        raise ValueError("workers must be at least 1")

    records, issues = load_manifests(manifest_dir, expected_splits)
    expected_full = {str(record["full_image_path"]) for record in records}
    expected_crop = {str(record["crop_image_path"]) for record in records}
    expected_paths = sorted(expected_full | expected_crop)

    for directory_name, expected in (
        ("images", expected_full),
        ("teacher_images", expected_crop),
    ):
        directory = subset_root / directory_name
        if not directory.is_dir():
            add_issue(issues, "missing_image_directory", "Directory does not exist", path=str(directory))
            continue
        actual = {
            f"{directory_name}/{path.name}"
            for path in directory.iterdir()
            if path.is_file() and not path.name.startswith(".")
        }
        for path in sorted(expected - actual):
            add_issue(issues, "missing_image_file", "Expected image is absent", path=path)
        for path in sorted(actual - expected):
            add_issue(issues, "unexpected_image_file", "File is not referenced by a manifest", path=path)

    tasks = [(relative, subset_root / Path(relative)) for relative in expected_paths]
    image_results: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for index, result in enumerate(executor.map(inspect_image, tasks), start=1):
            image_results.append(result)
            if index % 200 == 0 or index == len(tasks):
                print(f"Validated images: {index}/{len(tasks)}", flush=True)

    image_info = {str(result["relative_path"]): result for result in image_results}
    for result in image_results:
        if "error" in result:
            add_issue(
                issues,
                "image_decode_failure",
                str(result["error"]),
                path=str(result["relative_path"]),
            )
        elif result.get("format") != "PNG":
            add_issue(
                issues,
                "unexpected_image_format",
                f"Expected PNG, found {result.get('format')!r}",
                path=str(result["relative_path"]),
            )

    bbox_area_ratios: list[float] = []
    for record in records:
        sample_id = str(record["sample_id"])
        full_path = str(record["full_image_path"])
        info = image_info.get(full_path, {})
        if "error" in info or "width" not in info or "height" not in info:
            continue
        bbox = record.get("bbox")
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or any(not isinstance(value, int) or isinstance(value, bool) for value in bbox)
        ):
            add_issue(
                issues,
                "invalid_bbox",
                f"Expected four integer coordinates, got {bbox!r}",
                sample_id=sample_id,
            )
            continue
        x1, y1, x2, y2 = bbox
        width = int(info["width"])
        height = int(info["height"])
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            add_issue(
                issues,
                "bbox_out_of_bounds",
                f"bbox={bbox!r}, image_size={[width, height]!r}",
                sample_id=sample_id,
                path=full_path,
            )
            continue
        bbox_area_ratios.append(((x2 - x1) * (y2 - y1)) / (width * height))

    issue_counts = Counter(str(issue["kind"]) for issue in issues)
    validation = {
        "schema_version": 1,
        "status": "PASS" if not issues else "FAIL",
        "semantic_alignment_checked": False,
        "semantic_alignment_note": "Full-image/crop/question/answer meaning requires manual QA.",
        "manifest_dir": str(manifest_dir.resolve()),
        "subset_root": str(subset_root.resolve()),
        "record_count": len(records),
        "unique_sample_ids": len({record["sample_id"] for record in records}),
        "expected_student_images": len(expected_full),
        "expected_teacher_images": len(expected_crop),
        "images_checked": len(image_results),
        "issue_count": len(issues),
        "issue_counts": dict(sorted(issue_counts.items())),
        "issues": issues,
    }

    valid_images = [result for result in image_results if "error" not in result]
    student_images = [
        result for result in valid_images if str(result["relative_path"]).startswith("images/")
    ]
    teacher_images = [
        result
        for result in valid_images
        if str(result["relative_path"]).startswith("teacher_images/")
    ]
    statistics_report = {
        "schema_version": 1,
        "record_count": len(records),
        "split_counts": dict(sorted(Counter(str(r.get("split")) for r in records).items())),
        "answer_counts": dict(sorted(Counter(str(r.get("answer")) for r in records).items())),
        "problem_character_length": describe(
            len(str(record.get("problem", ""))) for record in records
        ),
        "bbox_area_ratio": describe(bbox_area_ratios),
        "student_images": {
            "count": len(student_images),
            "total_bytes": sum(int(result["byte_size"]) for result in student_images),
            "width": describe(int(result["width"]) for result in student_images),
            "height": describe(int(result["height"]) for result in student_images),
            "formats": dict(sorted(Counter(str(r["format"]) for r in student_images).items())),
            "modes": dict(sorted(Counter(str(r["mode"]) for r in student_images).items())),
        },
        "teacher_images": {
            "count": len(teacher_images),
            "total_bytes": sum(int(result["byte_size"]) for result in teacher_images),
            "width": describe(int(result["width"]) for result in teacher_images),
            "height": describe(int(result["height"]) for result in teacher_images),
            "formats": dict(sorted(Counter(str(r["format"]) for r in teacher_images).items())),
            "modes": dict(sorted(Counter(str(r["mode"]) for r in teacher_images).items())),
        },
    }

    sha_lines = [
        f"{result['sha256']}  {result['relative_path']}"
        for result in image_results
        if "sha256" in result
    ]
    return validation, statistics_report, sha_lines


def main() -> int:
    args = parse_args()
    expected_splits = configured_splits(args.config.resolve()) if args.config else EXPECTED_SPLITS
    validation, statistics_report, sha_lines = validate_dataset(
        args.manifest_dir.resolve(),
        args.subset_root.resolve(),
        workers=args.workers,
        expected_splits=expected_splits,
    )
    write_json_atomic(args.validation_report.resolve(), validation)
    write_json_atomic(args.statistics_report.resolve(), statistics_report)
    write_text_atomic(args.sha256_report.resolve(), "\n".join(sha_lines) + "\n")

    print(f"Validation report: {args.validation_report.resolve()}")
    print(f"Statistics report: {args.statistics_report.resolve()}")
    print(f"SHA256 report: {args.sha256_report.resolve()}")
    print(
        f"Status: {validation['status']} | records={validation['record_count']} "
        f"images={validation['images_checked']} issues={validation['issue_count']}"
    )
    return 0 if validation["status"] == "PASS" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
