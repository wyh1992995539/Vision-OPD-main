#!/usr/bin/env python3
"""Selectively extract the frozen Vision-OPD project subset.

The official Student image archive is one gzip-compressed tar stream split into
images.tar.gz00 ... images.tar.gz05.  This script reads those parts as a single
stream and writes only images referenced by the frozen project manifests.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sys
import tarfile
import time
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Iterable


EXPECTED_SPLIT_SIZES = {
    "train_1024.jsonl": 1024,
    "eval_128.jsonl": 128,
    "retention_64.jsonl": 64,
}


class ConcatenatedReader(io.RawIOBase):
    """Expose several byte-split files as one readable binary stream."""

    def __init__(self, parts: Iterable[Path]) -> None:
        super().__init__()
        self._parts = tuple(parts)
        self._index = 0
        self._handle: BinaryIO | None = None

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: bytearray | memoryview) -> int:
        while self._index < len(self._parts):
            if self._handle is None:
                self._handle = self._parts[self._index].open("rb")

            count = self._handle.readinto(buffer)
            if count:
                return count

            self._handle.close()
            self._handle = None
            self._index += 1
        return 0

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        super().close()


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Extract only images referenced by the frozen 1024/128/64 manifests."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
        help="Downloaded dataset root containing images/ and teacher_images/ archives.",
    )
    parser.add_argument(
        "--subset-root",
        type=Path,
        required=True,
        help="Destination root containing manifests/, images/, and teacher_images/.",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=repo_root / "artifacts" / "data",
        help="Directory containing train_1024/eval_128/retention_64 JSONL files.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=repo_root / "artifacts" / "data" / "extraction_report.json",
        help="JSON extraction report path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan archives and verify every requested member without writing images.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing destination images. The default is to reuse non-empty files.",
    )
    return parser.parse_args()


def load_targets(manifest_dir: Path) -> tuple[set[str], set[str], int]:
    full_names: set[str] = set()
    crop_names: set[str] = set()
    sample_ids: set[str] = set()
    total = 0

    for filename, expected_size in EXPECTED_SPLIT_SIZES.items():
        path = manifest_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing manifest: {path}")

        split_count = 0
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                sample_id = record.get("sample_id")
                full_path = record.get("full_image_path")
                crop_path = record.get("crop_image_path")

                if not isinstance(sample_id, str) or not sample_id:
                    raise ValueError(f"{path}:{line_number}: invalid sample_id")
                if sample_id in sample_ids:
                    raise ValueError(f"Duplicate sample_id across manifests: {sample_id}")
                sample_ids.add(sample_id)

                full_names.add(validate_manifest_path(full_path, "images", path, line_number))
                crop_names.add(
                    validate_manifest_path(crop_path, "teacher_images", path, line_number)
                )
                split_count += 1
                total += 1

        if split_count != expected_size:
            raise ValueError(
                f"{path} contains {split_count} records; expected {expected_size}"
            )

    if len(full_names) != total:
        raise ValueError(f"Expected {total} unique Student images, found {len(full_names)}")
    if len(crop_names) != total:
        raise ValueError(f"Expected {total} unique Teacher images, found {len(crop_names)}")
    return full_names, crop_names, total


def validate_manifest_path(
    value: object, expected_dir: str, manifest: Path, line_number: int
) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{manifest}:{line_number}: missing {expected_dir} path")
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{manifest}:{line_number}: unsafe path {value!r}")
    if len(path.parts) != 2 or path.parts[0] != expected_dir or not path.name:
        raise ValueError(
            f"{manifest}:{line_number}: expected {expected_dir}/<file>, got {value!r}"
        )
    return path.name


def safe_member_basename(member_name: str) -> str | None:
    normalized = member_name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts or not path.name:
        return None
    return path.name


def scan_or_extract(
    archive: tarfile.TarFile,
    wanted: set[str],
    destination_dir: Path,
    *,
    label: str,
    dry_run: bool,
    overwrite: bool,
) -> dict[str, object]:
    found: set[str] = set()
    duplicates: set[str] = set()
    written = 0
    reused = 0
    members_scanned = 0
    started = time.monotonic()

    if not dry_run:
        destination_dir.mkdir(parents=True, exist_ok=True)

    for member in archive:
        members_scanned += 1
        if members_scanned % 1000 == 0:
            elapsed = time.monotonic() - started
            print(
                f"[{label}] scanned={members_scanned} found={len(found)}/{len(wanted)} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

        if not member.isfile():
            continue
        filename = safe_member_basename(member.name)
        if filename is None or filename not in wanted:
            continue
        if filename in found:
            duplicates.add(filename)
            continue
        found.add(filename)

        if dry_run:
            continue

        destination = destination_dir / filename
        if destination.exists() and not overwrite:
            if destination.is_file() and destination.stat().st_size > 0:
                reused += 1
                continue
            raise FileExistsError(f"Refusing to reuse invalid destination: {destination}")

        source = archive.extractfile(member)
        if source is None:
            raise tarfile.ExtractError(f"Cannot read archive member: {member.name}")

        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
        try:
            with source, temporary.open("wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
            if temporary.stat().st_size <= 0:
                raise tarfile.ExtractError(f"Extracted empty file: {member.name}")
            temporary.replace(destination)
            written += 1
        finally:
            if temporary.exists():
                temporary.unlink()

    missing = sorted(wanted - found)
    elapsed = time.monotonic() - started
    return {
        "label": label,
        "requested": len(wanted),
        "found": len(found),
        "written": written,
        "reused": reused,
        "members_scanned": members_scanned,
        "duplicates": sorted(duplicates),
        "missing": missing,
        "elapsed_seconds": round(elapsed, 3),
    }


def require_archives(raw_dir: Path) -> tuple[list[Path], Path]:
    student_parts = [raw_dir / "images" / f"images.tar.gz{i:02d}" for i in range(6)]
    teacher_archive = raw_dir / "teacher_images" / "teacher_images.tar.gz"
    missing = [str(path) for path in [*student_parts, teacher_archive] if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing archive files:\n" + "\n".join(missing))
    return student_parts, teacher_archive


def write_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        temporary.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    args = parse_args()
    raw_dir = args.raw_dir.resolve()
    subset_root = args.subset_root.resolve()
    manifest_dir = args.manifest_dir.resolve()
    report_path = args.report.resolve()

    full_names, crop_names, total = load_targets(manifest_dir)
    student_parts, teacher_archive = require_archives(raw_dir)

    print(f"Frozen samples: {total}")
    print(f"Student targets: {len(full_names)}")
    print(f"Teacher targets: {len(crop_names)}")
    print(f"Mode: {'dry-run' if args.dry_run else 'extract'}", flush=True)

    with ConcatenatedReader(student_parts) as raw_stream:
        with io.BufferedReader(raw_stream, buffer_size=1024 * 1024) as buffered:
            with tarfile.open(fileobj=buffered, mode="r|gz") as archive:
                student_result = scan_or_extract(
                    archive,
                    full_names,
                    subset_root / "images",
                    label="student",
                    dry_run=args.dry_run,
                    overwrite=args.overwrite,
                )

    with tarfile.open(teacher_archive, mode="r|gz") as archive:
        teacher_result = scan_or_extract(
            archive,
            crop_names,
            subset_root / "teacher_images",
            label="teacher",
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )

    success = not any(
        (
            student_result["missing"],
            student_result["duplicates"],
            teacher_result["missing"],
            teacher_result["duplicates"],
        )
    )
    report = {
        "schema_version": 1,
        "status": "PASS" if success else "FAIL",
        "mode": "dry-run" if args.dry_run else "extract",
        "raw_dir": str(raw_dir),
        "subset_root": str(subset_root),
        "manifest_dir": str(manifest_dir),
        "frozen_sample_count": total,
        "student": student_result,
        "teacher": teacher_result,
    }

    if not args.dry_run:
        write_report(report_path, report)
        print(f"Report: {report_path}")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if success else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError, tarfile.TarError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
