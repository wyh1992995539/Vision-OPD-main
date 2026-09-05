#!/usr/bin/env python3
"""Audit overlap between Vision-OPD project splits and frozen benchmarks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow.parquet as pq
import yaml
from PIL import Image, ImageOps


BENCHMARKS = ("zoombench", "mmstar", "vstar")
WHITESPACE_RE = re.compile(r"\s+")
PHASH_SIZE = 8
PHASH_HIGHFREQ_FACTOR = 4


@dataclass(frozen=True)
class ImageRef:
    owner: str
    dataset: str
    split: str
    sample_uid: str
    view: str
    path: str


@dataclass
class Sample:
    owner: str
    dataset: str
    split: str
    sample_uid: str
    source_id: str
    question: str
    normalized_question: str
    images: list[ImageRef]


def normalize_question(
    value: str,
    *,
    unicode_form: str = "NFKC",
    casefold: bool = True,
    collapse_whitespace: bool = True,
) -> str:
    normalized = unicodedata.normalize(unicode_form, str(value))
    if casefold:
        normalized = normalized.casefold()
    if collapse_whitespace:
        normalized = WHITESPACE_RE.sub(" ", normalized).strip()
    return normalized


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def _dct_transform(side: int) -> np.ndarray:
    indices = np.arange(side, dtype=np.float64)
    frequencies = indices[:, None]
    transform = np.cos(np.pi * (2 * indices + 1) * frequencies / (2 * side))
    transform[0, :] *= np.sqrt(1 / side)
    transform[1:, :] *= np.sqrt(2 / side)
    return transform


def phash64(path: Path) -> int:
    side = PHASH_SIZE * PHASH_HIGHFREQ_FACTOR
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("L")
        pixels = np.asarray(
            image.resize((side, side), Image.Resampling.LANCZOS),
            dtype=np.float64,
        )
    transform = _dct_transform(side)
    low = (transform @ pixels @ transform.T)[:PHASH_SIZE, :PHASH_SIZE]
    flat = low.ravel()
    threshold = float(np.median(flat[1:]))
    value = 0
    for enabled in flat > threshold:
        value = (value << 1) | int(enabled)
    return value


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def _resolve_image_path(value: Any, base_dir: Path) -> Path | None:
    if isinstance(value, dict):
        value = value.get("path")
    if not value:
        return None
    path = Path(str(value))
    return path if path.is_absolute() else (base_dir / path).resolve()


def _image_refs(
    values: Any,
    *,
    owner: str,
    dataset: str,
    split: str,
    sample_uid: str,
    view: str,
    base_dir: Path,
) -> list[ImageRef]:
    refs: list[ImageRef] = []
    for value in values or []:
        path = _resolve_image_path(value, base_dir)
        if path is not None:
            refs.append(ImageRef(owner, dataset, split, sample_uid, view, str(path)))
    return refs


def load_project_samples(
    parquet_paths: Iterable[str],
    *,
    normalization: dict[str, Any],
    include_full: bool,
    include_crop: bool,
) -> list[Sample]:
    samples: list[Sample] = []
    for raw_path in parquet_paths:
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"project split is missing: {path}")
        rows = pq.read_table(
            path,
            columns=["images", "bbox_images", "extra_info"],
        ).to_pylist()
        fallback_split = path.stem.split("_", 1)[0]
        for position, row in enumerate(rows):
            extra = row.get("extra_info") or {}
            provenance = extra.get("provenance") or {}
            split = str(provenance.get("split") or fallback_split)
            source_id = str(provenance.get("source_id") or f"{path.name}:row:{position}")
            sample_uid = str(provenance.get("sample_id") or source_id)
            question = str(extra.get("question") or "")
            images: list[ImageRef] = []
            if include_full:
                images.extend(_image_refs(
                    row.get("images"),
                    owner="project",
                    dataset="vision_opd_project",
                    split=split,
                    sample_uid=sample_uid,
                    view="full",
                    base_dir=path.parent,
                ))
            if include_crop:
                images.extend(_image_refs(
                    row.get("bbox_images"),
                    owner="project",
                    dataset="vision_opd_project",
                    split=split,
                    sample_uid=sample_uid,
                    view="crop",
                    base_dir=path.parent,
                ))
            samples.append(Sample(
                owner="project",
                dataset="vision_opd_project",
                split=split,
                sample_uid=sample_uid,
                source_id=source_id,
                question=question,
                normalized_question=normalize_question(question, **normalization),
                images=images,
            ))
    return samples


def load_benchmark_samples(
    data_root: Path,
    names: Iterable[str],
    *,
    normalization: dict[str, Any],
) -> list[Sample]:
    samples: list[Sample] = []
    for name in names:
        path = data_root / "converted" / name / f"{name}.json"
        if not path.is_file():
            raise FileNotFoundError(f"converted benchmark is missing: {path}")
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError(f"{path}: expected a JSON list")
        for position, row in enumerate(rows):
            sample_uid = str(row.get("sample_uid") or f"{name}:row:{position}")
            split = str(row.get("source_split") or "unknown")
            question = str(row.get("query") or "")
            images: list[ImageRef] = []
            images.extend(_image_refs(
                row.get("images"),
                owner="benchmark",
                dataset=name,
                split=split,
                sample_uid=sample_uid,
                view="full",
                base_dir=path.parent,
            ))
            images.extend(_image_refs(
                row.get("crop_images"),
                owner="benchmark",
                dataset=name,
                split=split,
                sample_uid=sample_uid,
                view="crop",
                base_dir=path.parent,
            ))
            samples.append(Sample(
                owner="benchmark",
                dataset=name,
                split=split,
                sample_uid=sample_uid,
                source_id=str(row.get("source_id") or position),
                question=question,
                normalized_question=normalize_question(question, **normalization),
                images=images,
            ))
    return samples


def profile_samples(samples: list[Sample]) -> dict[str, Any]:
    uids = [sample.sample_uid for sample in samples]
    empty_questions = sum(not sample.question.strip() for sample in samples)
    no_images = sum(not sample.images for sample in samples)
    missing_images = sum(
        not Path(image.path).is_file()
        for sample in samples
        for image in sample.images
    )
    return {
        "sample_count": len(samples),
        "unique_sample_uids": len(set(uids)),
        "duplicate_sample_uid_count": len(uids) - len(set(uids)),
        "image_reference_count": sum(len(sample.images) for sample in samples),
        "empty_question_count": empty_questions,
        "samples_without_images": no_images,
        "missing_image_reference_count": missing_images,
    }


def _load_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _write_cache(path: Path, cache: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _compute_image_fingerprint(
    task: tuple[str, bool, bool],
) -> tuple[str, dict[str, Any] | None, dict[str, str] | None]:
    raw_path, enable_sha256, enable_phash = task
    path = Path(raw_path)
    try:
        if not path.is_file():
            return raw_path, None, {"path": raw_path, "error": "missing_file"}
        stat = path.stat()
        value: dict[str, Any] = {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
        if enable_sha256:
            value["sha256"] = sha256_file(path)
        if enable_phash:
            value["phash_hex"] = f"{phash64(path):016x}"
        return raw_path, value, None
    except (OSError, SyntaxError, ValueError) as exc:
        return raw_path, None, {"path": raw_path, "error": f"{type(exc).__name__}: {exc}"}


def fingerprint_images(
    refs: list[ImageRef],
    *,
    cache_path: Path,
    enable_sha256: bool,
    enable_phash: bool,
    workers: int = 1,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, str]]]:
    cache = _load_cache(cache_path)
    fingerprints: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, str]] = []
    unique_paths = sorted({ref.path for ref in refs})
    pending: list[tuple[str, bool, bool]] = []
    for raw_path in unique_paths:
        path = Path(raw_path)
        if not path.is_file():
            errors.append({"path": raw_path, "error": "missing_file"})
            continue
        stat = path.stat()
        cached = cache.get(raw_path, {})
        valid_cache = (
            cached.get("size") == stat.st_size
            and cached.get("mtime_ns") == stat.st_mtime_ns
            and (not enable_sha256 or cached.get("sha256"))
            and (not enable_phash or cached.get("phash_hex"))
        )
        if valid_cache:
            fingerprints[raw_path] = cached
        else:
            pending.append((raw_path, enable_sha256, enable_phash))

    print(
        f"Reused {len(fingerprints)}/{len(unique_paths)} cached image fingerprints; "
        f"computing {len(pending)} with {max(1, workers)} worker(s)",
        flush=True,
    )
    executor = ProcessPoolExecutor(max_workers=max(1, workers)) if workers > 1 else None
    results = (
        executor.map(_compute_image_fingerprint, pending, chunksize=1)
        if executor is not None
        else map(_compute_image_fingerprint, pending)
    )
    try:
        for pending_index, (raw_path, value, error) in enumerate(results, start=1):
            if error is not None:
                errors.append(error)
            elif value is not None:
                fingerprints[raw_path] = value
                cache[raw_path] = value
            if pending_index % 100 == 0 or pending_index == len(pending):
                _write_cache(cache_path, cache)
                completed = len(fingerprints) + len(errors)
                print(f"Fingerprinted {completed}/{len(unique_paths)} unique images", flush=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    _write_cache(cache_path, cache)
    return fingerprints, errors


def _sample_lookup(samples: list[Sample]) -> dict[str, Sample]:
    return {sample.sample_uid: sample for sample in samples}


def _candidate_id(project_uid: str, benchmark_uid: str) -> str:
    value = f"{project_uid}\0{benchmark_uid}".encode("utf-8")
    return "overlap_" + hashlib.sha256(value).hexdigest()[:20]


def detect_candidates(
    project_samples: list[Sample],
    benchmark_samples: list[Sample],
    fingerprints: dict[str, dict[str, Any]],
    *,
    phash_threshold: int,
    enable_sha256: bool,
    enable_question: bool,
    enable_phash: bool,
) -> list[dict[str, Any]]:
    project_lookup = _sample_lookup(project_samples)
    benchmark_lookup = _sample_lookup(benchmark_samples)
    candidates: dict[tuple[str, str], dict[str, Any]] = {}

    def ensure(project_uid: str, benchmark_uid: str) -> dict[str, Any]:
        key = (project_uid, benchmark_uid)
        if key not in candidates:
            project = project_lookup[project_uid]
            benchmark = benchmark_lookup[benchmark_uid]
            candidates[key] = {
                "candidate_id": _candidate_id(project_uid, benchmark_uid),
                "benchmark": benchmark.dataset,
                "project_split": project.split,
                "project_sample_uid": project_uid,
                "project_source_id": project.source_id,
                "benchmark_sample_uid": benchmark_uid,
                "benchmark_source_id": benchmark.source_id,
                "match_types": [],
                "minimum_phash_distance": None,
                "question_evidence": None,
                "image_evidence": [],
                "review_status": "unresolved",
                "severity": "medium",
                "confidence": "medium",
            }
        return candidates[key]

    if enable_question:
        project_questions: dict[str, list[str]] = defaultdict(list)
        benchmark_questions: dict[str, list[str]] = defaultdict(list)
        for sample in project_samples:
            if sample.normalized_question:
                project_questions[sample.normalized_question].append(sample.sample_uid)
        for sample in benchmark_samples:
            if sample.normalized_question:
                benchmark_questions[sample.normalized_question].append(sample.sample_uid)
        for question in sorted(set(project_questions) & set(benchmark_questions)):
            for project_uid in project_questions[question]:
                for benchmark_uid in benchmark_questions[question]:
                    candidate = ensure(project_uid, benchmark_uid)
                    candidate["match_types"].append("exact_question_match")
                    candidate["question_evidence"] = {
                        "normalized_question": question,
                        "project_question": project_lookup[project_uid].question,
                        "benchmark_question": benchmark_lookup[benchmark_uid].question,
                    }

    project_images = [image for sample in project_samples for image in sample.images if image.path in fingerprints]
    benchmark_images = [image for sample in benchmark_samples for image in sample.images if image.path in fingerprints]

    if enable_sha256:
        by_sha: dict[str, list[ImageRef]] = defaultdict(list)
        for image in project_images:
            digest = fingerprints[image.path].get("sha256")
            if digest:
                by_sha[digest].append(image)
        for benchmark_image in benchmark_images:
            digest = fingerprints[benchmark_image.path].get("sha256")
            if not digest:
                continue
            for project_image in by_sha.get(digest, []):
                candidate = ensure(project_image.sample_uid, benchmark_image.sample_uid)
                if "exact_image_match" not in candidate["match_types"]:
                    candidate["match_types"].append("exact_image_match")
                candidate["image_evidence"].append({
                    "match_type": "exact_image_match",
                    "project_view": project_image.view,
                    "project_path": project_image.path,
                    "benchmark_view": benchmark_image.view,
                    "benchmark_path": benchmark_image.path,
                    "sha256": digest,
                    "phash_distance": 0,
                })
                candidate["minimum_phash_distance"] = 0
                candidate["review_status"] = "confirmed_overlap"
                candidate["severity"] = "high"
                candidate["confidence"] = "high"

    if enable_phash:
        project_hashes = [
            (image, int(fingerprints[image.path]["phash_hex"], 16))
            for image in project_images
            if fingerprints[image.path].get("phash_hex")
        ]
        benchmark_hashes = [
            (image, int(fingerprints[image.path]["phash_hex"], 16))
            for image in benchmark_images
            if fingerprints[image.path].get("phash_hex")
        ]
        for benchmark_image, benchmark_hash in benchmark_hashes:
            benchmark_sha = fingerprints[benchmark_image.path].get("sha256")
            for project_image, project_hash in project_hashes:
                if (
                    enable_sha256
                    and benchmark_sha
                    and benchmark_sha == fingerprints[project_image.path].get("sha256")
                ):
                    continue
                distance = hamming_distance(project_hash, benchmark_hash)
                if distance > phash_threshold:
                    continue
                candidate = ensure(project_image.sample_uid, benchmark_image.sample_uid)
                if "suspected_perceptual_match" not in candidate["match_types"]:
                    candidate["match_types"].append("suspected_perceptual_match")
                candidate["image_evidence"].append({
                    "match_type": "suspected_perceptual_match",
                    "project_view": project_image.view,
                    "project_path": project_image.path,
                    "project_phash": f"{project_hash:016x}",
                    "benchmark_view": benchmark_image.view,
                    "benchmark_path": benchmark_image.path,
                    "benchmark_phash": f"{benchmark_hash:016x}",
                    "phash_distance": distance,
                })
                current = candidate["minimum_phash_distance"]
                candidate["minimum_phash_distance"] = distance if current is None else min(current, distance)

    result = list(candidates.values())
    for candidate in result:
        candidate["match_types"] = sorted(set(candidate["match_types"]))
        candidate["image_evidence"].sort(
            key=lambda item: (
                item["phash_distance"],
                item["project_view"],
                item["benchmark_view"],
                item["project_path"],
                item["benchmark_path"],
            )
        )
    result.sort(key=lambda item: (item["benchmark"], item["project_sample_uid"], item["benchmark_sample_uid"]))
    return result


def apply_manual_decisions(
    candidates: list[dict[str, Any]],
    decision_path: Path,
) -> None:
    if not decision_path.is_file():
        return
    payload = json.loads(decision_path.read_text(encoding="utf-8"))
    decisions = payload.get("decisions", [])
    if not isinstance(decisions, list):
        raise ValueError(f"{decision_path}: decisions must be a list")
    by_id = {item["candidate_id"]: item for item in candidates}
    seen: set[str] = set()
    for decision in decisions:
        candidate_id = str(decision.get("candidate_id", ""))
        status = str(decision.get("review_status", ""))
        reviewer = str(decision.get("reviewer", "")).strip()
        note = str(decision.get("review_note", "")).strip()
        if candidate_id not in by_id:
            raise ValueError(f"{decision_path}: unknown candidate_id {candidate_id!r}")
        if candidate_id in seen:
            raise ValueError(f"{decision_path}: duplicate candidate_id {candidate_id!r}")
        if status not in {"confirmed_overlap", "dismissed"}:
            raise ValueError(f"{decision_path}: invalid review_status {status!r}")
        if not reviewer or not note:
            raise ValueError(f"{decision_path}: reviewer and review_note are required")
        candidate = by_id[candidate_id]
        candidate["review_status"] = status
        candidate["manual_review"] = {
            "reviewer": reviewer,
            "review_note": note,
            "reviewed_at_utc": str(decision.get("reviewed_at_utc", "")),
            "decision_source": str(decision_path),
        }
        seen.add(candidate_id)


def summarize(
    project_samples: list[Sample],
    benchmark_samples: list[Sample],
    candidates: list[dict[str, Any]],
    fingerprint_errors: list[dict[str, str]],
    *,
    phash_threshold: int,
) -> dict[str, Any]:
    project_profile = profile_samples(project_samples)
    benchmark_profile = profile_samples(benchmark_samples)
    benchmark_names = sorted({sample.dataset for sample in benchmark_samples})
    per_benchmark: dict[str, dict[str, Any]] = {}
    for name in benchmark_names:
        population = [sample for sample in benchmark_samples if sample.dataset == name]
        matched = [item for item in candidates if item["benchmark"] == name]
        confirmed = [item for item in matched if item["review_status"] == "confirmed_overlap"]
        unresolved = [item for item in matched if item["review_status"] == "unresolved"]
        dismissed = [item for item in matched if item["review_status"] == "dismissed"]
        impacted = {item["benchmark_sample_uid"] for item in confirmed}
        per_benchmark[name] = {
            "benchmark_sample_count": len(population),
            "candidate_pair_count": len(matched),
            "confirmed_overlap_pair_count": len(confirmed),
            "unresolved_pair_count": len(unresolved),
            "dismissed_pair_count": len(dismissed),
            "confirmed_impacted_benchmark_samples": len(impacted),
            "confirmed_impacted_rate": len(impacted) / len(population) if population else 0.0,
            "exact_image_pair_count": sum("exact_image_match" in x["match_types"] for x in matched),
            "exact_question_pair_count": sum("exact_question_match" in x["match_types"] for x in matched),
            "suspected_perceptual_pair_count": sum("suspected_perceptual_match" in x["match_types"] for x in matched),
        }
    confirmed_total = sum(x["review_status"] == "confirmed_overlap" for x in candidates)
    unresolved_total = sum(x["review_status"] == "unresolved" for x in candidates)
    dismissed_total = sum(x["review_status"] == "dismissed" for x in candidates)
    quality_failures = (
        project_profile["duplicate_sample_uid_count"]
        + project_profile["empty_question_count"]
        + project_profile["samples_without_images"]
        + project_profile["missing_image_reference_count"]
        + benchmark_profile["duplicate_sample_uid_count"]
        + benchmark_profile["empty_question_count"]
        + benchmark_profile["samples_without_images"]
        + benchmark_profile["missing_image_reference_count"]
        + len(fingerprint_errors)
    )
    if quality_failures:
        status = "blocked_data_quality"
    elif confirmed_total:
        status = "confirmed_overlap"
    elif unresolved_total:
        status = "manual_review_required"
    else:
        status = "no_detected_overlap"
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision_status": status,
        "phash": {
            "algorithm": "dct_phash",
            "hash_size": PHASH_SIZE,
            "bits": PHASH_SIZE * PHASH_SIZE,
            "suspected_max_hamming_distance": phash_threshold,
            "exif_transpose": True,
        },
        "project_profile": project_profile,
        "benchmark_profile": benchmark_profile,
        "per_benchmark": per_benchmark,
        "candidate_pair_count": len(candidates),
        "confirmed_overlap_pair_count": confirmed_total,
        "unresolved_pair_count": unresolved_total,
        "dismissed_pair_count": dismissed_total,
        "fingerprint_error_count": len(fingerprint_errors),
        "fingerprint_errors": fingerprint_errors,
        "reporting_policy": {
            "preserve_official_test_samples": True,
            "report_official_full_score": True,
            "report_deduplicated_diagnostic_separately": True,
            "forbid_claiming_fully_independent_test_when_confirmed_or_unresolved": True,
        },
    }


def write_outputs(
    output_dir: Path,
    report: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "overlap_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "overlap_candidates.jsonl").open("w", encoding="utf-8") as handle:
        for candidate in candidates:
            handle.write(json.dumps(candidate, ensure_ascii=False, sort_keys=True) + "\n")
    with (output_dir / "manual_review.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "candidate_id",
            "benchmark",
            "project_split",
            "project_sample_uid",
            "benchmark_sample_uid",
            "match_types",
            "minimum_phash_distance",
            "review_status",
            "reviewer",
            "review_note",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for candidate in candidates:
            writer.writerow({
                "candidate_id": candidate["candidate_id"],
                "benchmark": candidate["benchmark"],
                "project_split": candidate["project_split"],
                "project_sample_uid": candidate["project_sample_uid"],
                "benchmark_sample_uid": candidate["benchmark_sample_uid"],
                "match_types": ",".join(candidate["match_types"]),
                "minimum_phash_distance": candidate["minimum_phash_distance"],
                "review_status": candidate["review_status"],
                "reviewer": candidate.get("manual_review", {}).get("reviewer", ""),
                "review_note": candidate.get("manual_review", {}).get("review_note", ""),
            })

    lines = [
        "# Vision-OPD Benchmark Overlap Audit",
        "",
        f"**Audit execution:** `{report.get('audit_execution_status', 'NOT_RECORDED')}`",
        "",
        f"**Decision status:** `{report['decision_status']}`",
        "",
        "## Technical summary",
        "",
        f"- Project samples: {report['project_profile']['sample_count']}; benchmark samples: {report['benchmark_profile']['sample_count']}.",
        f"- Candidate sample pairs: {report['candidate_pair_count']}; confirmed: {report['confirmed_overlap_pair_count']}; dismissed: {report['dismissed_pair_count']}; unresolved: {report['unresolved_pair_count']}.",
        f"- Fingerprint/data errors: {report['fingerprint_error_count']}.",
        f"- Input/data gate checks: {sum(report.get('input_gate_checks', {}).values())}/{len(report.get('input_gate_checks', {}))} passed.",
        "",
        "## Results by benchmark",
        "",
        "| Benchmark | Samples | Candidates | Confirmed | Dismissed | Unresolved | Confirmed impacted rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, values in sorted(report["per_benchmark"].items()):
        lines.append(
            f"| {name} | {values['benchmark_sample_count']} | {values['candidate_pair_count']} | "
            f"{values['confirmed_overlap_pair_count']} | {values['dismissed_pair_count']} | {values['unresolved_pair_count']} | "
            f"{100 * values['confirmed_impacted_rate']:.3f}% |"
        )
    lines.extend([
        "",
        "## Method",
        "",
        "- Exact images: SHA256 equality across project and benchmark image references.",
        "- Exact questions: NFKC normalization, Unicode casefold, and whitespace collapse.",
        f"- Perceptual images: 64-bit DCT pHash after EXIF transpose; distance <= {report['phash']['suspected_max_hamming_distance']} is unresolved until manual review.",
        "- Exact image matches are automatically confirmed; question-only and pHash-only matches remain unresolved.",
        "",
        "## Reporting consequence",
        "",
        "Official full-set scores must always be preserved. If confirmed or unresolved candidates exist, any deduplicated score is diagnostic only and the benchmark must not be described as fully independent.",
        "",
        "## Files",
        "",
        "- Machine-readable summary: `overlap_report.json`",
        "- Candidate evidence: `overlap_candidates.jsonl`",
        "- Manual review sheet: `manual_review.csv`",
        "- Applied review decisions: `manual_review_decisions.json`",
        "- Reusable image fingerprint cache: `image_fingerprint_cache.json`",
        "",
    ])
    (output_dir / "overlap_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_audit(
    config_path: str | Path,
    *,
    benchmarks: Iterable[str] = BENCHMARKS,
    output_dir_override: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    selected = list(dict.fromkeys(benchmarks))
    unknown = sorted(set(selected) - set(BENCHMARKS))
    if unknown:
        raise ValueError(f"unsupported benchmark(s): {', '.join(unknown)}")
    audit_cfg = config["overlap_audit"]
    checks = audit_cfg["checks"]
    normalization_cfg = checks["normalized_question"]
    normalization = {
        "unicode_form": str(normalization_cfg.get("unicode_normalization", "NFKC")),
        "casefold": bool(normalization_cfg.get("casefold", True)),
        "collapse_whitespace": bool(normalization_cfg.get("collapse_whitespace", True)),
    }
    repo_root = config_path.parent.parent
    run_root = Path(config["paths"]["run_root"])
    if not run_root.is_absolute():
        run_root = repo_root / run_root
    output_dir = Path(output_dir_override).resolve() if output_dir_override else run_root / "overlap"
    report_path = output_dir / "overlap_report.json"
    if report_path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite {report_path}; pass --force")

    data_root = Path(config["paths"]["data_root"]).resolve()
    project_samples = load_project_samples(
        audit_cfg["project_splits"],
        normalization=normalization,
        include_full=bool(audit_cfg.get("include_project_full_images", True)),
        include_crop=bool(audit_cfg.get("include_project_crop_images", True)),
    )
    benchmark_samples = load_benchmark_samples(
        data_root,
        selected,
        normalization=normalization,
    )
    project_inputs = []
    for raw_path in audit_cfg["project_splits"]:
        path = Path(raw_path).resolve()
        project_inputs.append({"path": str(path), "sha256": sha256_file(path)})
    benchmark_inputs = {}
    for name in selected:
        path = data_root / "converted" / name / f"{name}.json"
        rows = json.loads(path.read_text(encoding="utf-8"))
        benchmark_inputs[name] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "sample_count": len(rows),
        }
    source_benchmark = config.get("protocol", {}).get("source_benchmark_config")
    source_benchmark_evidence = None
    if source_benchmark:
        source_path = Path(source_benchmark["path"])
        source_path = source_path if source_path.is_absolute() else (repo_root / source_path).resolve()
        source_benchmark_evidence = {"path": str(source_path), "sha256": sha256_file(source_path)}
    all_refs = [
        image
        for sample in project_samples + benchmark_samples
        for image in sample.images
    ]
    fingerprints, fingerprint_errors = fingerprint_images(
        all_refs,
        cache_path=output_dir / "image_fingerprint_cache.json",
        enable_sha256=bool(checks["file_sha256"].get("enabled", True)),
        enable_phash=bool(checks["perceptual_hash"].get("enabled", True)),
        workers=int(audit_cfg.get("fingerprint_workers", 1)),
    )
    candidates = detect_candidates(
        project_samples,
        benchmark_samples,
        fingerprints,
        phash_threshold=int(checks["perceptual_hash"]["suspected_max_hamming_distance"]),
        enable_sha256=bool(checks["file_sha256"].get("enabled", True)),
        enable_question=bool(checks["normalized_question"].get("enabled", True)),
        enable_phash=bool(checks["perceptual_hash"].get("enabled", True)),
    )
    apply_manual_decisions(candidates, output_dir / "manual_review_decisions.json")
    report = summarize(
        project_samples,
        benchmark_samples,
        candidates,
        fingerprint_errors,
        phash_threshold=int(checks["perceptual_hash"]["suspected_max_hamming_distance"]),
    )
    expected_project_count = audit_cfg.get("expected_project_sample_count")
    expected_project_hashes = audit_cfg.get("expected_project_split_sha256", {})
    expected_benchmark_counts = audit_cfg.get("expected_benchmark_sample_counts", {})
    expected_benchmark_hashes = audit_cfg.get("expected_benchmark_sha256", {})
    project_hashes_match = all(
        not expected_project_hashes
        or expected_project_hashes.get(item["path"]) == item["sha256"]
        for item in project_inputs
    ) and (not expected_project_hashes or len(expected_project_hashes) == len(project_inputs))
    benchmark_counts_match = (
        all(
            name in expected_benchmark_counts
            and int(expected_benchmark_counts[name]) == benchmark_inputs[name]["sample_count"]
            for name in selected
        )
        if expected_benchmark_counts
        else True
    )
    benchmark_hashes_match = (
        all(
            name in expected_benchmark_hashes
            and expected_benchmark_hashes[name] == benchmark_inputs[name]["sha256"]
            for name in selected
        )
        if expected_benchmark_hashes
        else True
    )
    source_hash_matches = (
        source_benchmark_evidence is not None
        and source_benchmark_evidence["sha256"] == source_benchmark["sha256"]
        if source_benchmark
        else True
    )
    profile_fields = (
        "duplicate_sample_uid_count",
        "empty_question_count",
        "samples_without_images",
        "missing_image_reference_count",
    )
    input_gate_checks = {
        "project_sample_count_matches": (
            expected_project_count is None
            or report["project_profile"]["sample_count"] == int(expected_project_count)
        ),
        "project_split_sha256_matches": project_hashes_match,
        "selected_benchmarks_complete": (
            not audit_cfg.get("require_all_benchmarks", False)
            or set(selected) == set(BENCHMARKS)
        ),
        "benchmark_sample_counts_match": benchmark_counts_match,
        "benchmark_sha256_matches": benchmark_hashes_match,
        "source_benchmark_config_sha256_matches": source_hash_matches,
        "project_profile_quality_passes": all(
            report["project_profile"][field] == 0 for field in profile_fields
        ),
        "benchmark_profile_quality_passes": all(
            report["benchmark_profile"][field] == 0 for field in profile_fields
        ),
        "fingerprint_errors_zero": report["fingerprint_error_count"] == 0,
    }
    report["audit_execution_status"] = "PASS" if all(input_gate_checks.values()) else "FAIL"
    report["input_gate_checks"] = input_gate_checks
    report["provenance"] = {
        "audit_config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "source_benchmark_config": source_benchmark_evidence,
        "project_splits": project_inputs,
        "benchmark_converted_inputs": benchmark_inputs,
        "selected_benchmarks": selected,
    }
    if report["audit_execution_status"] != "PASS":
        report["decision_status"] = "blocked_data_quality"
    write_outputs(output_dir, report, candidates)
    print(
        f"Overlap audit: execution={report['audit_execution_status']} "
        f"status={report['decision_status']} "
        f"candidates={report['candidate_pair_count']} output={output_dir}",
        flush=True,
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Vision-OPD/benchmark overlap")
    parser.add_argument("--config", required=True, help="Explicit legacy E-D5/E-D6 config path; R3 uses the paper-aligned entrypoints")
    parser.add_argument("--benchmarks", default=",".join(BENCHMARKS))
    parser.add_argument("--output-dir")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    names = [value.strip() for value in args.benchmarks.split(",") if value.strip()]
    run_audit(
        args.config,
        benchmarks=names,
        output_dir_override=args.output_dir,
        force=args.force,
    )


if __name__ == "__main__":
    main()

