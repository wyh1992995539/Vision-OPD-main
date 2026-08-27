#!/usr/bin/env python3
"""Create the frozen deterministic Day 5 benchmark Smoke selection manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml
from PIL import Image


SUPPORTED_BENCHMARKS = ("zoombench", "mmstar", "vstar")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(value: str | Path, repo_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (repo_root / path).resolve()


def parse_benchmarks(value: str) -> list[str]:
    names = list(dict.fromkeys(part.strip() for part in value.split(",") if part.strip()))
    unknown = sorted(set(names) - set(SUPPORTED_BENCHMARKS))
    if unknown:
        raise ValueError(f"unsupported benchmark(s): {', '.join(unknown)}")
    if not names:
        raise ValueError("at least one benchmark is required")
    return names


def rank_sample(seed: int, benchmark: str, sample_uid: str) -> str:
    value = f"{seed}:{benchmark}:{sample_uid}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _image_paths(row: dict[str, Any]) -> list[Path]:
    values = list(row.get("images") or []) + list(row.get("crop_images") or [])
    paths: list[Path] = []
    for value in values:
        if isinstance(value, dict):
            value = value.get("path")
        if value:
            paths.append(Path(str(value)))
    return paths


def validate_row(
    row: dict[str, Any],
    *,
    benchmark: str,
    expected_revision: str,
    stratum_key: str,
) -> None:
    required = ("sample_uid", "source_id", "query", "response", "source_revision")
    missing = [key for key in required if not str(row.get(key, "")).strip()]
    if missing:
        raise ValueError(f"{benchmark}: sample is missing required fields {missing}")
    if str(row["source_revision"]) != expected_revision:
        raise ValueError(
            f"{benchmark}: {row['sample_uid']} revision {row['source_revision']!r} "
            f"does not match frozen revision {expected_revision!r}"
        )
    if not str(row.get(stratum_key, "")).strip():
        raise ValueError(f"{benchmark}: {row['sample_uid']} has empty {stratum_key}")
    paths = _image_paths(row)
    if not paths:
        raise ValueError(f"{benchmark}: {row['sample_uid']} has no image references")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"{benchmark}: {row['sample_uid']} missing image {path}")
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception as exc:
            raise ValueError(f"{benchmark}: {row['sample_uid']} has undecodable image {path}: {exc}") from exc


def load_overlap_statuses(overlap_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    path = overlap_dir / "overlap_candidates.jsonl"
    if not path.is_file():
        return {}
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        key = (str(item["benchmark"]), str(item["benchmark_sample_uid"]))
        result[key] = {
            "status": str(item.get("review_status", "unresolved")),
            "candidate_id": str(item.get("candidate_id", "")),
        }
    return result


def select_rows(
    rows: list[dict[str, Any]],
    *,
    benchmark: str,
    seed: int,
    stratum_key: str,
    quotas: dict[str, int],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for stratum, quota in quotas.items():
        pool = [row for row in rows if str(row.get(stratum_key, "")) == stratum]
        if len(pool) < quota:
            raise ValueError(
                f"{benchmark}: stratum {stratum!r} has {len(pool)} samples, below frozen quota {quota}"
            )
        pool.sort(key=lambda row: rank_sample(seed, benchmark, str(row["sample_uid"])))
        selected.extend(pool[:quota])
    uids = [str(row["sample_uid"]) for row in selected]
    if len(uids) != len(set(uids)):
        raise ValueError(f"{benchmark}: selected duplicate sample_uid")
    return selected


def build_manifest(
    config_path: Path,
    *,
    benchmarks: Iterable[str],
) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    repo_root = config_path.parent.parent
    smoke = config["smoke"]
    if smoke.get("selection_method") != "sha256_ranked_stratified":
        raise ValueError("smoke.selection_method must be sha256_ranked_stratified")
    seed = int(smoke["selection_seed"])
    per_benchmark = int(smoke["samples_per_benchmark"])
    data_root = resolve_path(config["paths"]["data_root"], repo_root)
    overlap_dir = resolve_path(config["paths"]["overlap_dir"], repo_root)
    overlap_statuses = load_overlap_statuses(overlap_dir)

    manifest_benchmarks: dict[str, Any] = {}
    for benchmark in benchmarks:
        benchmark_cfg = config["benchmarks"][benchmark]
        selection_cfg = smoke["selection"][benchmark]
        stratum_key = str(selection_cfg["stratify_by"])
        quotas = {str(key): int(value) for key, value in selection_cfg["quotas"].items()}
        if sum(quotas.values()) != per_benchmark:
            raise ValueError(
                f"{benchmark}: frozen quotas sum to {sum(quotas.values())}, expected {per_benchmark}"
            )
        converted_path = data_root / "converted" / benchmark / f"{benchmark}.json"
        rows = json.loads(converted_path.read_text(encoding="utf-8"))
        if len(rows) != int(benchmark_cfg["expected_sample_count"]):
            raise ValueError(
                f"{benchmark}: converted row count {len(rows)} does not match expected_sample_count"
            )
        for row in rows:
            validate_row(
                row,
                benchmark=benchmark,
                expected_revision=str(benchmark_cfg["dataset_revision"]),
                stratum_key=stratum_key,
            )
        selected = select_rows(
            rows,
            benchmark=benchmark,
            seed=seed,
            stratum_key=stratum_key,
            quotas=quotas,
        )
        samples = []
        for row in selected:
            uid = str(row["sample_uid"])
            overlap = overlap_statuses.get((benchmark, uid))
            samples.append({
                "sample_uid": uid,
                "source_id": str(row["source_id"]),
                "stratum": str(row[stratum_key]),
                "selection_rank": rank_sample(seed, benchmark, uid),
                "image_sha256": str(row.get("image_sha256", "")),
                "crop_image_sha256": list(row.get("crop_image_sha256") or []),
                "overlap_status": overlap["status"] if overlap else "none",
                "overlap_candidate_id": overlap["candidate_id"] if overlap else None,
            })
        manifest_benchmarks[benchmark] = {
            "source_revision": str(benchmark_cfg["dataset_revision"]),
            "converted_data_sha256": sha256_file(converted_path),
            "stratify_by": stratum_key,
            "quotas": quotas,
            "sample_count": len(samples),
            "sample_uids": [item["sample_uid"] for item in samples],
            "samples": samples,
        }

    return {
        "schema_version": 1,
        "experiment_id": str(config["protocol"]["experiment_id"]),
        "protocol_revision": int(config["protocol"]["protocol_revision"]),
        "selected_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_seed": seed,
        "selection_method": str(smoke["selection_method"]),
        "selection_rank_input": str(smoke["selection_rank_input"]),
        "selection_rank_order": str(smoke["selection_rank_order"]),
        "config_sha256": sha256_file(config_path),
        "benchmarks": manifest_benchmarks,
    }


def write_manifest(manifest: dict[str, Any], output_path: Path, *, force: bool) -> Path:
    if output_path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite frozen manifest: {output_path}; pass --force")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temp_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp_path.replace(output_path)
    hash_path = output_path.with_suffix(".sha256")
    hash_path.write_text(f"{sha256_file(output_path)}  {output_path}\n", encoding="utf-8")
    return hash_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Explicit legacy E-D5/E-D6 config path; R3 uses the paper-aligned entrypoints")
    parser.add_argument("--benchmarks", default=",".join(SUPPORTED_BENCHMARKS))
    parser.add_argument("--output")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    repo_root = config_path.parent.parent
    output_path = (
        resolve_path(args.output, repo_root)
        if args.output
        else resolve_path(config["smoke"]["manifest"], repo_root)
    )
    manifest = build_manifest(config_path, benchmarks=parse_benchmarks(args.benchmarks))
    hash_path = write_manifest(manifest, output_path, force=args.force)
    print(f"Frozen Smoke manifest: {output_path}")
    print(f"SHA256 manifest: {hash_path}")


if __name__ == "__main__":
    main()
