#!/usr/bin/env python3
"""Prepare the frozen Vision-OPD 6,241-row dataset with fail-closed gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(*args: str) -> None:
    subprocess.run(args, cwd=PROJECT_ROOT, check=True)


def configure_environment(config: dict[str, Any]) -> None:
    cache = config["storage"]["cache_paths"]
    values = {
        "HF_HOME": cache["hf_home"],
        "HUGGINGFACE_HUB_CACHE": str(Path(cache["hf_home"]) / "hub"),
        "HF_DATASETS_CACHE": str(Path(cache["hf_home"]) / "datasets"),
        "TORCH_HOME": cache["torch_home"],
        "PIP_CACHE_DIR": cache["pip_cache_dir"],
        "XDG_CACHE_HOME": cache["xdg_cache_home"],
        "TMPDIR": cache["tmpdir"],
    }
    for name, value in values.items():
        os.environ[name] = str(value)
        Path(value).mkdir(parents=True, exist_ok=True)


def disk_snapshot(config: dict[str, Any], *, scope: str) -> dict[str, Any]:
    root = Path(config["storage"]["data_root"])
    usage = shutil.disk_usage(root)
    system_usage = shutil.disk_usage("/")
    dataset_root = Path(config["storage"]["dataset_root"])
    dataset_bytes = int(
        subprocess.run(
            ["du", "-sb", str(dataset_root)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.split()[0]
    )
    git_status = subprocess.run(
        ["git", "status", "--short"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": scope,
        "git_status_short": git_status,
        "path": str(root),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
        "free_gib": usage.free / (1024**3),
        "system_disk_free_bytes": system_usage.free,
        "dataset_root_bytes": dataset_bytes,
        "cache_environment": {
            name: os.environ.get(name)
            for name in (
                "HF_HOME", "HUGGINGFACE_HUB_CACHE", "HF_DATASETS_CACHE",
                "TORCH_HOME", "PIP_CACHE_DIR", "XDG_CACHE_HOME", "TMPDIR",
            )
        },
    }


def pre_download_storage_gate(config: dict[str, Any]) -> dict[str, Any]:
    """Reserve room for remaining archives, extraction, and post-extraction safety."""
    storage = config["storage"]
    plan = storage["full_download_peak_plan"]
    raw_root = Path(storage["raw_download_root"])
    dataset_root = Path(storage["dataset_root"])
    expected_archive_bytes = sum(
        int(item["byte_size"]) for item in config["data"]["source_archives"].values()
    )
    present_archive_bytes = sum(
        min((raw_root / relative).stat().st_size, int(item["byte_size"]))
        for relative, item in config["data"]["source_archives"].items()
        if (raw_root / relative).is_file()
    )
    extracted_bytes = sum(
        path.stat().st_size
        for directory in (dataset_root / "images", dataset_root / "teacher_images")
        if directory.is_dir()
        for path in directory.rglob("*")
        if path.is_file()
    )
    remaining_archive_bytes = max(0, expected_archive_bytes - present_archive_bytes)
    remaining_extracted_bytes = max(
        0, int(plan["conservative_extracted_bytes"]) - extracted_bytes
    )
    safety_free_bytes = int(plan["safety_free_bytes_after_extraction"])
    free_bytes = shutil.disk_usage(storage["data_root"]).free
    required_free_bytes = remaining_archive_bytes + remaining_extracted_bytes + safety_free_bytes
    return {
        "status": "PASS" if free_bytes >= required_free_bytes else "FAIL",
        "free_bytes": free_bytes,
        "expected_archive_bytes": expected_archive_bytes,
        "present_archive_bytes": present_archive_bytes,
        "remaining_archive_bytes": remaining_archive_bytes,
        "conservative_extracted_bytes": int(plan["conservative_extracted_bytes"]),
        "present_extracted_bytes": extracted_bytes,
        "remaining_extracted_bytes": remaining_extracted_bytes,
        "safety_free_bytes_after_extraction": safety_free_bytes,
        "required_free_bytes_for_remaining_work": required_free_bytes,
    }


def download(config: dict[str, Any], *, include_images: bool) -> None:
    from huggingface_hub import snapshot_download

    raw_root = Path(config["storage"]["raw_download_root"])
    raw_root.mkdir(parents=True, exist_ok=True)
    patterns = ["train.jsonl"]
    if include_images:
        patterns.extend(["images/images.tar.gz*", "teacher_images/teacher_images.tar.gz"])
    snapshot_download(
        repo_id=config["data"]["source"],
        repo_type="dataset",
        revision=config["data"]["source_revision"],
        local_dir=raw_root,
        allow_patterns=patterns,
    )


def verify_archives(config: dict[str, Any]) -> dict[str, Any]:
    raw_root = Path(config["storage"]["raw_download_root"])
    results = {}
    for relative, expected in config["data"]["source_archives"].items():
        path = raw_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"source archive is missing: {path}")
        actual = {"byte_size": path.stat().st_size, "sha256": sha256_file(path)}
        checks = {
            "byte_size": actual["byte_size"] == int(expected["byte_size"]),
            "sha256": actual["sha256"] == expected["sha256"],
        }
        if not all(checks.values()):
            raise ValueError(f"source archive identity mismatch for {relative}: {checks}")
        results[relative] = actual
    return results


def verify_source(config: dict[str, Any]) -> Path:
    source = Path(config["data"]["source_metadata"]["local_path"])
    expected = config["data"]["source_metadata"]
    if not source.is_file():
        raise FileNotFoundError(f"frozen source metadata is missing: {source}")
    checks = {
        "sha256": sha256_file(source) == expected["sha256"],
        "byte_size": source.stat().st_size == int(expected["byte_size"]),
        "record_count": sum(1 for line in source.open(encoding="utf-8") if line.strip())
        == int(expected["record_count"]),
    }
    if not all(checks.values()):
        raise ValueError(f"source metadata identity mismatch: {checks}")
    return source


def manifest_dir(config: dict[str, Any]) -> Path:
    value = Path(config["data"]["manifests"]["train_jsonl"])
    return (value if value.is_absolute() else PROJECT_ROOT / value).parent


def reconcile_historical_ids(candidate_path: Path) -> dict[str, Any]:
    old_root = Path("/root/autodl-tmp/data/vision_opd_1024/manifests")
    old_files = [old_root / name for name in ("train_1024.jsonl", "eval_128.jsonl", "retention_64.jsonl")]
    missing = [str(path) for path in old_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"historical manifests are missing: {missing}")
    candidate_ids = {
        json.loads(line)["sample_id"]
        for line in candidate_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    old_ids = []
    for path in old_files:
        old_ids.extend(
            json.loads(line)["sample_id"]
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    missing_ids = sorted(set(old_ids) - candidate_ids)
    if missing_ids:
        raise ValueError(f"historical sample IDs missing from regenerated candidate manifest: {missing_ids[:10]}")
    return {
        "historical_rows": len(old_ids),
        "historical_unique_ids": len(set(old_ids)),
        "candidate_unique_ids": len(candidate_ids),
        "missing_historical_ids": 0,
    }


def prepare_metadata(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    source = verify_source(config)
    output = manifest_dir(config)
    run(
        sys.executable,
        "scripts/prepare_project_subset.py",
        "--config", str(config_path),
        "--input-jsonl", str(source),
        "--output-dir", str(output),
    )
    candidate = output / "candidate_manifest.jsonl"
    reconciliation = reconcile_historical_ids(candidate)
    return {
        "source_path": str(source),
        "source_sha256": sha256_file(source),
        "manifest_dir": str(output),
        "candidate_sha256": sha256_file(candidate),
        "historical_reconciliation": reconciliation,
    }


def prepare_images(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    archive_identity = verify_archives(config)
    root = Path(config["storage"]["dataset_root"])
    manifests = manifest_dir(config)
    validation = PROJECT_ROOT / config["data"]["frozen_outputs"]["validation"]
    statistics = PROJECT_ROOT / config["data"]["frozen_outputs"]["statistics"]
    hashes = PROJECT_ROOT / config["data"]["frozen_outputs"]["sha256"]
    extraction = manifests / "extraction_report.json"
    run(
        sys.executable, "scripts/extract_project_images.py",
        "--config", str(config_path), "--raw-dir", config["storage"]["raw_download_root"],
        "--subset-root", str(root), "--manifest-dir", str(manifests), "--report", str(extraction),
    )
    run(
        sys.executable, "scripts/validate_project_data.py",
        "--config", str(config_path), "--manifest-dir", str(manifests),
        "--subset-root", str(root), "--validation-report", str(validation),
        "--statistics-report", str(statistics), "--sha256-report", str(hashes),
    )
    run(
        sys.executable, "scripts/build_project_parquet.py",
        "--config", str(config_path), "--manifest-dir", str(manifests),
        "--data-root", str(root), "--output-dir", str(root),
    )
    return {
        "archive_identity": archive_identity,
        "extraction_report": str(extraction),
        "validation_report": str(validation),
        "statistics_report": str(statistics),
        "image_sha256_report": str(hashes),
        "train_parquet": config["data"]["frozen_outputs"]["train"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/project_6241.yaml"))
    parser.add_argument("--stage", choices=("metadata", "all"), default="metadata")
    args = parser.parse_args()
    config_path = (args.config if args.config.is_absolute() else PROJECT_ROOT / args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    configure_environment(config)
    evidence = PROJECT_ROOT / "artifacts/runs/E-D10-6K-DATA-001"
    evidence.mkdir(parents=True, exist_ok=True)
    before = disk_snapshot(
        config, scope="before_current_invocation; raw metadata may already be present"
    )
    first_snapshot = evidence / "pre_download_snapshot.json"
    if not first_snapshot.exists():
        first_snapshot.write_text(
            json.dumps(before, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    (evidence / "latest_invocation_snapshot.json").write_text(
        json.dumps(before, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    storage_preflight = None
    if args.stage == "all":
        storage_preflight = pre_download_storage_gate(config)
        if storage_preflight["status"] != "PASS":
            result = {
                "schema_version": 1,
                "stage": args.stage,
                "status": "FAIL_STORAGE_PREFLIGHT",
                "overall_data_gate_status": "FAIL",
                "pre_download_storage": storage_preflight,
            }
            (evidence / "data_gate_summary.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 1

    download(config, include_images=args.stage == "all")
    result = {"metadata": prepare_metadata(config_path, config)}
    if args.stage == "all":
        result["pre_download_storage"] = storage_preflight
        result["images_and_parquet"] = prepare_images(config_path, config)
    after = disk_snapshot(config, scope="after_current_invocation")
    result.update(
        {
            "schema_version": 1,
            "stage": args.stage,
            "status": "PASS" if args.stage == "all" else "METADATA_PASS",
            "overall_data_gate_status": "PASS" if args.stage == "all" else "INCOMPLETE",
            "disk_after": after,
        }
    )
    minimum = int(config["storage"]["minimum_free_gib_after_extraction"])
    if args.stage == "all" and after["free_gib"] < minimum:
        result["status"] = "FAIL_STORAGE_GATE"
        result["overall_data_gate_status"] = "FAIL"
    (evidence / "data_gate_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] in {"PASS", "METADATA_PASS"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
