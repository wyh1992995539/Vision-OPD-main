"""Frozen, reproducible data preparation for the Day 5 benchmark suite."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import yaml
from PIL import Image


def load_dataset(*args: Any, **kwargs: Any) -> Any:
    from datasets import load_dataset as huggingface_load_dataset

    return huggingface_load_dataset(*args, **kwargs)


def snapshot_download(*args: Any, **kwargs: Any) -> Any:
    from huggingface_hub import snapshot_download as huggingface_snapshot_download

    return huggingface_snapshot_download(*args, **kwargs)


PROJECT_BENCHMARKS = ("zoombench", "mmstar", "vstar")
COMMIT_RE = re.compile(r"[0-9a-f]{40}")
OPTION_RE = re.compile(r"(?:^|\s)\(?([A-F])(?:\)|\.)\s", re.MULTILINE)


def _repo_root(config_path: Path) -> Path:
    return config_path.resolve().parent.parent


def _resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _sha256(path: Path) -> str:
    import os

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
        if hasattr(os, "posix_fadvise"):
            os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    return digest.hexdigest()


def _hash_tree(root: Path, output: Path) -> None:
    lines: list[str] = []
    if root.exists():
        for path in sorted(root.rglob("*")):
            if not path.is_file() or "/.cache/" in path.as_posix():
                continue
            lines.append(f"{_sha256(path)}  {path.relative_to(root).as_posix()}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def load_frozen_config(config_path: str | Path) -> tuple[Path, dict[str, Any]]:
    path = Path(config_path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("benchmark config must be a YAML mapping")
    if config.get("protocol", {}).get("status") != "frozen":
        raise ValueError("benchmark config must have protocol.status=frozen")
    for name in PROJECT_BENCHMARKS:
        item = config.get("benchmarks", {}).get(name)
        if not isinstance(item, dict):
            raise ValueError(f"missing benchmark config: {name}")
        revision = str(item.get("dataset_revision", ""))
        if not COMMIT_RE.fullmatch(revision):
            raise ValueError(f"{name}: dataset_revision must be a 40-character commit SHA")
        if not item.get("dataset_repo_id") or not item.get("split"):
            raise ValueError(f"{name}: repo ID and split are required")
    expected_vstar = "craigwu/vstar_bench"
    if config["benchmarks"]["vstar"]["dataset_repo_id"] != expected_vstar:
        raise ValueError(f"vstar source must be the frozen official-linked repo {expected_vstar}")
    return path, config


def parse_benchmarks(raw: str) -> list[str]:
    names = [part.strip() for part in raw.split(",") if part.strip()]
    if not names:
        raise ValueError("--benchmarks must name at least one benchmark")
    unknown = sorted(set(names) - set(PROJECT_BENCHMARKS))
    if unknown:
        raise ValueError(f"unsupported frozen benchmark(s): {', '.join(unknown)}")
    return list(dict.fromkeys(names))


def _first(row: dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        value = row.get(name)
        if value is not None and value != "":
            return value
    return default


def _safe_name(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "sample"
    suffix = hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]
    return f"{stem[:80]}_{suffix}"


def _save_image(value: Any, raw_dir: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        try:
            with Image.open(destination) as image:
                image.verify()
            return
        except (OSError, SyntaxError):
            pass
    if isinstance(value, Image.Image):
        value.convert("RGB").save(destination, format="PNG")
        return
    if isinstance(value, dict):
        if isinstance(value.get("bytes"), (bytes, bytearray)):
            value = value["bytes"]
        elif value.get("path"):
            value = value["path"]
    if isinstance(value, (bytes, bytearray)):
        with Image.open(BytesIO(value)) as image:
            image.convert("RGB").save(destination, format="PNG")
        return
    if isinstance(value, str):
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = raw_dir / candidate
        if not candidate.is_file():
            matches = list(raw_dir.rglob(Path(value).name))
            if len(matches) == 1:
                candidate = matches[0]
        if not candidate.is_file():
            raise ValueError(f"image path cannot be resolved: {value}")
        with Image.open(candidate) as image:
            image.convert("RGB").save(destination, format="PNG")
        return
    raise ValueError(f"unsupported image payload type: {type(value).__name__}")


def _question_format(query: str, answer: str) -> str:
    options = set(OPTION_RE.findall(query.upper()))
    return "multiple_choice" if len(options) >= 2 and answer.strip().upper() in options else "open_question"


def _iter_local_parquet(path: Path, expected_count: int, out_dir: Path) -> Any:
    """Read ZoomBench row groups, omitting image columns for completed checkpoints."""
    import ctypes
    import gc

    import pyarrow as pa
    import pyarrow.parquet as parquet

    if not path.is_file():
        raise FileNotFoundError(f"frozen Parquet source is missing: {path}")
    parquet_file = parquet.ParquetFile(path)
    if parquet_file.metadata.num_rows != expected_count:
        raise ValueError(
            f"{path}: expected {expected_count} Parquet rows, "
            f"got {parquet_file.metadata.num_rows}"
        )

    metadata_columns = ["id", "query", "response", "bbox", "question_type"]
    image_columns = ["image", "crop_image"]
    row_offset = 0
    for row_group_index in range(parquet_file.num_row_groups):
        metadata_rows = parquet_file.read_row_group(
            row_group_index,
            columns=metadata_columns,
            use_threads=False,
        ).to_pylist()
        needs_images = False
        for position, row in enumerate(metadata_rows, start=row_offset):
            source_id = str(_first(row, "id", default=position))
            filename = _safe_name(source_id) + ".png"
            expected_images = (
                out_dir / "images" / "full" / filename,
                out_dir / "images" / "crop" / filename,
            )
            for image_path in expected_images:
                try:
                    with Image.open(image_path) as image:
                        image.verify()
                except (FileNotFoundError, OSError, SyntaxError):
                    needs_images = True
                    break
            if needs_images:
                break

        if needs_images:
            image_rows = parquet_file.read_row_group(
                row_group_index,
                columns=image_columns,
                use_threads=False,
            ).to_pylist()
            for row, image_row in zip(metadata_rows, image_rows, strict=True):
                row.update(image_row)
            del image_rows

        yield from metadata_rows
        row_offset += len(metadata_rows)
        metadata_rows.clear()
        del metadata_rows
        gc.collect()
        pa.default_memory_pool().release_unused()
        try:
            ctypes.CDLL(None).malloc_trim(0)
        except (AttributeError, OSError):
            pass

def _load_local_jsonl(path: Path, expected_count: int) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"frozen JSONL source is missing: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    if len(rows) != expected_count:
        raise ValueError(f"{path}: expected {expected_count} rows, got {len(rows)}")
    return rows



def _append_suffix(query: str, suffix: str) -> str:
    query = query.strip()
    suffix = suffix.strip()
    return query if query.endswith(suffix) else f"{query}\n{suffix}"


def _make_record(
    *,
    benchmark: str,
    source_id: str,
    benchmark_cfg: dict[str, Any],
    query: str,
    response: str,
    image_path: Path,
    category: str,
    question_format: str,
    crop_paths: list[Path] | None = None,
    l2_category: str | None = None,
) -> dict[str, Any]:
    if not source_id.strip() or not query.strip() or not response.strip():
        raise ValueError(f"{benchmark}: empty source ID, question, or answer")
    crops = crop_paths or []
    record: dict[str, Any] = {
        "benchmark": benchmark,
        "sample_uid": f"{benchmark}:source_id:{source_id}",
        "source_id": source_id,
        "source_repo_id": benchmark_cfg["dataset_repo_id"],
        "source_revision": benchmark_cfg["dataset_revision"],
        "source_split": benchmark_cfg["split"],
        "question_format": question_format,
        "category": category or "unknown",
        "images": [str(image_path)],
        "crop_images": [str(path) for path in crops],
        "query": query.strip(),
        "response": response.strip(),
        "image_sha256": _sha256(image_path),
        "crop_image_sha256": [_sha256(path) for path in crops],
        "conversion_version": 1,
    }
    if l2_category is not None:
        record["l2_category"] = l2_category or "unknown"
    return record


def _prepare_zoombench(dataset: Any, cfg: dict[str, Any], raw_dir: Path, out_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for position, raw_row in enumerate(dataset):
        row = dict(raw_row)
        source_id = str(_first(row, "id", "index", "question_id", default=position))
        filename = _safe_name(source_id) + ".png"
        image_path = out_dir / "images" / "full" / filename
        _save_image(_first(row, "image"), raw_dir, image_path)
        crop_paths: list[Path] = []
        crop = _first(row, "crop_image", default=None)
        crop_path = out_dir / "images" / "crop" / filename
        if crop is not None or crop_path.is_file():
            _save_image(crop, raw_dir, crop_path)
            crop_paths.append(crop_path)
        query = str(_first(row, "prompt", "query", "text", default=""))
        response = str(_first(row, "answer", "response", "label", default=""))
        official_question_type = str(_first(row, "question_type", default="")).strip()
        normalized_question_type = official_question_type.casefold().replace("-", "_")
        if normalized_question_type in {"mcq", "multiple_choice"}:
            question_format = "multiple_choice"
        elif normalized_question_type in {"oq", "open", "open_question", "open_ended"}:
            question_format = "open_question"
        else:
            question_format = _question_format(query, response)
        record = _make_record(
            benchmark="zoombench",
            source_id=source_id,
            benchmark_cfg=cfg,
            query=query,
            response=response,
            image_path=image_path,
            category=str(_first(row, "category", "dimension", default="unavailable_official")),
            question_format=question_format,
            crop_paths=crop_paths,
        )
        record["official_question_type"] = official_question_type or "unknown"
        record["bbox"] = row.get("bbox") or []
        records.append(record)
    return records


def _prepare_mmstar(dataset: Any, cfg: dict[str, Any], raw_dir: Path, out_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for position, raw_row in enumerate(dataset):
        row = dict(raw_row)
        source_id = str(_first(row, "index", "id", default=position))
        image_path = out_dir / "images" / (_safe_name(source_id) + ".png")
        _save_image(_first(row, "image"), raw_dir, image_path)
        response = str(_first(row, "answer", "label", default="")).strip().upper()
        if response not in {"A", "B", "C", "D"}:
            raise ValueError(f"mmstar:{source_id}: expected A/B/C/D answer, got {response!r}")
        records.append(_make_record(
            benchmark="mmstar",
            source_id=source_id,
            benchmark_cfg=cfg,
            query=str(_first(row, "question", "query", default="")),
            response=response,
            image_path=image_path,
            category=str(_first(row, "category", default="unknown")),
            l2_category=str(_first(row, "l2_category", default="unknown")),
            question_format="multiple_choice",
        ))
    return records


def _prepare_vstar(dataset: Any, cfg: dict[str, Any], raw_dir: Path, out_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    suffix = str(cfg["prompt_suffix"])
    for position, raw_row in enumerate(dataset):
        row = dict(raw_row)
        source_id = str(_first(row, "question_id", "id", "index", default=position))
        image_path = out_dir / "images" / (_safe_name(source_id) + ".png")
        _save_image(_first(row, "image"), raw_dir, image_path)
        response = str(_first(row, "label", "answer", default="")).strip().upper()
        if response not in {"A", "B", "C", "D"}:
            raise ValueError(f"vstar:{source_id}: expected A/B/C/D answer, got {response!r}")
        query = _append_suffix(str(_first(row, "text", "question", "query", default="")), suffix)
        records.append(_make_record(
            benchmark="vstar",
            source_id=source_id,
            benchmark_cfg=cfg,
            query=query,
            response=response,
            image_path=image_path,
            category=str(_first(row, "category", default="unknown")),
            question_format="multiple_choice",
        ))
    return records


def _validate_records(name: str, records: list[dict[str, Any]], expected_count: int) -> dict[str, Any]:
    if len(records) != expected_count:
        raise ValueError(f"{name}: expected {expected_count} records, got {len(records)}")
    uids = [record["sample_uid"] for record in records]
    if len(set(uids)) != len(uids):
        raise ValueError(f"{name}: duplicate sample_uid detected")
    categories: dict[str, int] = {}
    formats: dict[str, int] = {}
    crop_count = 0
    for record in records:
        if not record["query"] or not record["response"]:
            raise ValueError(f"{name}:{record['sample_uid']}: empty question or answer")
        for image_value in record["images"] + record["crop_images"]:
            image_path = Path(image_value)
            if not image_path.is_file():
                raise ValueError(f"{name}:{record['sample_uid']}: missing image {image_path}")
            with Image.open(image_path) as image:
                image.verify()
        categories[record["category"]] = categories.get(record["category"], 0) + 1
        formats[record["question_format"]] = formats.get(record["question_format"], 0) + 1
        crop_count += len(record["crop_images"])
    return {
        "status": "pass",
        "count": len(records),
        "unique_sample_uids": len(set(uids)),
        "categories": dict(sorted(categories.items())),
        "question_formats": dict(sorted(formats.items())),
        "crop_image_count": crop_count,
    }


def _load_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_frozen_preparation(
    config_path: str | Path,
    benchmarks: str,
    data_root_override: str | None = None,
    force: bool = False,
) -> None:
    config_path, config = load_frozen_config(config_path)
    selected = parse_benchmarks(benchmarks)
    repo_root = _repo_root(config_path)
    data_root = Path(data_root_override or config["paths"]["data_root"]).resolve()
    raw_root = data_root / "raw"
    converted_root = data_root / "converted"
    cache_dir = Path(config["paths"]["hf_cache"]).resolve()
    manifest_path = _resolve_path(repo_root, config["paths"]["dataset_manifest"])
    run_root = _resolve_path(repo_root, config["paths"]["run_root"])
    validation_path = run_root / "data_validation.json"

    manifest = _load_json(manifest_path, {"schema_version": 1, "benchmarks": {}})
    validations = _load_json(validation_path, {"schema_version": 1, "benchmarks": {}})

    for name in selected:
        benchmark_cfg = config["benchmarks"][name]
        raw_dir = raw_root / name
        output_dir = converted_root / name
        output_json = output_dir / f"{name}.json"
        if output_json.exists() and not force:
            raise FileExistsError(f"refusing to overwrite {output_json}; validate it or pass --force")

        if name == "zoombench":
            source_file = raw_dir / str(benchmark_cfg.get("source_file", ""))
            revision_marker = (
                raw_dir / ".cache" / "huggingface" / "trees"
                / f"{benchmark_cfg['dataset_revision']}.json"
            )
            if source_file.is_file() and revision_marker.is_file():
                print(f"Reusing frozen local ZoomBench Parquet: {source_file}")
            else:
                print(f"Downloading frozen {name} revision {benchmark_cfg['dataset_revision']} ...")
                snapshot_download(
                    repo_id=benchmark_cfg["dataset_repo_id"],
                    repo_type="dataset",
                    revision=benchmark_cfg["dataset_revision"],
                    local_dir=str(raw_dir),
                )
            dataset = _iter_local_parquet(
                source_file,
                int(benchmark_cfg["expected_sample_count"]),
                output_dir,
            )
            records = _prepare_zoombench(dataset, benchmark_cfg, raw_dir, output_dir)
        elif name == "vstar":
            source_file = raw_dir / "test_questions.jsonl"
            revision_marker = (
                raw_dir / ".cache" / "huggingface" / "trees"
                / f"{benchmark_cfg['dataset_revision']}.json"
            )
            if source_file.is_file() and revision_marker.is_file():
                print(f"Reusing frozen local VStar JSONL: {source_file}")
            else:
                print(f"Downloading frozen {name} revision {benchmark_cfg['dataset_revision']} ...")
                snapshot_download(
                    repo_id=benchmark_cfg["dataset_repo_id"],
                    repo_type="dataset",
                    revision=benchmark_cfg["dataset_revision"],
                    local_dir=str(raw_dir),
                )
            if not source_file.is_file():
                dataset = load_dataset(
                    benchmark_cfg["dataset_repo_id"],
                    split=benchmark_cfg["split"],
                    revision=benchmark_cfg["dataset_revision"],
                    cache_dir=str(cache_dir),
                )
            else:
                dataset = _load_local_jsonl(
                    source_file,
                    int(benchmark_cfg["expected_sample_count"]),
                )
            records = _prepare_vstar(dataset, benchmark_cfg, raw_dir, output_dir)
        else:
            print(f"Downloading frozen {name} revision {benchmark_cfg['dataset_revision']} ...")
            snapshot_download(
                repo_id=benchmark_cfg["dataset_repo_id"],
                repo_type="dataset",
                revision=benchmark_cfg["dataset_revision"],
                local_dir=str(raw_dir),
            )
            dataset = load_dataset(
                benchmark_cfg["dataset_repo_id"],
                split=benchmark_cfg["split"],
                revision=benchmark_cfg["dataset_revision"],
                cache_dir=str(cache_dir),
            )
            records = _prepare_mmstar(dataset, benchmark_cfg, raw_dir, output_dir)

        validation = _validate_records(name, records, int(benchmark_cfg["expected_sample_count"]))
        _write_json(output_json, records)
        manifest["benchmarks"][name] = {
            "repo_id": benchmark_cfg["dataset_repo_id"],
            "requested_revision": benchmark_cfg["dataset_revision"],
            "split": benchmark_cfg["split"],
            "expected_count": benchmark_cfg["expected_sample_count"],
            "actual_count": len(records),
            "converted_json": str(output_json),
        }
        validations["benchmarks"][name] = validation
        print(f"Prepared {name}: records={len(records)} output={output_json}")

    timestamp = datetime.now(timezone.utc).isoformat()
    manifest["updated_at_utc"] = timestamp
    validations["updated_at_utc"] = timestamp
    _write_json(manifest_path, manifest)
    _write_json(validation_path, validations)
    _hash_tree(raw_root, _resolve_path(repo_root, config["paths"]["raw_hash_manifest"]))
    _hash_tree(converted_root, _resolve_path(repo_root, config["paths"]["converted_hash_manifest"]))
