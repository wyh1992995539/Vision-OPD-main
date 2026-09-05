#!/usr/bin/env python3
"""Build nested, auditable 16/64-row Pilot subsets for the 6,241-row run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq


SEED = 42
ALGORITHM = (
    "round_robin_unique(student_tail_new,teacher_tail_new,"
    "student_tail_historical,teacher_tail_historical,"
    "sha256_new,sha256_historical)"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_key(sample_id: str, seed: int = SEED) -> str:
    return hashlib.sha256(f"{seed}|{sample_id}".encode()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
    return rows


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_parquet(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        pq.write_table(table, temporary, compression="snappy")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def table_sample_ids(table: pa.Table) -> list[str]:
    result: list[str] = []
    for index, row in enumerate(table.select(["extra_info"]).to_pylist()):
        try:
            sample_id = str(row["extra_info"]["provenance"]["sample_id"]).strip()
        except (KeyError, TypeError) as exc:
            raise ValueError(f"source row {index}: missing sample_id") from exc
        if not sample_id:
            raise ValueError(f"source row {index}: empty sample_id")
        result.append(sample_id)
    if len(set(result)) != len(result):
        raise ValueError("source Parquet contains duplicate sample IDs")
    return result


def prompt_lengths(path: Path) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for row in read_jsonl(path):
        sample_id = str(row.get("sample_id", ""))
        view = str(row.get("view", ""))
        if view not in {"student", "teacher"}:
            raise ValueError(f"unexpected prompt audit view: {view!r}")
        if view in result.setdefault(sample_id, {}):
            raise ValueError(f"duplicate prompt audit row: {sample_id}/{view}")
        result[sample_id][view] = int(row["total_tokens"])
    incomplete = [key for key, value in result.items() if set(value) != {"student", "teacher"}]
    if incomplete:
        raise ValueError(f"prompt audit is missing a view for: {incomplete[:5]}")
    return result


def historical_ids(paths: Iterable[Path]) -> set[str]:
    result: set[str] = set()
    for path in paths:
        for row in read_jsonl(path):
            sample_id = str(row.get("sample_id", "")).strip()
            if not sample_id:
                raise ValueError(f"historical manifest contains an empty sample ID: {path}")
            result.add(sample_id)
    return result


def select_samples(
    sample_ids: list[str],
    lengths: dict[str, dict[str, int]],
    historical: set[str],
    *,
    count: int,
    seed: int = SEED,
) -> list[tuple[str, str]]:
    if count <= 0 or count > len(sample_ids):
        raise ValueError("count must be within the source dataset")
    source = set(sample_ids)
    if set(lengths) != source:
        missing = sorted(source - set(lengths))
        extra = sorted(set(lengths) - source)
        raise ValueError(f"prompt audit/source mismatch: missing={missing[:5]} extra={extra[:5]}")
    if not historical <= source:
        raise ValueError("historical IDs are not a subset of the source dataset")
    new = source - historical
    if not new or not historical:
        raise ValueError("both historical and newly added cohorts are required")

    def tail(cohort: set[str], view: str) -> list[str]:
        return sorted(cohort, key=lambda item: (-lengths[item][view], item))

    def hashed(cohort: set[str]) -> list[str]:
        return sorted(cohort, key=lambda item: (stable_key(item, seed), item))

    categories = [
        ("student_tail_new", tail(new, "student")),
        ("teacher_tail_new", tail(new, "teacher")),
        ("student_tail_historical", tail(historical, "student")),
        ("teacher_tail_historical", tail(historical, "teacher")),
        ("sha256_new", hashed(new)),
        ("sha256_historical", hashed(historical)),
    ]
    positions = [0] * len(categories)
    selected: list[tuple[str, str]] = []
    seen: set[str] = set()
    while len(selected) < count:
        progressed = False
        for category_index, (reason, candidates) in enumerate(categories):
            while positions[category_index] < len(candidates):
                candidate = candidates[positions[category_index]]
                positions[category_index] += 1
                if candidate in seen:
                    continue
                seen.add(candidate)
                selected.append((candidate, reason))
                progressed = True
                break
            if len(selected) == count:
                break
        if not progressed:
            raise ValueError(f"selection exhausted at {len(selected)} rows")
    return selected


def build_subset(
    source_path: Path,
    prompt_lengths_path: Path,
    historical_paths: list[Path],
    output_path: Path,
    manifest_path: Path,
    *,
    experiment_id: str,
    count: int,
    seed: int = SEED,
    overwrite: bool = False,
) -> dict[str, Any]:
    paths = [source_path, prompt_lengths_path, *historical_paths]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required Pilot inputs are missing: {missing}")
    for path in (output_path, manifest_path):
        if path.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite existing output: {path}")

    table = pq.read_table(source_path)
    ids = table_sample_ids(table)
    lengths = prompt_lengths(prompt_lengths_path)
    historical = historical_ids(historical_paths)
    selection = select_samples(ids, lengths, historical, count=count, seed=seed)
    source_index = {sample_id: index for index, sample_id in enumerate(ids)}
    indices = [source_index[sample_id] for sample_id, _reason in selection]
    subset = table.take(pa.array(indices, type=pa.int64()))
    metadata = dict(subset.schema.metadata or {})
    metadata.update(
        {
            b"vision_opd_pilot_algorithm": ALGORITHM.encode(),
            b"vision_opd_pilot_seed": str(seed).encode(),
            b"vision_opd_pilot_count": str(count).encode(),
        }
    )
    atomic_parquet(output_path, subset.replace_schema_metadata(metadata))

    samples = []
    for selection_index, (sample_id, reason) in enumerate(selection):
        row_index = source_index[sample_id]
        provenance = table.slice(row_index, 1).select(["extra_info"]).to_pylist()[0][
            "extra_info"
        ]["provenance"]
        samples.append(
            {
                "selection_index": selection_index,
                "source_row_index": row_index,
                "sample_id": sample_id,
                "source_id": str(provenance.get("source_id", "")),
                "source_row": int(provenance.get("source_row", -1)),
                "group_id": str(provenance.get("group_id", "")),
                "cohort": "historical_1216" if sample_id in historical else "new_5025",
                "selection_reason": reason,
                "student_total_tokens": lengths[sample_id]["student"],
                "teacher_total_tokens": lengths[sample_id]["teacher"],
                "selection_key": stable_key(sample_id, seed),
            }
        )

    manifest = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "purpose": f"nested_{count}_row_1024_token_training_pilot",
        "selection": {
            "algorithm": ALGORITHM,
            "seed": seed,
            "requested_rows": count,
            "selected_rows": len(samples),
            "preserve_ranked_order": True,
            "nested_contract": "the 16-row selection is the prefix of the 64-row selection",
        },
        "source": {
            "path": str(source_path.resolve()),
            "rows": table.num_rows,
            "sha256": sha256_file(source_path),
            "historical_rows": len(historical),
            "new_rows": table.num_rows - len(historical),
        },
        "prompt_length_audit": {
            "path": str(prompt_lengths_path.resolve()),
            "sha256": sha256_file(prompt_lengths_path),
        },
        "historical_manifests": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in historical_paths
        ],
        "output": {
            "path": str(output_path.resolve()),
            "rows": subset.num_rows,
            "sha256": sha256_file(output_path),
        },
        "samples": samples,
    }
    atomic_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path,
        default=Path("/root/autodl-tmp/data/vision_opd_6241/train_6241.parquet"),
    )
    parser.add_argument(
        "--prompt-lengths", type=Path,
        default=root / "artifacts/runs/E-D11-6K-GATE-001/prompt_length/prompt_lengths.jsonl",
    )
    parser.add_argument("--count", type=int, choices=(16, 64), required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    historical = [
        Path("/root/autodl-tmp/data/vision_opd_1024/manifests/train_1024.jsonl"),
        Path("/root/autodl-tmp/data/vision_opd_1024/manifests/eval_128.jsonl"),
        Path("/root/autodl-tmp/data/vision_opd_1024/manifests/retention_64.jsonl"),
    ]
    manifest = build_subset(
        args.source.resolve(), args.prompt_lengths.resolve(), historical,
        args.output.resolve(), args.manifest.resolve(),
        experiment_id=args.experiment_id, count=args.count, seed=args.seed,
        overwrite=args.overwrite,
    )
    print("VOPD_6241_PILOT_SUBSET=PASS")
    print(f"ROWS={manifest['output']['rows']}")
    print(f"SHA256={manifest['output']['sha256']}")
    print(f"MANIFEST={args.manifest.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
