#!/usr/bin/env python3
"""Convert the frozen Vision-OPD 6K Parquet into an auditable GRPO dataset.

The conversion is deliberately non-destructive: it preserves the original
prompt, student image, ground truth, and source identifiers, while removing
the Vision-OPD teacher crop and duplicate answer fields.  Runtime rollouts,
rewards, group UIDs, and advantages are not materialized in the dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import sys
import tempfile
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path("/root/autodl-tmp/data/vision_opd_6241/train_6241.parquet")
DEFAULT_OUTPUT = Path("/root/autodl-tmp/data/vision_opd_6241/grpo_train_6241.parquet")
DEFAULT_RUN_DIR = ROOT / "artifacts/runs/E-D17-GRPO-DATA-001"
SOURCE_QA = ROOT / "artifacts/data/vision_opd_6241/vision_opd_6241_data_qa.json"
DATA_SOURCE = "vision_opd_mcq_grpo_v1"
REWARD_ROUTE = "vision_opd_mcq_v1"
VALID_OPTIONS = ("A", "B", "C", "D")
EXPECTED_SOURCE_DATA = "zwz_rl_vqa_bbox_teacher"
REQUIRED_SOURCE_COLUMNS = {
    "data_source",
    "prompt",
    "images",
    "bbox_images",
    "ability",
    "reward_model",
    "extra_info",
}
OUTPUT_COLUMNS = {
    "data_source",
    "prompt",
    "images",
    "ability",
    "reward_model",
    "extra_info",
}
OPTION_PATTERN = re.compile(r"(?m)^\s*([A-D])\.\s+\S.*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the frozen 6,241-row Vision-OPD Parquet for GRPO."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--expected-rows", type=int, default=6241)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace existing conversion outputs.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_parquet(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        pq.write_table(table, temporary, compression="zstd")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def require_mapping(value: Any, *, field: str, row_index: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"source row {row_index}: {field} must be a mapping")
    return value


def require_string(value: Any, *, field: str, row_index: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"source row {row_index}: {field} must be a non-empty string")
    return value.strip()


def convert_row(
    row: dict[str, Any], row_index: int, *, expected_image_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    if row.get("data_source") != EXPECTED_SOURCE_DATA:
        raise ValueError(
            f"source row {row_index}: unexpected data_source={row.get('data_source')!r}"
        )

    prompt = row.get("prompt")
    if not isinstance(prompt, list) or len(prompt) != 1:
        raise ValueError(f"source row {row_index}: prompt must contain exactly one message")
    message = require_mapping(prompt[0], field="prompt[0]", row_index=row_index)
    if message.get("role") != "user":
        raise ValueError(f"source row {row_index}: prompt must be a user-only message")
    content = require_string(message.get("content"), field="prompt.content", row_index=row_index)
    if content.count("<image>") != 1:
        raise ValueError(f"source row {row_index}: prompt must contain exactly one <image>")
    option_labels = OPTION_PATTERN.findall(content)
    if option_labels != list(VALID_OPTIONS):
        raise ValueError(
            f"source row {row_index}: expected ordered A-D options, got {option_labels}"
        )

    images = row.get("images")
    if not isinstance(images, list) or len(images) != 1:
        raise ValueError(f"source row {row_index}: images must contain exactly one item")
    image = require_mapping(images[0], field="images[0]", row_index=row_index)
    image_path = Path(require_string(image.get("path"), field="images[0].path", row_index=row_index))
    resolved_image = image_path.resolve()
    if not resolved_image.is_relative_to(expected_image_root):
        raise ValueError(f"source row {row_index}: student image escapes expected image root")
    if not resolved_image.is_file():
        raise FileNotFoundError(f"source row {row_index}: student image not found: {resolved_image}")

    reward_model = require_mapping(
        row.get("reward_model"), field="reward_model", row_index=row_index
    )
    ground_truth = require_string(
        reward_model.get("ground_truth"), field="reward_model.ground_truth", row_index=row_index
    ).upper()
    if ground_truth not in VALID_OPTIONS:
        raise ValueError(
            f"source row {row_index}: ground truth must be one of {VALID_OPTIONS}, got {ground_truth!r}"
        )

    source_extra = require_mapping(row.get("extra_info"), field="extra_info", row_index=row_index)
    provenance = require_mapping(
        source_extra.get("provenance"), field="extra_info.provenance", row_index=row_index
    )
    sample_id = require_string(
        provenance.get("sample_id"), field="extra_info.provenance.sample_id", row_index=row_index
    )
    question_type = require_string(
        provenance.get("question_type"),
        field="extra_info.provenance.question_type",
        row_index=row_index,
    )
    if question_type != "multiple_choice":
        raise ValueError(
            f"source row {row_index}: unsupported question_type={question_type!r}"
        )
    source_id = require_string(
        provenance.get("source_id"), field="extra_info.provenance.source_id", row_index=row_index
    )
    group_id = require_string(
        provenance.get("group_id"), field="extra_info.provenance.group_id", row_index=row_index
    )
    source_row = provenance.get("source_row")
    if not isinstance(source_row, int) or isinstance(source_row, bool) or source_row < 0:
        raise ValueError(f"source row {row_index}: provenance.source_row must be a non-negative integer")

    converted = {
        "data_source": DATA_SOURCE,
        "prompt": prompt,
        "images": images,
        "ability": row.get("ability", "visual_question_answering"),
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": {
            "index": row_index,
            "sample_id": sample_id,
            "question_type": question_type,
            "reward_route": REWARD_ROUTE,
            "valid_options": list(VALID_OPTIONS),
            "source_id": source_id,
            "source_row": source_row,
            "group_id": group_id,
        },
    }
    route = {
        "index": row_index,
        "sample_id": sample_id,
        "data_source": DATA_SOURCE,
        "reward_route": REWARD_ROUTE,
        "normalized_gold": ground_truth,
        "valid_options": list(VALID_OPTIONS),
        "data_scorable": True,
        "reason": "four_ordered_options_and_gold_in_valid_options",
    }
    return converted, route


def convert_grpo_dataset(
    source: Path,
    output: Path,
    run_dir: Path,
    *,
    expected_rows: int = 6241,
    source_qa: Path = SOURCE_QA,
    overwrite: bool = False,
    command: str | None = None,
) -> dict[str, Any]:
    source = source.resolve()
    output = output.resolve()
    run_dir = run_dir.resolve()
    source_qa = source_qa.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"source Parquet not found: {source}")
    if source == output:
        raise ValueError("source and output must be different paths")
    if expected_rows <= 0:
        raise ValueError("expected_rows must be positive")

    report_path = run_dir / "conversion_report.json"
    routes_path = run_dir / "reward_routes.jsonl"
    binding_path = run_dir / "source_binding.json"
    hash_path = run_dir / "grpo_parquet_sha256.txt"
    command_path = run_dir / "command.txt"
    targets = [output, report_path, routes_path, binding_path, hash_path, command_path]
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        formatted = "\n".join(f"  - {path}" for path in existing)
        raise FileExistsError(f"conversion output already exists; use --overwrite:\n{formatted}")

    source_sha256 = sha256_file(source)
    source_table = pq.read_table(source)
    source_columns = set(source_table.column_names)
    missing_columns = sorted(REQUIRED_SOURCE_COLUMNS - source_columns)
    if missing_columns:
        raise ValueError(f"source Parquet is missing required columns: {missing_columns}")
    if source_table.num_rows != expected_rows:
        raise ValueError(
            f"source row count mismatch: expected {expected_rows}, got {source_table.num_rows}"
        )

    image_root = (source.parent / "images").resolve()
    converted_rows: list[dict[str, Any]] = []
    route_rows: list[dict[str, Any]] = []
    seen_sample_ids: set[str] = set()
    gold_distribution = {option: 0 for option in VALID_OPTIONS}
    for row_index, row in enumerate(source_table.to_pylist()):
        converted, route = convert_row(row, row_index, expected_image_root=image_root)
        sample_id = converted["extra_info"]["sample_id"]
        if sample_id in seen_sample_ids:
            raise ValueError(f"duplicate sample_id in source Parquet: {sample_id}")
        seen_sample_ids.add(sample_id)
        gold_distribution[converted["reward_model"]["ground_truth"]] += 1
        converted_rows.append(converted)
        route_rows.append(route)

    output_table = pa.Table.from_pylist(converted_rows)
    metadata = dict(output_table.schema.metadata or {})
    metadata.update(
        {
            b"schema_version": b"1",
            b"dataset_role": b"vision_opd_grpo_train",
            b"source_parquet_sha256": source_sha256.encode(),
            b"reward_route": REWARD_ROUTE.encode(),
        }
    )
    output_table = output_table.replace_schema_metadata(metadata)
    atomic_write_parquet(output, output_table)

    readback = pq.read_table(output)
    if readback.num_rows != expected_rows:
        raise AssertionError(
            f"output read-back row mismatch: expected {expected_rows}, got {readback.num_rows}"
        )
    if set(readback.column_names) != OUTPUT_COLUMNS:
        raise AssertionError(
            f"output schema mismatch: expected {sorted(OUTPUT_COLUMNS)}, got {readback.column_names}"
        )
    if "bbox_images" in readback.column_names:
        raise AssertionError("GRPO output must not contain bbox_images")
    readback_rows = readback.to_pylist()
    if any(row["data_source"] != DATA_SOURCE for row in readback_rows):
        raise AssertionError("GRPO output contains an unexpected data_source")
    if any(row["extra_info"]["index"] != index for index, row in enumerate(readback_rows)):
        raise AssertionError("GRPO output indices are not unique and sequential")

    output_sha256 = sha256_file(output)
    qa_binding: dict[str, Any]
    if source_qa.is_file():
        qa_payload = json.loads(source_qa.read_text(encoding="utf-8"))
        qa_binding = {
            "path": source_qa.as_posix(),
            "sha256": sha256_file(source_qa),
            "status": qa_payload.get("status"),
            "record_count": qa_payload.get("record_count"),
            "images_checked": qa_payload.get("images_checked"),
            "issue_count": qa_payload.get("issue_count"),
        }
    else:
        qa_binding = {"path": source_qa.as_posix(), "status": "MISSING"}

    source_binding = {
        "schema_version": 1,
        "source_parquet": source.as_posix(),
        "source_parquet_sha256": source_sha256,
        "source_rows": source_table.num_rows,
        "source_columns": source_table.column_names,
        "source_image_qa": qa_binding,
        "prompt_and_student_image_policy": "copied_without_modification",
        "teacher_crop_policy": "excluded_from_grpo_output",
    }
    report = {
        "schema_version": 1,
        "experiment_id": "E-D17-GRPO-DATA-001",
        "status": "PASS",
        "scope": "grpo_data_conversion_only",
        "source": source.as_posix(),
        "source_sha256": source_sha256,
        "output": output.as_posix(),
        "output_sha256": output_sha256,
        "output_bytes": output.stat().st_size,
        "source_rows": source_table.num_rows,
        "output_rows": readback.num_rows,
        "unique_sample_ids": len(seen_sample_ids),
        "gold_distribution": gold_distribution,
        "data_source": DATA_SOURCE,
        "reward_route": REWARD_ROUTE,
        "checks": {
            "source_hash_recorded": True,
            "source_row_count_exact": source_table.num_rows == expected_rows,
            "output_row_count_exact": readback.num_rows == expected_rows,
            "unique_sample_ids": len(seen_sample_ids) == expected_rows,
            "prompt_preserved": True,
            "student_image_preserved": True,
            "all_student_image_paths_exist": True,
            "ordered_a_to_d_options": True,
            "ground_truth_in_valid_options": True,
            "data_scorable_rows": len(route_rows) == expected_rows,
            "bbox_images_excluded": "bbox_images" not in readback.column_names,
            "duplicate_answer_field_excluded": True,
            "runtime_rollout_fields_not_materialized": True,
            "parquet_readback_schema_and_rows": True,
        },
        "runtime_contract": {
            "parquet_rows": 6241,
            "rollout_n_is_not_materialized": True,
            "trainer_must_generate_runtime_uid": True,
            "trainer_must_apply_native_drop_last": True,
            "expected_effective_prompts_at_batch_8": 6240,
            "expected_dropped_prompts_at_batch_8": 1,
        },
        "pilot_authorized": False,
        "pending_gates": [
            "implement_and_test_vision_opd_mcq_grpo_v1_reward",
            "load_output_through_local_RLHFDataset",
            "freeze_grpo_training_config_and_guarded_launcher",
        ],
    }

    route_text = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in route_rows
    )
    atomic_write_text(routes_path, route_text)
    atomic_write_text(binding_path, json.dumps(source_binding, ensure_ascii=False, indent=2) + "\n")
    atomic_write_text(hash_path, f"{output_sha256}  {output.as_posix()}\n")
    atomic_write_text(report_path, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    atomic_write_text(command_path, (command or "programmatic invocation") + "\n")
    return report


def main() -> None:
    args = parse_args()
    command = shlex.join([sys.executable, *sys.argv])
    report = convert_grpo_dataset(
        args.source,
        args.output,
        args.run_dir,
        expected_rows=args.expected_rows,
        overwrite=args.overwrite,
        command=command,
    )
    print("PASS: GRPO data conversion completed")
    print(f"  rows: {report['output_rows']}")
    print(f"  output: {report['output']}")
    print(f"  sha256: {report['output_sha256']}")
    print(f"  evidence: {args.run_dir.resolve()}")
    print("  pilot_authorized: false (Reward and trainer gates remain)")


if __name__ == "__main__":
    main()
