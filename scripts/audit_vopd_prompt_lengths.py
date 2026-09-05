#!/usr/bin/env python3
"""Audit Vision-OPD multimodal prompt lengths with the training processor."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

import yaml


VIEW_TO_COLUMN = {"student": "images", "teacher": "bbox_images"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/vopd_1024.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--views", nargs="+", choices=sorted(VIEW_TO_COLUMN), default=sorted(VIEW_TO_COLUMN))
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N rows for a smoke check.")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument(
        "--parquet-batch-size",
        type=int,
        default=8,
        help="Rows converted to Python at once; keep small in memory-constrained cgroups.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(project_root: Path, value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else project_root / path).resolve()


def build_messages(row: dict[str, Any], image_column: str) -> list[dict[str, Any]]:
    """Mirror RLHFDataset._build_messages for a selected image view."""
    messages = copy.deepcopy(row["prompt"])
    images = copy.deepcopy(row.get(image_column) or [])
    image_offset = 0

    for message in messages:
        content = message.get("content")
        if not isinstance(content, str):
            continue
        content_list: list[dict[str, Any]] = []
        for segment in filter(None, re.split("(<image>)", content)):
            if segment == "<image>":
                if image_offset >= len(images):
                    raise ValueError(f"more <image> placeholders than {image_column} entries")
                image = dict(images[image_offset])
                if "image" not in image and "path" in image:
                    image["image"] = image["path"]
                content_list.append({"type": "image", **image})
                image_offset += 1
            else:
                content_list.append({"type": "text", "text": segment})
        message["content"] = content_list

    if image_offset != len(images):
        raise ValueError(f"used {image_offset} image placeholders but found {len(images)} {image_column} entries")
    return messages


def sample_id(row: dict[str, Any], row_index: int) -> str:
    extra_info = row.get("extra_info") or {}
    provenance = extra_info.get("provenance") or {}
    return str(provenance.get("sample_id") or f"row_{row_index:04d}")


def nearest_rank(values: list[int], quantile: float) -> int:
    if not values:
        raise ValueError("cannot calculate a quantile from an empty list")
    ordered = sorted(values)
    rank = max(1, math.ceil(quantile * len(ordered)))
    return ordered[rank - 1]


def distribution(values: list[int]) -> dict[str, int]:
    return {
        "p50": nearest_rank(values, 0.50),
        "p95": nearest_rank(values, 0.95),
        "p99": nearest_rank(values, 0.99),
        "max": max(values),
    }


def count_tokens(processor: Any, messages: list[dict[str, Any]]) -> tuple[int, int, int, str]:
    from qwen_vl_utils import process_vision_info

    images, videos = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_metadata=True,
    )
    if videos is not None:
        videos, video_metadatas = zip(*videos, strict=False)
        videos, video_metadatas = list(videos), list(video_metadatas)
    else:
        video_metadatas = None

    rendered = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    model_inputs = processor(
        text=[rendered],
        images=images,
        videos=videos,
        video_metadatas=video_metadatas,
        return_tensors="pt",
        do_sample_frames=False,
    )
    input_ids = model_inputs["input_ids"].reshape(-1)
    total_tokens = int(input_ids.numel())

    image_token_id = getattr(processor, "image_token_id", None)
    if image_token_id is not None:
        image_tokens = int((input_ids == int(image_token_id)).sum().item())
        method = "input_ids_equals_processor_image_token_id"
    elif "mm_token_type_ids" in model_inputs:
        token_types = model_inputs["mm_token_type_ids"].reshape(-1)
        image_tokens = int((token_types == 1).sum().item())
        method = "mm_token_type_ids_equals_1"
    else:
        raise RuntimeError("processor exposes neither image_token_id nor mm_token_type_ids")

    # This bucket includes natural-language, template, and vision boundary/control tokens.
    text_tokens = total_tokens - image_tokens
    if text_tokens < 0 or image_tokens <= 0:
        raise RuntimeError(
            f"invalid token split: text={text_tokens}, image={image_tokens}, total={total_tokens}"
        )
    return text_tokens, image_tokens, total_tokens, method


def build_summary(
    *,
    records: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    views: list[str],
    row_count: int,
    expected_rows: int,
    limit: int | None,
    max_prompt_length: int,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    view_summaries: dict[str, Any] = {}
    for view in views:
        selected = [record for record in records if record["view"] == view]
        view_summaries[view] = {
            "processed": len(selected),
            "text_tokens": distribution([record["text_tokens"] for record in selected]) if selected else None,
            "image_tokens": distribution([record["image_tokens"] for record in selected]) if selected else None,
            "total_tokens": distribution([record["total_tokens"] for record in selected]) if selected else None,
            "overlength_count": sum(record["overlength"] for record in selected),
        }

    expected_processed = (min(row_count, limit) if limit is not None else row_count) * len(views)
    full_audit = limit is None
    gate_checks = {
        "full_expected_rows": full_audit and row_count == expected_rows,
        "all_requested_views_processed": len(records) == expected_processed,
        "processing_errors_zero": not errors,
        "overlength_count_zero": all(view_summaries[view]["overlength_count"] == 0 for view in views),
        "silent_truncation_disabled": True,
    }
    smoke_checks = all(passed for name, passed in gate_checks.items() if name != "full_expected_rows")
    status = "PASS" if all(gate_checks.values()) else ("SMOKE_PASS" if not full_audit and smoke_checks else "FAIL")
    return {
        "schema_version": 1,
        "experiment_id": metadata["experiment_id"],
        "status": status,
        "gpu_used": False,
        "processor_contract": {
            "add_generation_prompt": True,
            "truncation_argument_passed": False,
            "max_length_argument_passed": False,
            "image_processing": "qwen_vl_utils.process_vision_info",
            "text_tokens_definition": "total_tokens minus expanded image_token_id tokens; includes template/control tokens",
            "quantile_method": "nearest_rank",
        },
        "input": metadata,
        "row_count": row_count,
        "expected_rows": expected_rows,
        "limit": limit,
        "max_prompt_length_candidate": max_prompt_length,
        "views": view_summaries,
        "error_count": len(errors),
        "errors": errors,
        "gate_checks": gate_checks,
    }


def write_report(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Vision-OPD Prompt Length Audit",
        "",
        f"- Status: **{summary['status']}**",
        f"- Rows: **{summary['row_count']}**",
        f"- Candidate limit: **{summary['max_prompt_length_candidate']}**",
        f"- Processing errors: **{summary['error_count']}**",
        "- GPU used: **No**",
        "",
        "| View | Metric | P50 | P95 | P99 | Max | Overlength |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for view, values in summary["views"].items():
        for metric in ("text_tokens", "image_tokens", "total_tokens"):
            stats = values[metric]
            if stats is None:
                continue
            lines.append(
                f"| {view} | {metric} | {stats['p50']} | {stats['p95']} | "
                f"{stats['p99']} | {stats['max']} | {values['overlength_count']} |"
            )
    lines.extend(["", "The processor was called without truncation or max-length arguments.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    import pyarrow.parquet as pq
    from transformers import AutoProcessor

    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config_path = resolve(project_root, args.config)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_path = resolve(project_root, cfg["paths"]["model"])
    train_file = resolve(project_root, cfg["paths"]["train_file"])
    chat_template = resolve(project_root, cfg["paths"]["chat_template"])
    output_dir = resolve(project_root, args.output_dir or f"{cfg['paths']['output_dir']}/preflight")
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_file = pq.ParquetFile(train_file)
    row_count = parquet_file.metadata.num_rows
    expected_rows = int(cfg["data"]["expected_train_rows"])
    if args.parquet_batch_size <= 0:
        raise ValueError("--parquet-batch-size must be positive")
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        rows_to_process = min(row_count, args.limit)
    else:
        rows_to_process = row_count

    def iter_rows():
        emitted = 0
        for batch in parquet_file.iter_batches(batch_size=args.parquet_batch_size):
            for row in batch.to_pylist():
                if emitted >= rows_to_process:
                    return
                yield row
                emitted += 1

    processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True, local_files_only=True)
    processor.chat_template = chat_template.read_text(encoding="utf-8")
    max_prompt_length = int(cfg["data"]["max_prompt_length"])
    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    token_count_method: str | None = None
    started = time.monotonic()
    partial_path = output_dir / "prompt_lengths.partial.jsonl"

    with partial_path.open("w", encoding="utf-8") as output:
        for row_index, row in enumerate(iter_rows()):
            sid = sample_id(row, row_index)
            for view in args.views:
                try:
                    messages = build_messages(row, VIEW_TO_COLUMN[view])
                    text_tokens, image_tokens, total_tokens, method = count_tokens(processor, messages)
                    token_count_method = token_count_method or method
                    if token_count_method != method:
                        raise RuntimeError(f"token count method changed from {token_count_method} to {method}")
                    record = {
                        "row_index": row_index,
                        "sample_id": sid,
                        "view": view,
                        "image_column": VIEW_TO_COLUMN[view],
                        "text_tokens": text_tokens,
                        "image_tokens": image_tokens,
                        "total_tokens": total_tokens,
                        "overlength": total_tokens > max_prompt_length,
                    }
                    records.append(record)
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    output.flush()
                except Exception as exc:  # continue to produce a complete failure manifest
                    errors.append(
                        {"row_index": row_index, "sample_id": sid, "view": view, "error": repr(exc)}
                    )
            completed = row_index + 1
            if args.progress_every > 0 and (completed % args.progress_every == 0 or completed == rows_to_process):
                elapsed = time.monotonic() - started
                print(f"processed_rows={completed}/{rows_to_process} elapsed_seconds={elapsed:.1f}", flush=True)

    final_records_path = output_dir / "prompt_lengths.jsonl"
    partial_path.replace(final_records_path)
    metadata = {
        "experiment_id": str(cfg["experiment"]["id"]),
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "model_path": str(model_path),
        "processor_class": type(processor).__name__,
        "processor_name_or_path": str(getattr(processor, "name_or_path", model_path)),
        "chat_template": str(chat_template),
        "chat_template_sha256": sha256(chat_template),
        "train_file": str(train_file),
        "train_file_sha256": sha256(train_file),
        "token_count_method": token_count_method,
    }
    summary = build_summary(
        records=records,
        errors=errors,
        views=args.views,
        row_count=row_count,
        expected_rows=expected_rows,
        limit=args.limit,
        max_prompt_length=max_prompt_length,
        metadata=metadata,
    )
    summary["elapsed_seconds"] = round(time.monotonic() - started, 3)
    summary_path = output_dir / "prompt_length_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(summary, output_dir / "prompt_length_report.md")
    print(f"PROMPT_LENGTH_AUDIT={summary['status']}")
    print(f"SUMMARY={summary_path}")
    return 0 if summary["status"] in {"PASS", "SMOKE_PASS"} else 1


if __name__ == "__main__":
    sys.exit(main())
