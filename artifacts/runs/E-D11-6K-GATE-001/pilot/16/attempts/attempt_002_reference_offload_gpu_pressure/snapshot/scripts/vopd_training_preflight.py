#!/usr/bin/env python3
"""Validate a config-driven Vision-OPD two-GPU training run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import yaml


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(project_root: Path, value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else project_root / path).resolve()


def training_config(config: dict[str, Any]) -> dict[str, Any]:
    value = config.get("training", config.get("smoke"))
    if not isinstance(value, dict):
        raise ValueError("config must contain a training or legacy smoke mapping")
    return value


def sample_ids(table: Any) -> list[str]:
    values: list[str] = []
    for row_index, row in enumerate(table.select(["extra_info"]).to_pylist()):
        try:
            value = str(row["extra_info"]["provenance"]["sample_id"]).strip()
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"row {row_index}: missing extra_info.provenance.sample_id"
            ) from exc
        if not value:
            raise ValueError(f"row {row_index}: sample_id is empty")
        values.append(value)
    if len(set(values)) != len(values):
        raise ValueError("training Parquet contains duplicate sample_id values")
    return values


def validate_selection_manifest(
    manifest_path: Path,
    *,
    experiment_id: str,
    train_file: Path,
    train_sha256: str,
    actual_sample_ids: list[str],
    seed: int,
) -> list[str]:
    errors: list[str] = []
    if not manifest_path.is_file():
        return [f"selection manifest not found: {manifest_path}"]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"selection manifest is unreadable: {exc}"]

    output = manifest.get("output", {})
    source = manifest.get("source", {})
    selection = manifest.get("selection", {})
    samples = manifest.get("samples", [])
    manifest_ids = [str(item.get("sample_id", "")) for item in samples]
    if manifest.get("experiment_id") != experiment_id:
        errors.append("selection manifest experiment ID does not match config")
    if output.get("sha256") != train_sha256:
        errors.append("selection manifest output SHA256 does not match training Parquet")
    if int(output.get("rows", -1)) != len(actual_sample_ids):
        errors.append("selection manifest output row count does not match training Parquet")
    if Path(str(output.get("path", ""))).resolve() != train_file:
        errors.append("selection manifest output path does not match training Parquet")
    if int(selection.get("seed", -1)) != seed:
        errors.append("selection manifest seed does not match experiment seed")
    if manifest_ids != actual_sample_ids:
        errors.append("selection manifest sample order does not match training Parquet")
    source_path = Path(str(source.get("path", ""))).resolve()
    if not source_path.is_file():
        errors.append(f"selection manifest source Parquet not found: {source_path}")
    elif source.get("sha256") != sha256_file(source_path):
        errors.append("selection manifest source SHA256 does not match source Parquet")
    return errors


def validate_config(config_path: Path, project_root: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = project_root.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    training = training_config(config)
    data = config["data"]
    actor = config["actor"]
    rollout = config["rollout"]
    resources = config["resources"]
    experiment = config["experiment"]

    errors: list[str] = []
    model_path = resolve(project_root, config["paths"]["model"])
    train_file = resolve(project_root, config["paths"]["train_file"])
    chat_template = resolve(project_root, config["paths"]["chat_template"])

    required_model_files = [
        "config.json",
        "model.safetensors.index.json",
        "tokenizer_config.json",
        "preprocessor_config.json",
    ]
    missing_model_files = [name for name in required_model_files if not (model_path / name).is_file()]
    if missing_model_files:
        errors.append(f"missing model files: {missing_model_files}")
    if not list(model_path.glob("model-*.safetensors")) and not list(
        model_path.glob("model.safetensors-*.safetensors")
    ):
        errors.append("no model safetensors shards found")
    if not train_file.is_file():
        errors.append(f"training Parquet not found: {train_file}")
    if not chat_template.is_file():
        errors.append(f"chat template not found: {chat_template}")

    row_count: int | None = None
    columns: list[str] = []
    actual_sample_ids: list[str] = []
    missing_image_paths: list[dict[str, Any]] = []
    train_sha256: str | None = None
    if train_file.is_file():
        train_sha256 = sha256_file(train_file)
        table = pq.read_table(train_file)
        row_count = table.num_rows
        columns = table.column_names
        expected_rows = int(data["expected_train_rows"])
        if row_count != expected_rows:
            errors.append(f"expected {expected_rows} rows, found {row_count}")
        required_columns = {
            "prompt",
            data["image_key"],
            data["teacher_image_key"],
            "extra_info",
        }
        missing_columns = sorted(required_columns.difference(columns))
        if missing_columns:
            errors.append(f"missing Parquet columns: {missing_columns}")
        else:
            try:
                actual_sample_ids = sample_ids(table)
            except ValueError as exc:
                errors.append(str(exc))
            for column_name in (data["image_key"], data["teacher_image_key"]):
                for row_index, items in enumerate(table[column_name].to_pylist()):
                    for item in items or []:
                        image_path = Path(item["path"])
                        if not image_path.is_file():
                            missing_image_paths.append(
                                {"row": row_index, "column": column_name, "path": str(image_path)}
                            )
                            if len(missing_image_paths) >= 20:
                                break
                    if len(missing_image_paths) >= 20:
                        break
                if len(missing_image_paths) >= 20:
                    break
            if missing_image_paths:
                errors.append("one or more image paths are missing; first 20 are recorded")

    expected_samples = int(training["expected_samples"])
    total_steps = int(training["total_optimizer_steps"])
    batch_size = int(data["train_batch_size"])
    total_epochs = int(training.get("total_epochs", 1))
    require_full_epoch = bool(training.get("require_full_epoch", False))
    coverage = data.get("full_coverage_padding") or {}
    coverage_enabled = bool(coverage.get("enabled", False))
    expected_padding_rows = int(training.get("padding_rows", 0))
    expected_dropped_rows = int(training.get("dropped_rows", 0))
    padded_samples = int(training.get("padded_samples", expected_samples))
    paper_alignment = config.get("paper_alignment") or {}
    required_response_length = int(
        paper_alignment.get("required_max_response_length", data["max_response_length"])
    )
    self_distillation = config.get("self_distillation") or {}
    vllm_engine = rollout.get("engine_kwargs", {}).get("vllm", {})
    vllm_compilation = vllm_engine.get("compilation_config", {}).get("pass_config", {})
    vllm_kernel = vllm_engine.get("kernel_config", {})
    checks = {
        "config_not_explicitly_blocked": not str(config.get("status", "")).startswith("blocked"),
        "prefix_source_online": experiment["prefix_source"] == "online",
        "seed_is_42": int(experiment["seed"]) == 42,
        "two_gpus": int(resources["gpus_per_node"]) == 2,
        "global_batch_is_8": batch_size == 8,
        "rollout_n_is_1": int(rollout["n"]) == 1,
        "positive_optimizer_steps": total_steps > 0,
        "sample_budget_matches_steps": (
            padded_samples == total_steps * batch_size
            and expected_samples + expected_padding_rows == padded_samples
            if coverage_enabled
            else expected_samples == total_steps * batch_size
        ),
        "sample_budget_within_dataset": row_count is None or expected_samples <= row_count * total_epochs,
        "full_epoch_contract": not require_full_epoch
        or (row_count is not None and expected_samples == row_count * total_epochs),
        "full_coverage_contract": (
            not coverage_enabled
            or (
                int(coverage.get("multiple", -1)) == batch_size
                and int(coverage.get("expected_unique_samples", -1)) == expected_samples
                and int(coverage.get("expected_padding_rows", -1)) == expected_padding_rows
                and expected_padding_rows == (-expected_samples) % batch_size
                and bool(coverage.get("receipt_path"))
            )
        ),
        "native_drop_last_contract": (
            coverage_enabled
            or (
                data.get("tail_policy", "native_drop_last") == "native_drop_last"
                and expected_padding_rows == 0
                and padded_samples == expected_samples
                and row_count is not None
                and total_steps == (row_count // batch_size) * total_epochs
                and expected_samples == total_steps * batch_size
                and expected_dropped_rows
                == (row_count * total_epochs) - expected_samples
            )
        ),
        "prompt_limit_is_8192": int(data["max_prompt_length"]) == 8192,
        "response_limit_matches_frozen_contract": (
            int(data["max_response_length"]) == required_response_length
        ),
        "truncation_is_error": data["truncation"] == "error",
        "teacher_uses_bbox_images": data["teacher_image_key"] == "bbox_images",
        "actor_parameter_offload_disabled": actor["parameter_offload"] is False,
        "actor_optimizer_offload_disabled": actor["optimizer_offload"] is False,
        "reference_parameter_offload_matches_resource_contract": (
            "reference_parameter_offload_required" not in resources
            or actor["reference_parameter_offload"]
            is bool(resources["reference_parameter_offload_required"])
        ),
        "rollout_gpu_memory_utilization_safe_start": float(rollout["gpu_memory_utilization"])
        <= 0.5,
    }
    if paper_alignment:
        checks.update(
            {
                "paper_jsd_beta_is_0_5": float(self_distillation["alpha"]) == 0.5,
                "paper_distillation_top_k_is_100": int(self_distillation["top_k"]) == 100,
                "paper_teacher_ema_rate_is_0_05": (
                    self_distillation["teacher_regularization"] == "ema"
                    and float(self_distillation["teacher_update_rate"]) == 0.05
                ),
                "paper_epoch_is_1": total_epochs == 1,
                "paper_rollout_sampling_is_frozen": (
                    float(rollout["temperature"]) == 1.0
                    and float(rollout["top_p"]) == 1.0
                    and int(rollout["top_k"]) == -1
                    and rollout["ignore_eos"] is False
                ),
                "official_lr_and_warmup_match": (
                    float(actor["learning_rate"]) == 2.0e-6
                    and int(actor["lr_warmup_steps"]) == 10
                ),
                "official_clip_contract_matches": (
                    float(actor["clip_ratio_low"]) == 0.2
                    and float(actor["clip_ratio_high"]) == 0.3
                ),
                "official_max_reprompt_length_matches": (
                    int(self_distillation["max_reprompt_length"]) == 10240
                ),
                "official_vllm_engine_switches_match": (
                    vllm_compilation.get("fuse_allreduce_rms") is False
                    and vllm_kernel.get("enable_flashinfer_autotune") is False
                ),
                "sequence_token_budget_matches": (
                    int(actor["max_token_length_per_gpu"])
                    == int(data["max_prompt_length"]) + int(data["max_response_length"])
                ),
            }
        )
    failed_checks = sorted(name for name, passed in checks.items() if not passed)
    if failed_checks:
        errors.append(f"frozen config checks failed: {failed_checks}")

    selection_manifest_path: Path | None = None
    selection_manifest_value = config["paths"].get("selection_manifest")
    if selection_manifest_value:
        selection_manifest_path = resolve(project_root, selection_manifest_value)
        if train_sha256 is not None and actual_sample_ids:
            errors.extend(
                validate_selection_manifest(
                    selection_manifest_path,
                    experiment_id=str(experiment["id"]),
                    train_file=train_file,
                    train_sha256=train_sha256,
                    actual_sample_ids=actual_sample_ids,
                    seed=int(experiment["seed"]),
                )
            )

    return {
        "schema_version": 1,
        "experiment_id": experiment["id"],
        "status": "PASS" if not errors else "FAIL",
        "gpu_used": False,
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "model_path": str(model_path),
        "train_file": str(train_file),
        "train_file_sha256": train_sha256,
        "train_rows": row_count,
        "sample_ids": actual_sample_ids,
        "parquet_columns": columns,
        "chat_template": str(chat_template),
        "selection_manifest": str(selection_manifest_path) if selection_manifest_path else None,
        "training_contract": {
            "expected_samples": expected_samples,
            "padded_samples": padded_samples,
            "padding_rows": expected_padding_rows,
            "dropped_rows": expected_dropped_rows,
            "full_coverage_padding": coverage_enabled,
            "tail_policy": data.get("tail_policy", "native_drop_last"),
            "total_optimizer_steps": total_steps,
            "total_epochs": total_epochs,
            "require_full_epoch": require_full_epoch,
            "train_batch_size": batch_size,
            "shuffle": bool(data["shuffle"]),
            "save_frequency": int(training["save_frequency"]),
            "test_frequency": int(training["test_frequency"]),
        },
        "checks": checks,
        "missing_image_paths": missing_image_paths,
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = validate_config(args.config, args.project_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"VOPD_PREFLIGHT={summary['status']}")
    print(f"EXPERIMENT_ID={summary['experiment_id']}")
    print(f"SUMMARY={args.output.resolve()}")
    print(f"TRAIN_ROWS={summary['train_rows']}")
    print(f"MISSING_IMAGE_PATHS={len(summary['missing_image_paths'])}")
    if summary["errors"]:
        for error in summary["errors"]:
            print(f"ERROR={error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
