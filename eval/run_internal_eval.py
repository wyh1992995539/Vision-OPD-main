"""Run the frozen deterministic internal evaluation through a vLLM OpenAI API."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import sys
import time
from pathlib import Path
from typing import Any

try:
    from .internal_eval import (
        build_prediction_record,
        load_jsonl,
        summarize_predictions,
        write_json_atomic,
        write_jsonl_atomic,
    )
except ImportError:
    from internal_eval import (  # type: ignore[no-redef]
        build_prediction_record,
        load_jsonl,
        summarize_predictions,
        write_json_atomic,
        write_jsonl_atomic,
    )


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Evaluate the frozen multiple-choice split without an LLM judge."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=repository_root / "configs" / "day4_generation.yaml",
    )
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model-id", required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--limit", type=int, default=None)
    selection.add_argument(
        "--sample-id",
        action="append",
        default=None,
        help="Evaluate one known eval sample_id; repeat for a targeted smoke test.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--score-only",
        type=Path,
        default=None,
        help="Recompute summary.json from an existing predictions.jsonl without inference.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read the generation protocol") from exc
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    return value


def image_to_data_uri(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"student image not found: {path}")
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _single_path(value: Any, field_name: str) -> Path:
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], dict):
        raise ValueError(f"{field_name} must contain exactly one path record")
    raw_path = value[0].get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"{field_name}[0].path must be non-empty")
    return Path(raw_path)


def _prompt_text(value: Any) -> str:
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], dict):
        raise ValueError("prompt must contain exactly one message")
    if value[0].get("role") != "user":
        raise ValueError("prompt message role must be user")
    content = value[0].get("content")
    if not isinstance(content, str) or content.count("<image>") != 1:
        raise ValueError("prompt content must contain exactly one <image> placeholder")
    return content.replace("<image>", "", 1).strip()


def extract_eval_sample(row: dict[str, Any]) -> dict[str, Any]:
    extra_info = row.get("extra_info")
    provenance = extra_info.get("provenance") if isinstance(extra_info, dict) else None
    reward_model = row.get("reward_model")
    if not isinstance(provenance, dict):
        raise ValueError("extra_info.provenance must be a mapping")
    if not isinstance(reward_model, dict):
        raise ValueError("reward_model must be a mapping")

    split = str(provenance.get("split", ""))
    if split != "eval":
        raise ValueError(f"internal evaluation only accepts eval rows, got {split!r}")
    question_type = str(provenance.get("question_type", ""))
    sample_id = str(provenance.get("sample_id", "")).strip()
    if not sample_id:
        raise ValueError("extra_info.provenance.sample_id must be non-empty")

    return {
        "sample_id": sample_id,
        "source_id": str(provenance.get("source_id", "")),
        "question_type": question_type,
        "ground_truth": reward_model.get("ground_truth"),
        "prompt_text": _prompt_text(row.get("prompt")),
        "image_path": _single_path(row.get("images"), "images"),
    }


def load_eval_rows(path: Path, expected_count: int) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to load eval Parquet") from exc
    table = pq.read_table(path)
    rows = table.to_pylist()
    if len(rows) != expected_count:
        raise ValueError(f"expected {expected_count} eval rows, found {len(rows)}")
    samples = [extract_eval_sample(row) for row in rows]
    sample_ids = [sample["sample_id"] for sample in samples]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("eval Parquet contains duplicate sample_id values")
    return samples


def select_eval_samples(
    samples: list[dict[str, Any]],
    limit: int | None = None,
    sample_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Select either the leading smoke rows or explicitly named eval rows."""

    if sample_ids:
        requested = [sample_id.strip() for sample_id in sample_ids]
        if not all(requested):
            raise ValueError("--sample-id must be non-empty")
        if len(set(requested)) != len(requested):
            raise ValueError("--sample-id values must be unique")
        by_id = {sample["sample_id"]: sample for sample in samples}
        missing = [sample_id for sample_id in requested if sample_id not in by_id]
        if missing:
            raise ValueError(f"unknown eval sample_id values: {missing}")
        return [by_id[sample_id] for sample_id in requested]

    if limit is not None:
        if limit <= 0 or limit > len(samples):
            raise ValueError(f"--limit must be between 1 and {len(samples)}")
        return samples[:limit]
    return samples


def _resolve_repo_path(repository_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repository_root / path


def _validate_eval_protocol(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    shared = config.get("shared")
    evaluation = config.get("evaluation")
    if not isinstance(shared, dict) or not isinstance(evaluation, dict):
        raise ValueError("config must contain shared and evaluation mappings")
    generation = evaluation.get("generation")
    if not isinstance(generation, dict):
        raise ValueError("evaluation.generation must be a mapping")
    if generation.get("do_sample") is not False or float(generation.get("temperature")) != 0.0:
        raise ValueError("internal evaluation must use greedy generation")
    if int(generation.get("num_return_sequences")) != 1:
        raise ValueError("internal evaluation requires exactly one response per sample")
    if shared.get("student_image_key") != "images":
        raise ValueError("internal evaluation must use the full student image key 'images'")
    return evaluation, generation


def run_inference(args: argparse.Namespace, config: dict[str, Any]) -> None:
    try:
        from openai import OpenAI
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("openai and transformers are required for inference") from exc

    repository_root = Path(__file__).resolve().parents[1]
    evaluation, generation = _validate_eval_protocol(config)
    expected_count = int(evaluation["expected_samples"])
    eval_path = _resolve_repo_path(repository_root, evaluation["input_parquet"])
    output_dir = args.output_dir or _resolve_repo_path(repository_root, evaluation["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    summary_path = output_dir / "summary.json"

    samples = select_eval_samples(
        load_eval_rows(eval_path, expected_count),
        limit=args.limit,
        sample_ids=args.sample_id,
    )

    existing_records: list[dict[str, Any]] = []
    if predictions_path.exists() and not args.overwrite:
        existing_records = load_jsonl(predictions_path)
    completed = {str(record.get("sample_id", "")) for record in existing_records}
    if len(completed) != len(existing_records):
        raise ValueError("existing predictions contain missing or duplicate sample_id values")

    model_path = Path(config["experiment"]["base_model_path"])
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    client = OpenAI(api_key=args.api_key, base_url=args.api_base, timeout=3600)
    records = list(existing_records)

    for index, sample in enumerate(samples, start=1):
        if sample["sample_id"] in completed:
            continue
        started = time.perf_counter()
        raw_prediction = ""
        finish_reason = "error"
        inference_error = None
        try:
            response = client.chat.completions.create(
                model=args.model_id,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": image_to_data_uri(sample["image_path"])},
                            },
                            {"type": "text", "text": sample["prompt_text"]},
                        ],
                    }
                ],
                max_tokens=int(generation["max_new_tokens"]),
                temperature=float(generation["temperature"]),
                top_p=float(generation["top_p"]),
                n=1,
                seed=int(generation["seed"]),
                extra_body={
                    "top_k": int(generation["top_k"]),
                    "chat_template_kwargs": {
                        "enable_thinking": bool(
                            config["shared"]["chat_template"]["enable_thinking"]
                        )
                    },
                },
            )
            choice = response.choices[0]
            raw_prediction = (choice.message.content or "").strip()
            finish_reason = str(choice.finish_reason or "unknown")
        except Exception as exc:  # Preserve a per-sample failure as evidence.
            inference_error = f"{type(exc).__name__}: {exc}"

        response_token_ids = tokenizer.encode(raw_prediction, add_special_tokens=False)
        record = build_prediction_record(
            sample_id=sample["sample_id"],
            ground_truth=sample["ground_truth"],
            raw_prediction=raw_prediction,
            question_type=sample["question_type"],
            metadata={
                "source_id": sample["source_id"],
                "image_path": str(sample["image_path"]),
                "finish_reason": finish_reason,
                "inference_error": inference_error,
                "latency_seconds": round(time.perf_counter() - started, 6),
                "response_token_ids": response_token_ids,
                "response_token_count": len(response_token_ids),
                "token_ids_source": "retokenized_response_text_without_special_tokens",
                "dataset_index": index - 1,
            },
        )
        records.append(record)
        write_jsonl_atomic(predictions_path, records)
        print(
            f"[{index}/{len(samples)}] {sample['sample_id']} "
            f"status={record['score_status']} prediction={record['parsed_prediction']}"
        )

    requested_ids = {sample["sample_id"] for sample in samples}
    selected_records = [record for record in records if record.get("sample_id") in requested_ids]
    selected_records.sort(key=lambda record: int(record.get("dataset_index", 0)))
    write_jsonl_atomic(predictions_path, selected_records)
    summary = summarize_predictions(selected_records, expected_count=len(samples))
    summary.update(
        {
            "experiment_id": config["experiment"]["id"],
            "model_id": args.model_id,
            "input_parquet": str(eval_path),
            "is_smoke": len(samples) != expected_count,
        }
    )
    write_json_atomic(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def score_only(args: argparse.Namespace, config: dict[str, Any]) -> None:
    evaluation, _generation = _validate_eval_protocol(config)
    records = load_jsonl(args.score_only)
    expected_count = (
        len(args.sample_id)
        if args.sample_id is not None
        else args.limit
        if args.limit is not None
        else int(evaluation["expected_samples"])
    )
    summary = summarize_predictions(records, expected_count=expected_count)
    output_dir = args.output_dir or args.score_only.parent
    write_json_atomic(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config.resolve())
    if args.score_only is not None:
        score_only(args, config)
    else:
        run_inference(args, config)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
