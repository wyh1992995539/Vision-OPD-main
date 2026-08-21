"""Generate frozen Base-model prefixes for the controlled cached-prefix ablation.

The OpenAI-compatible response exposes decoded text, not a guaranteed copy of
the engine's original sampled token IDs.  This script therefore re-encodes the
raw response with the exact Base tokenizer and records that provenance
explicitly.  Cached training must consume the frozen IDs written here.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Generate frozen Base cached prefixes.")
    parser.add_argument(
        "--config",
        type=Path,
        default=repository_root / "configs" / "day4_generation.yaml",
    )
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-parquet", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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


def extract_train_sample(row: dict[str, Any]) -> dict[str, Any]:
    """Extract only the fields allowed to enter Base prefix generation."""

    extra_info = row.get("extra_info")
    provenance = extra_info.get("provenance") if isinstance(extra_info, dict) else None
    if not isinstance(provenance, dict):
        raise ValueError("extra_info.provenance must be a mapping")
    split = str(provenance.get("split", ""))
    if split != "train":
        raise ValueError(f"cached prefix generation only accepts train rows, got {split!r}")
    sample_id = str(provenance.get("sample_id", "")).strip()
    if not sample_id:
        raise ValueError("extra_info.provenance.sample_id must be non-empty")

    prompt_text = _prompt_text(row.get("prompt"))
    return {
        "sample_id": sample_id,
        "source_id": str(provenance.get("source_id", "")),
        "split": split,
        "prompt_text": prompt_text,
        "prompt_sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
        "image_path": _single_path(row.get("images"), "images"),
    }


def validate_cached_protocol(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    shared = config.get("shared")
    cached = config.get("cached_prefix")
    if not isinstance(shared, dict) or not isinstance(cached, dict):
        raise ValueError("config must contain shared and cached_prefix mappings")
    generation = cached.get("generation")
    if not isinstance(generation, dict):
        raise ValueError("cached_prefix.generation must be a mapping")
    if generation.get("do_sample") is not True:
        raise ValueError("cached prefix generation must sample like the online rollout")
    if float(generation.get("temperature")) != 1.0:
        raise ValueError("cached prefix temperature must match the frozen online value 1.0")
    if int(generation.get("num_return_sequences")) != 1:
        raise ValueError("cached prefix generation requires one response per sample")
    if shared.get("student_image_key") != "images":
        raise ValueError("cached prefix generation must use the full student image key 'images'")
    return cached, generation


def load_train_samples(path: Path, expected_count: int) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to load train Parquet") from exc
    table = pq.read_table(path)
    rows = table.to_pylist()
    if len(rows) != expected_count:
        raise ValueError(f"expected {expected_count} train rows, found {len(rows)}")
    samples = [extract_train_sample(row) for row in rows]
    sample_ids = [sample["sample_id"] for sample in samples]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("train Parquet contains duplicate sample_id values")
    return samples


def encode_cached_response(
    tokenizer: Any, raw_text: str, finish_reason: str, keep_eos_token: bool
) -> tuple[list[int], bool]:
    token_ids = list(tokenizer.encode(raw_text, add_special_tokens=False))
    eos_appended = False
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if keep_eos_token and finish_reason == "stop" and eos_token_id is not None:
        if not token_ids or token_ids[-1] != int(eos_token_id):
            token_ids.append(int(eos_token_id))
            eos_appended = True
    return token_ids, eos_appended


def validate_cached_records(
    records: Iterable[dict[str, Any]], expected_sample_ids: Iterable[str]
) -> dict[str, Any]:
    materialized = list(records)
    expected_ids = list(expected_sample_ids)
    expected_set = set(expected_ids)
    if len(expected_set) != len(expected_ids):
        raise ValueError("expected sample IDs contain duplicates")

    actual_ids: list[str] = []
    empty_responses = 0
    inference_errors = 0
    empty_token_ids = 0
    truncated = 0
    for index, record in enumerate(materialized):
        sample_id = str(record.get("sample_id", "")).strip()
        if not sample_id:
            raise ValueError(f"cached record at index {index} has no sample_id")
        actual_ids.append(sample_id)
        if not str(record.get("raw_response_text", "")).strip():
            empty_responses += 1
        if record.get("inference_error"):
            inference_errors += 1
        token_ids = record.get("response_token_ids")
        if not isinstance(token_ids, list) or not token_ids:
            empty_token_ids += 1
        if record.get("finish_reason") == "length":
            truncated += 1

    actual_set = set(actual_ids)
    duplicates = len(actual_ids) - len(actual_set)
    missing = sorted(expected_set - actual_set)
    extra = sorted(actual_set - expected_set)
    status = (
        "PASS"
        if not duplicates
        and not missing
        and not extra
        and not empty_responses
        and not inference_errors
        and not empty_token_ids
        and not truncated
        else "FAIL"
    )
    report = {
        "status": status,
        "expected_records": len(expected_ids),
        "actual_records": len(materialized),
        "unique_sample_ids": len(actual_set),
        "duplicate_sample_ids": duplicates,
        "missing_sample_ids": missing,
        "extra_sample_ids": extra,
        "empty_responses": empty_responses,
        "inference_errors": inference_errors,
        "empty_token_ids": empty_token_ids,
        "truncated_responses": truncated,
    }
    if status != "PASS":
        raise ValueError(f"cached prefix validation failed: {json.dumps(report)}")
    return report


def write_jsonl_atomic(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            records.append(value)
    return records


def write_cache_artifacts(
    output_path: Path,
    records: list[dict[str, Any]],
    report_path: Path,
    hash_path: Path,
    report: dict[str, Any],
) -> dict[str, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to write cached Parquet") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    table = pa.Table.from_pylist(records)
    pq.write_table(table, temporary, compression="zstd")
    os.replace(temporary, output_path)
    digest = sha256_file(output_path)

    final_report = {
        **report,
        "output_parquet": output_path.as_posix(),
        "output_sha256": digest,
        "schema": table.schema.names,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(final_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    hash_path.parent.mkdir(parents=True, exist_ok=True)
    hash_path.write_text(f"{digest}  {output_path.as_posix()}\n", encoding="utf-8")
    return final_report


def smoke_paths(output_path: Path, limit: int) -> tuple[Path, Path, Path]:
    smoke_output = output_path.with_name(f"{output_path.stem}_smoke_{limit}{output_path.suffix}")
    report = output_path.parent / "manifests" / f"cached_prefix_report_smoke_{limit}.json"
    hashes = output_path.parent / "manifests" / f"cached_prefix_sha256_smoke_{limit}.txt"
    return smoke_output, report, hashes


def run(args: argparse.Namespace) -> None:
    try:
        from openai import OpenAI
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("openai and transformers are required for generation") from exc

    config = load_yaml(args.config.resolve())
    cached, generation = validate_cached_protocol(config)
    expected_count = int(cached["expected_samples"])
    train_path = Path(cached["input_parquet"])
    samples = load_train_samples(train_path, expected_count)

    if args.limit is not None:
        if args.limit <= 0 or args.limit > expected_count:
            raise ValueError(f"--limit must be between 1 and {expected_count}")
        samples = samples[: args.limit]

    configured_output = Path(cached["output_parquet"])
    report_path = Path(cached["report_file"])
    hash_path = Path(cached["sha256_file"])
    if args.limit is not None and args.output_parquet is None:
        output_path, report_path, hash_path = smoke_paths(configured_output, args.limit)
    else:
        output_path = args.output_parquet or configured_output
        if args.output_parquet is not None:
            report_path = output_path.with_suffix(".report.json")
            hash_path = output_path.with_suffix(".sha256.txt")

    checkpoint_path = output_path.with_suffix(output_path.suffix + ".inprogress.jsonl")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite existing cache: {output_path}")
    if args.overwrite:
        checkpoint_path.unlink(missing_ok=True)

    existing_records = load_jsonl(checkpoint_path) if checkpoint_path.exists() else []
    completed_ids = [str(record.get("sample_id", "")) for record in existing_records]
    if len(set(completed_ids)) != len(completed_ids):
        raise ValueError("checkpoint contains duplicate sample_id values")
    completed = set(completed_ids)

    model_path = Path(config["experiment"]["base_model_path"])
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    client = OpenAI(api_key=args.api_key, base_url=args.api_base, timeout=3600)
    keep_eos = bool(config["shared"]["output"]["keep_eos_token"])
    generation_hash = sha256_json(generation)
    records = list(existing_records)

    for index, sample in enumerate(samples, start=1):
        if sample["sample_id"] in completed:
            continue
        started = time.perf_counter()
        raw_text = ""
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
            raw_text = (choice.message.content or "").strip()
            finish_reason = str(choice.finish_reason or "unknown")
        except Exception as exc:  # Keep the failure in the checkpoint for audit.
            inference_error = f"{type(exc).__name__}: {exc}"

        response_ids, eos_appended = encode_cached_response(
            tokenizer, raw_text, finish_reason, keep_eos
        )
        records.append(
            {
                "sample_id": sample["sample_id"],
                "source_id": sample["source_id"],
                "split": "train",
                "dataset_index": index - 1,
                "image_path": str(sample["image_path"]),
                "prompt_sha256": sample["prompt_sha256"],
                "raw_response_text": raw_text,
                "response_token_ids": response_ids,
                "response_length": len(response_ids),
                "finish_reason": finish_reason,
                "inference_error": inference_error,
                "eos_appended_after_retokenization": eos_appended,
                "token_ids_source": "base_tokenizer_reencoded_openai_response_text",
                "generation_config_sha256": generation_hash,
                "model_id": args.model_id,
                "model_path": str(model_path),
                "latency_seconds": round(time.perf_counter() - started, 6),
            }
        )
        write_jsonl_atomic(checkpoint_path, records)
        print(
            f"[{index}/{len(samples)}] {sample['sample_id']} "
            f"tokens={len(response_ids)} finish={finish_reason}"
        )

    requested_ids = [sample["sample_id"] for sample in samples]
    requested_set = set(requested_ids)
    selected = [record for record in records if record.get("sample_id") in requested_set]
    selected.sort(key=lambda record: int(record["dataset_index"]))
    report = validate_cached_records(selected, requested_ids)
    report.update(
        {
            "experiment_id": config["experiment"]["id"],
            "model_id": args.model_id,
            "input_parquet": train_path.as_posix(),
            "generation_config_sha256": generation_hash,
            "token_ids_source": "base_tokenizer_reencoded_openai_response_text",
            "is_smoke": len(samples) != expected_count,
        }
    )
    final_report = write_cache_artifacts(
        output_path, selected, report_path, hash_path, report
    )
    checkpoint_path.unlink(missing_ok=True)
    print(json.dumps(final_report, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
