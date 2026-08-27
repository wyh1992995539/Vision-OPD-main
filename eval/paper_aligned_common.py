#!/usr/bin/env python3
"""Shared contracts for the E-PAPER-BASEJUDGE-001 evaluation pipeline."""

from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import os
import re
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Callable

import yaml
from PIL import Image


BENCHMARK_ORDER = ("zoombench", "mmstar", "vstar")
VIEW = "full"
VSTAR_MAX_DEFAULT = 20 * 1024 * 1024
FROZEN_R3_CONFIG_SHA256 = (
    "e71255e817b11c120b4ac22d7ace81d12ffe01e25f7ea94de2e2ffb62e592903"
)
FORMAL_COMPARABILITY_FIELDS = (
    "experiment_id",
    "run_mode",
    "config_sha256_raw_bytes",
    "amendment_sha256_raw_bytes",
    "dataset_files",
    "request_contract",
    "expected_requests",
    "expected_request_count",
    "limit_per_benchmark",
    "resume_key",
)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_path(value: str | Path, root: Path | None = None) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ((root or repo_root()) / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def record_key(item: dict[str, Any]) -> str:
    return f"{item['benchmark']}\0{item.get('view', VIEW)}\0{item['sample_uid']}"


def read_jsonl_map(
    path: Path,
    *,
    complete: Callable[[dict[str, Any]], bool] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    records: dict[str, dict[str, Any]] = {}
    malformed = duplicates = 0
    if not path.is_file():
        return records, {"malformed_lines": 0, "duplicate_keys": 0}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            malformed += 1
            continue
        try:
            item = json.loads(raw_line)
            key = record_key(item)
        except (json.JSONDecodeError, KeyError, TypeError):
            malformed += 1
            continue
        if key in records:
            duplicates += 1
            old = records[key]
            if complete is None or not complete(old) or complete(item):
                records[key] = item
        else:
            records[key] = item
    return records, {"malformed_lines": malformed, "duplicate_keys": duplicates}


def write_jsonl_map(path: Path, records: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    ordered = sorted(records.values(), key=lambda x: (x["benchmark"], x.get("view", VIEW), x["sample_uid"]))
    with temporary.open("w", encoding="utf-8") as handle:
        for item in ordered:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def append_jsonl(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def prediction_complete(item: dict[str, Any]) -> bool:
    return not item.get("error") and bool(str(item.get("raw_model_answer") or "").strip())


def judge_complete(item: dict[str, Any]) -> bool:
    if item.get("finalized") is True:
        return True
    return (
        not item.get("error")
        and str(item.get("normalized_decision") or "").casefold() in {"yes", "no"}
    )


def load_config(path: str | Path) -> tuple[Path, dict[str, Any]]:
    config_path = resolve_path(path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_config(config, config_path)
    return config_path, config


def validate_config(config: dict[str, Any], config_path: Path | None = None) -> None:
    protocol = config.get("protocol", {})
    if protocol.get("experiment_id") != "E-PAPER-BASEJUDGE-001":
        raise ValueError("paper-aligned config must use experiment_id E-PAPER-BASEJUDGE-001")
    prompt = config.get("prompt_and_image", {})
    generation = config.get("generation", {})
    if prompt.get("system_prompt") is not None or prompt.get("message_roles") != ["user"]:
        raise ValueError("paper-aligned inference must contain one user message and no system prompt")
    if generation.get("enable_thinking") is not False:
        raise ValueError("paper-aligned inference requires enable_thinking=false")
    if generation.get("temperature") != 0 or generation.get("max_tokens") != 1024:
        raise ValueError("paper-aligned inference requires temperature=0 and max_tokens=1024")
    forbidden = {"seed", "top_p", "top_k", "presence_penalty", "repetition_penalty"}
    declared = set(generation.get("forbidden_request_parameters", []))
    if declared != forbidden:
        raise ValueError(f"forbidden inference parameter set mismatch: {declared}")
    judge = config.get("judge", {})
    if judge.get("system_prompt") is not None:
        raise ValueError("paper-aligned Judge must not use a system prompt")
    if judge.get("forbid_trained_checkpoint_as_judge") is not True:
        raise ValueError("trained checkpoints must be forbidden as Judge")
    judge_generation = judge.get("generation", {})
    if (
        judge_generation.get("enable_thinking") is not False
        or judge_generation.get("temperature") != 0
        or judge_generation.get("max_tokens") != 2048
    ):
        raise ValueError("Judge requires thinking=false, temperature=0, max_tokens=2048")
    expected_total = 0
    for name in BENCHMARK_ORDER:
        benchmark = config.get("benchmarks", {}).get(name, {})
        count = int(benchmark.get("expected_sample_count", -1))
        denominator = int(benchmark.get("primary_summary_denominator", -2))
        if count <= 0 or denominator != count or benchmark.get("primary_view") != VIEW:
            raise ValueError(f"{name}: invalid count, denominator, or primary view")
        expected_total += count
    if expected_total != 2536 or config["reporting"].get("expected_total_visual_requests") != 2536:
        raise ValueError("formal paper-aligned request count must be 2536")
    if config["benchmarks"]["vstar"].get("primary_summary_denominator") != 191:
        raise ValueError("V* denominator must be 191")
    if int(protocol.get("protocol_revision", 1)) >= 2:
        serving = config.get("serving", {})
        if (
            serving.get("tensor_parallel_size") != 1
            or serving.get("gpu_memory_utilization") != 0.75
            or serving.get("additional_config", {}).get("gdn_prefill_backend") != "triton"
        ):
            raise ValueError("R2 serving requires TP=1, gpu_memory_utilization=0.75, and GDN Triton")
        if prompt.get("vstar_encoding") != "always_rgb_png":
            raise ValueError("R2 V* requests must always use RGB PNG encoding")
        if prompt.get("zoombench_mmstar_encoding") != "preserve_source_bytes":
            raise ValueError("R2 ZoomBench/MMStar preparation must preserve source bytes")
        for name in ("zoombench", "mmstar"):
            benchmark = config["benchmarks"][name]
            if not benchmark.get("source_parquet") or "/paper_aligned/" not in benchmark.get("converted_json", ""):
                raise ValueError(f"R2 {name} must use its pinned Parquet and paper_aligned output")
    if config_path is not None:
        amendment = resolve_path(protocol["amendment"], config_path.parent.parent)
        expected_hash = str(protocol["amendment_sha256"])
        if not amendment.is_file() or sha256_file(amendment) != expected_hash:
            raise ValueError("paper-aligned amendment is missing or its SHA256 changed")


def checkpoint_identity(model_path: Path) -> dict[str, Any]:
    model_path = model_path.resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"model path is not a directory: {model_path}")
    weight_files = sorted(model_path.glob("*.safetensors"))
    if not weight_files:
        raise FileNotFoundError(f"no *.safetensors files in {model_path}")
    hashes = {path.name: sha256_file(path) for path in weight_files}
    metadata = {}
    for name in ("config.json", "generation_config.json", "tokenizer_config.json"):
        path = model_path / name
        if path.is_file():
            metadata[name] = sha256_file(path)
    return {
        "path": str(model_path),
        "weight_sha256": hashes,
        "metadata_sha256": metadata,
    }


def require_frozen_base_identity(identity: dict[str, Any], config: dict[str, Any]) -> None:
    expected = config["judge"]["model"]["base_weight_sha256"]
    actual = identity.get("weight_sha256", {})
    if actual != expected:
        raise ValueError(
            "Judge checkpoint is not the frozen original Qwen3.5-4B Base; "
            f"expected {expected}, got {actual}"
        )


def _validate_row(name: str, row: dict[str, Any], revision: str) -> None:
    required = ("sample_uid", "source_id", "query", "response", "images")
    missing = [key for key in required if key not in row]
    if missing:
        raise ValueError(f"{name}: row missing fields {missing}")
    if str(row.get("source_revision")) != revision:
        raise ValueError(
            f"{name}/{row['sample_uid']}: source_revision={row.get('source_revision')!r}, "
            f"expected {revision!r}; prepare the paper-aligned dataset first"
        )
    images = row.get("images")
    if not isinstance(images, list) or len(images) != 1:
        raise ValueError(f"{name}/{row['sample_uid']}: exactly one full image is required")
    if not Path(str(images[0])).is_file():
        raise FileNotFoundError(f"{name}/{row['sample_uid']}: missing image {images[0]}")


def load_tasks(config: dict[str, Any], limit_per_benchmark: int | None = None) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for name in BENCHMARK_ORDER:
        benchmark = config["benchmarks"][name]
        path = resolve_path(benchmark["converted_json"])
        rows = json.loads(path.read_text(encoding="utf-8"))
        expected = int(benchmark["expected_sample_count"])
        if len(rows) != expected or len({str(row.get("sample_uid")) for row in rows}) != expected:
            raise ValueError(f"{name}: expected {expected} unique converted rows, found {len(rows)}")
        for row in rows:
            _validate_row(name, row, str(benchmark["dataset_revision"]))
        selected = sorted(rows, key=lambda row: str(row["sample_uid"]))
        if limit_per_benchmark is not None:
            if limit_per_benchmark <= 0:
                raise ValueError("--limit-per-benchmark must be positive")
            selected = selected[:limit_per_benchmark]
        for row in selected:
            tasks.append(
                {
                    "benchmark": name,
                    "view": VIEW,
                    "row": row,
                    "image_path": Path(str(row["images"][0])),
                }
            )
    return tasks


def require_frozen_model_under_test_base_identity(
    identity: dict[str, Any], config: dict[str, Any]
) -> None:
    expected = config["model_under_test"]["base_weight_sha256"]
    actual = identity.get("weight_sha256", {})
    if actual != expected:
        raise ValueError(
            "model_role=base requires the frozen original Qwen3.5-4B Base; "
            f"expected {expected}, got {actual}"
        )


def require_frozen_r3_config(config_path: Path) -> None:
    actual = sha256_file(config_path)
    if actual != FROZEN_R3_CONFIG_SHA256:
        raise ValueError(
            "formal evaluation requires the sole frozen R3 config SHA256; "
            f"expected {FROZEN_R3_CONFIG_SHA256}, got {actual}"
        )


def require_formal_manifest_comparable_with_base(
    manifest: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    """Reject formal runs that cannot be compared to the frozen Base manifest."""
    if manifest.get("run_mode") != "formal":
        return {"status": "not_applicable", "reason": "smoke_run"}

    actual_config_sha = str(manifest.get("config_sha256_raw_bytes") or "")
    if actual_config_sha != FROZEN_R3_CONFIG_SHA256:
        raise ValueError(
            "formal run manifest does not use the frozen R3 config SHA256: "
            f"{actual_config_sha}"
        )

    model_role = str(manifest.get("model_role") or "")
    allowed_roles = set(config["model_under_test"]["allowed_roles"])
    if model_role not in allowed_roles:
        raise ValueError(f"unsupported formal model_role: {model_role!r}")
    if model_role == "base":
        require_frozen_model_under_test_base_identity(
            manifest.get("model_checkpoint_identity", {}), config
        )
        return {
            "status": "pass",
            "reference_role": "base",
            "checked_fields": list(FORMAL_COMPARABILITY_FIELDS),
            "base_identity": "pass",
        }

    base_manifest_path = (
        resolve_path(config["paths"]["run_root"])
        / "base"
        / config["paths"]["run_manifest_name"]
    )
    if not base_manifest_path.is_file():
        raise FileNotFoundError(
            f"frozen Base run manifest required for comparison: {base_manifest_path}"
        )
    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    mismatches = [
        field
        for field in FORMAL_COMPARABILITY_FIELDS
        if manifest.get(field) != base_manifest.get(field)
    ]
    if mismatches:
        raise ValueError(
            "formal run is not comparable with the frozen Base manifest; "
            f"mismatched fields: {mismatches}"
        )
    require_frozen_model_under_test_base_identity(
        base_manifest.get("model_checkpoint_identity", {}), config
    )
    return {
        "status": "pass",
        "reference_role": "base",
        "reference_manifest": str(base_manifest_path),
        "checked_fields": list(FORMAL_COMPARABILITY_FIELDS),
        "base_identity": "pass",
    }


def expected_counts(config: dict[str, Any], limit_per_benchmark: int | None = None) -> dict[str, int]:
    return {
        f"{name}/{VIEW}": (
            min(int(config["benchmarks"][name]["expected_sample_count"]), limit_per_benchmark)
            if limit_per_benchmark is not None
            else int(config["benchmarks"][name]["expected_sample_count"])
        )
        for name in BENCHMARK_ORDER
    }


def inference_messages(image_uri: str, query: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_uri}},
                {"type": "text", "text": query.replace("<image>", "").strip()},
            ],
        }
    ]


def inference_request_kwargs(
    *,
    model_id: str,
    messages: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    generation = config["generation"]
    result = {
        "model": model_id,
        "messages": messages,
        "temperature": 0,
        "max_tokens": int(generation["max_tokens"]),
        "extra_body": {
            "chat_template_kwargs": {
                "enable_thinking": bool(generation["enable_thinking"]),
            }
        },
    }
    forbidden = set(generation["forbidden_request_parameters"])
    leaked = forbidden.intersection(result)
    if leaked:
        raise AssertionError(f"forbidden inference request parameters leaked: {sorted(leaked)}")
    return result


def image_data_uri(
    path: Path,
    benchmark: str,
    max_vstar_bytes: int = VSTAR_MAX_DEFAULT,
    *,
    vstar_always_rgb_png: bool = False,
) -> str:
    raw = path.read_bytes()
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    if benchmark != "vstar" or (not vstar_always_rgb_png and len(raw) <= max_vstar_bytes):
        return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"
    with Image.open(BytesIO(raw)) as opened:
        image = opened.convert("RGB")
    while True:
        output = BytesIO()
        image.save(output, format="PNG")
        encoded = output.getvalue()
        if len(encoded) <= max_vstar_bytes or min(image.size) <= 100:
            return f"data:image/png;base64,{base64.b64encode(encoded).decode('ascii')}"
        image = image.resize(
            (max(1, int(image.width * 0.75)), max(1, int(image.height * 0.75))),
            Image.Resampling.LANCZOS,
        )


def extract_answer_official(raw: Any) -> str:
    text = str(raw or "")
    if "<answer>" in text:
        start = text.find("<answer>")
        end = text.find("</answer>")
        if start != -1 and end != -1:
            return text[start + len("<answer>") : end].strip()
    if "Answer:" in text:
        return text[text.find("Answer:") :].strip()
    return text.strip()


def extract_first_option(text: Any) -> str:
    value = str(text or "")
    match = re.search(r"\(([A-Z])\)", value)
    if match:
        return match.group(1)
    match = re.search(r"([A-Z])[\.\)\s]", value)
    if match:
        return match.group(1)
    match = re.search(r"([A-Z])", value)
    return match.group(1) if match else ""


def extract_mcq_option(answer: Any) -> str:
    value = str(answer or "").strip()
    match = re.match(r"^[ (\[]*([A-F])(?:(?=$)|[\.\)\]]|(?:[\:\-]\s+))", value)
    return match.group(1) if match else ""


def first_letter_match(reference: Any, answer: Any) -> bool:
    expected = extract_mcq_option(reference)
    predicted = extract_first_option(answer)
    return bool(expected and predicted and expected == predicted)


def judge_prompt(config: dict[str, Any], prediction: dict[str, Any]) -> str:
    return str(config["judge"]["prompt"]).format(
        question=str(prediction["prompt"]).replace("<image>", "").strip(),
        reference_answer=prediction["reference_answer"],
        model_answer=extract_answer_official(prediction["raw_model_answer"]),
    )


def judge_request_kwargs(*, model_id: str, prompt: str, config: dict[str, Any]) -> dict[str, Any]:
    generation = config["judge"]["generation"]
    return {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": int(generation["max_tokens"]),
        "extra_body": {
            "chat_template_kwargs": {
                "enable_thinking": bool(generation["enable_thinking"]),
            }
        },
    }


def normalize_judge_decision(raw: Any) -> str | None:
    value = str(raw or "").strip().casefold()
    if value == "yes":
        return "Yes"
    if value == "no":
        return "No"
    return None


def usage_value(response: Any, name: str) -> int:
    usage = getattr(response, "usage", None)
    return int(getattr(usage, name, 0) or 0) if usage is not None else 0


def finish_reason(response: Any) -> str | None:
    try:
        return str(response.choices[0].finish_reason or "") or None
    except Exception:
        return None


def append_session(path: Path, session: dict[str, Any]) -> None:
    append_jsonl(path, session)


def update_cost_from_sessions(out: Path, config: dict[str, Any]) -> None:
    session_path = out / "run_sessions.jsonl"
    sessions = []
    if session_path.is_file():
        for line in session_path.read_text(encoding="utf-8").splitlines():
            try:
                sessions.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    by_stage: dict[str, float] = {}
    for session in sessions:
        stage = str(session.get("stage") or "unknown")
        by_stage[stage] = by_stage.get(stage, 0.0) + float(session.get("wall_seconds") or 0)
    total_seconds = sum(by_stage.values())
    budget = config["budget"]
    hourly = float(
        budget.get("instance_cost_per_wall_hour", budget.get("dual_gpu_cost_per_wall_hour"))
    )
    write_json(
        out / config["paths"]["cost_name"],
        {
            "schema_version": 1,
            "updated_at_utc": now_utc(),
            "measurement_scope": "client_observed_inference_and_judge_sessions_only",
            "gpu_count": int(budget["gpu_count"]),
            "instance_cost_per_wall_hour_cny": hourly,
            "wall_seconds_by_stage": dict(sorted(by_stage.items())),
            "total_wall_hours": total_seconds / 3600.0,
            "estimated_cost_cny": total_seconds / 3600.0 * hourly,
            "excluded_time": "model_server_startup_shutdown and idle time",
        },
    )
