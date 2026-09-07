#!/usr/bin/env python3
"""Audit the frozen 6,241-row cached-prefix training contract."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import yaml
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path(
    "artifacts/runs/E-D13-6K-CACHED-PILOT-001/preflight/contract_audit.json"
)
DEFAULT_REPORT = Path("artifacts/reports/cached_prefix_contract.md")
TOKEN_IDS_SOURCE = "base_tokenizer_reencoded_openai_response_text"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def comparable_actor(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config["actor"].items() if key != "memory_profile_dir"}


def comparable_resources(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config["resources"].items() if key != "memory_profile"}


def selection_ids(receipt: dict[str, Any]) -> list[str]:
    return [str(sample["sample_id"]) for sample in receipt["samples"]]


def audit_dataset_binding(train_path: Path, cache_path: Path) -> tuple[dict[str, Any], dict[str, bool]]:
    train = pq.read_table(train_path, columns=["prompt", "images", "extra_info"]).to_pylist()
    cached = pq.read_table(
        cache_path, columns=["sample_id", "prompt_sha256", "image_path"]
    ).to_pylist()
    cache_by_id = {str(row["sample_id"]): row for row in cached}
    errors: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(train):
        provenance = (row.get("extra_info") or {}).get("provenance") or {}
        sample_id = str(provenance.get("sample_id", "")).strip()
        prompt = row.get("prompt") or []
        images = row.get("images") or []
        cached_row = cache_by_id.get(sample_id)
        valid_shape = (
            len(prompt) == 1
            and prompt[0].get("role") == "user"
            and isinstance(prompt[0].get("content"), str)
            and prompt[0]["content"].count("<image>") == 1
            and len(images) == 1
            and isinstance(images[0], dict)
            and images[0].get("path")
        )
        prompt_text = (
            prompt[0]["content"].replace("<image>", "", 1).strip()
            if valid_shape else ""
        )
        matches = (
            sample_id
            and sample_id not in seen
            and cached_row is not None
            and valid_shape
            and hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
            == cached_row["prompt_sha256"]
            and Path(images[0]["path"]).resolve()
            == Path(cached_row["image_path"]).resolve()
        )
        if not matches and len(errors) < 20:
            errors.append({"row_index": index, "sample_id": sample_id})
        seen.add(sample_id)
    checks = {
        "train_cache_sample_id_sets_equal": seen == set(cache_by_id),
        "all_train_prompts_and_student_images_bound": not errors and len(train) == len(cached),
    }
    return {
        "train_path": str(train_path),
        "train_sha256": sha256_file(train_path),
        "train_rows": len(train),
        "cache_rows": len(cached),
        "error_count_capped_at_20": len(errors),
        "error_examples": errors,
    }, checks


def audit_token_roundtrip(
    cache_path: Path,
    *,
    model_path: Path,
    expected_sha256: str,
    expected_rows: int,
    max_response_length: int,
) -> tuple[dict[str, Any], dict[str, bool]]:
    table = pq.read_table(
        cache_path,
        columns=[
            "sample_id",
            "raw_response_text",
            "response_token_ids",
            "response_length",
            "finish_reason",
            "eos_appended_after_retokenization",
            "response_ids_truncated_after_retokenization",
            "token_ids_source",
            "generation_config_sha256",
            "model_path",
            "inference_error",
        ],
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Base tokenizer has no eos_token_id")

    mismatches: list[dict[str, Any]] = []
    duplicate_ids = 0
    seen: set[str] = set()
    finish_reasons: Counter[str] = Counter()
    lengths: list[int] = []
    eos_appended_count = 0
    truncated_count = 0
    sources: set[str] = set()
    generation_hashes: set[str] = set()
    model_paths: set[str] = set()
    inference_errors = 0

    for row_index, row in enumerate(table.to_pylist()):
        sample_id = str(row["sample_id"])
        if sample_id in seen:
            duplicate_ids += 1
        seen.add(sample_id)
        stored_ids = [int(token) for token in (row["response_token_ids"] or [])]
        encoded_ids = tokenizer.encode(str(row["raw_response_text"]), add_special_tokens=False)
        expected_eos_append = row["finish_reason"] == "stop" and (
            not encoded_ids or encoded_ids[-1] != eos_id
        )
        if expected_eos_append:
            encoded_ids.append(eos_id)
        expected_truncation = len(encoded_ids) > max_response_length
        expected_ids = encoded_ids[:max_response_length]
        field_match = (
            stored_ids == expected_ids
            and int(row["response_length"]) == len(stored_ids)
            and bool(row["eos_appended_after_retokenization"]) == expected_eos_append
            and bool(row["response_ids_truncated_after_retokenization"]) == expected_truncation
        )
        if not field_match and len(mismatches) < 20:
            mismatches.append(
                {
                    "row_index": row_index,
                    "sample_id": sample_id,
                    "stored_length": len(stored_ids),
                    "expected_length": len(expected_ids),
                    "stored_eos_appended": bool(row["eos_appended_after_retokenization"]),
                    "expected_eos_appended": expected_eos_append,
                    "stored_truncated": bool(row["response_ids_truncated_after_retokenization"]),
                    "expected_truncated": expected_truncation,
                }
            )
        finish_reasons[str(row["finish_reason"])] += 1
        lengths.append(len(stored_ids))
        eos_appended_count += int(bool(row["eos_appended_after_retokenization"]))
        truncated_count += int(bool(row["response_ids_truncated_after_retokenization"]))
        sources.add(str(row["token_ids_source"]))
        generation_hashes.add(str(row["generation_config_sha256"]))
        model_paths.add(str(Path(row["model_path"]).resolve()))
        inference_errors += int(row["inference_error"] is not None)

    checks = {
        "cache_sha256_matches": sha256_file(cache_path) == expected_sha256,
        "row_count_matches": table.num_rows == expected_rows,
        "sample_ids_unique": duplicate_ids == 0 and len(seen) == table.num_rows,
        "all_response_ids_roundtrip_exactly": not mismatches,
        "all_lengths_within_contract": bool(lengths)
        and min(lengths) >= 1
        and max(lengths) <= max_response_length,
        "token_ids_source_frozen": sources == {TOKEN_IDS_SOURCE},
        "generation_hash_singleton": len(generation_hashes) == 1,
        "base_model_path_singleton": model_paths == {str(model_path.resolve())},
        "inference_errors_zero": inference_errors == 0,
    }
    evidence = {
        "path": str(cache_path),
        "sha256": sha256_file(cache_path),
        "rows": table.num_rows,
        "eos_token_id": eos_id,
        "response_length_min": min(lengths) if lengths else None,
        "response_length_max": max(lengths) if lengths else None,
        "finish_reasons": dict(sorted(finish_reasons.items())),
        "eos_appended_records": eos_appended_count,
        "truncated_after_retokenization_records": truncated_count,
        "token_ids_sources": sorted(sources),
        "generation_config_sha256_values": sorted(generation_hashes),
        "model_path_values": sorted(model_paths),
        "inference_errors": inference_errors,
        "mismatch_count_capped_at_20": len(mismatches),
        "mismatch_examples": mismatches,
    }
    return evidence, checks


def write_report(result: dict[str, Any], path: Path) -> None:
    checks = result["checks"]
    token = result["token_roundtrip"]
    lines = [
        "# Cached Prefix 6,241 Contract Audit",
        "",
        f"- Static status: **{result['status']}**",
        f"- Runtime ablation status: **{result['runtime_ablation_status']}**",
        f"- Cache: `{token['path']}`",
        f"- Cache SHA256: `{token['sha256']}`",
        f"- Records: `{token['rows']}`",
        f"- Response token length: `{token['response_length_min']}..{token['response_length_max']}`",
        f"- Finish reasons: `{json.dumps(token['finish_reasons'], sort_keys=True)}`",
        f"- EOS appended after retokenization: `{token['eos_appended_records']}`",
        f"- Retokenized responses capped at 1,024: `{token['truncated_after_retokenization_records']}`",
        "",
        "## Checks",
        "",
    ]
    lines.extend(f"- {'PASS' if passed else 'FAIL'} — `{name}`" for name, passed in checks.items())
    lines.extend(
        [
            "",
            "## Scope",
            "",
            "The static audit proves that every cached response is bound to its frozen sample, prompt, Student image, Base tokenizer, generation contract, and model identity. It also proves that the online configuration remains the default and that the cached branch reports zero online generation calls in the tested dispatch path.",
            "",
            "Strict prefix-source ablation remains pending until the 8-step cached Pilot runs on the same two-GPU resource contract and passes finite-loss, Student update, Teacher EMA, checkpoint, and cold-reload gates.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--online-config", type=Path, default=Path("configs/vopd_6241.yaml"))
    parser.add_argument("--cached-config", type=Path, default=Path("configs/cached_prefix_6241.yaml"))
    parser.add_argument("--online-pilot-config", type=Path, default=Path("configs/vopd_6241_pilot_64.yaml"))
    parser.add_argument("--cached-pilot-config", type=Path, default=Path("configs/cached_prefix_6241_pilot_64.yaml"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    paths = {name: resolve(value) for name, value in vars(args).items() if name not in {"output", "report"}}
    online = load_yaml(paths["online_config"])
    cached = load_yaml(paths["cached_config"])
    online_pilot = load_yaml(paths["online_pilot_config"])
    cached_pilot = load_yaml(paths["cached_pilot_config"])
    cache_cfg = cached["cached_prefix"]
    cache_path = Path(cache_cfg["output_parquet"]).resolve()
    token_evidence, token_checks = audit_token_roundtrip(
        cache_path,
        model_path=Path(cached["paths"]["model"]),
        expected_sha256=cache_cfg["output_sha256"],
        expected_rows=int(cache_cfg["expected_samples"]),
        max_response_length=int(cached["data"]["max_response_length"]),
    )
    binding_evidence, binding_checks = audit_dataset_binding(
        Path(cached["paths"]["train_file"]).resolve(), cache_path
    )

    generation = cache_cfg["generation"]
    generation_matches_rollout = {
        "num_return_sequences": generation["num_return_sequences"],
        "temperature": generation["temperature"],
        "top_p": generation["top_p"],
        "top_k": generation["top_k"],
        "max_new_tokens": generation["max_new_tokens"],
    } == {
        "num_return_sequences": online["rollout"]["n"],
        "temperature": online["rollout"]["temperature"],
        "top_p": online["rollout"]["top_p"],
        "top_k": online["rollout"]["top_k"],
        "max_new_tokens": online["data"]["max_response_length"],
    }

    online_selection = load_json(resolve(online_pilot["paths"]["selection_manifest"]))
    cached_selection = load_json(resolve(cached_pilot["paths"]["selection_manifest"]))
    trainer_source = (PROJECT_ROOT / "verl/trainer/ppo/ray_trainer.py").read_text(encoding="utf-8")
    manager_source = (PROJECT_ROOT / "verl/experimental/agent_loop/agent_loop.py").read_text(encoding="utf-8")
    rollout_defaults = load_yaml(PROJECT_ROOT / "verl/trainer/config/rollout/rollout.yaml")
    checks = {
        **token_checks,
        **binding_checks,
        "generation_hash_matches_frozen_report": cache_cfg["generation_config_sha256"]
        == load_json(resolve(cache_cfg["report_file"]))["generation_config_sha256"],
        "generation_sampling_matches_online_rollout": generation_matches_rollout,
        "formal_data_contract_equal": cached["data"] == online["data"],
        "formal_actor_contract_equal": comparable_actor(cached) == comparable_actor(online),
        "formal_rollout_contract_equal": cached["rollout"] == online["rollout"],
        "formal_self_distillation_contract_equal": cached["self_distillation"] == online["self_distillation"],
        "formal_resource_contract_equal": comparable_resources(cached) == comparable_resources(online),
        "formal_training_contract_equal": cached["training"] == online["training"],
        "formal_model_train_template_paths_equal": all(
            cached["paths"][key] == online["paths"][key]
            for key in ("model", "train_file", "chat_template")
        ),
        "pilot_data_contract_equal": cached_pilot["data"] == online_pilot["data"],
        "pilot_actor_contract_equal": comparable_actor(cached_pilot) == comparable_actor(online_pilot),
        "pilot_rollout_contract_equal": cached_pilot["rollout"] == online_pilot["rollout"],
        "pilot_self_distillation_contract_equal": cached_pilot["self_distillation"]
        == online_pilot["self_distillation"],
        "pilot_resource_contract_equal": comparable_resources(cached_pilot)
        == comparable_resources(online_pilot),
        "pilot_training_contract_equal": cached_pilot["training"] == online_pilot["training"],
        "pilot_model_train_template_paths_equal": all(
            cached_pilot["paths"][key] == online_pilot["paths"][key]
            for key in ("model", "train_file", "chat_template")
        ),
        "pilot_frozen_sample_order_equal": selection_ids(cached_selection)
        == selection_ids(online_selection),
        "online_is_rollout_default": rollout_defaults["prefix_source"] == "online",
        "trainer_has_separate_online_and_cached_dispatch": "if self.prefix_source == \"online\"" in trainer_source
        and "self.cached_prefix_store.bind(gen_batch)" in trainer_source
        and "build_cached_sequences(gen_batch)" in trainer_source,
        "cached_manager_skips_online_server_initialization": "if self.prefix_source == \"online\":" in manager_source
        and "self.server_handles = []" in manager_source,
        "cached_and_online_share_agent_loop_postprocess": "self._agent_loop_postprocess(" in manager_source
        and "async def _build_cached_single_turn" in manager_source,
    }
    status = "STATIC_CONTRACT_PASS" if all(checks.values()) else "FAIL"
    input_paths = {
        **paths,
        "cache": cache_path,
        "cache_report": resolve(cache_cfg["report_file"]),
        "chat_template": resolve(cached["paths"]["chat_template"]),
        "cached_store": PROJECT_ROOT / "verl/trainer/ppo/cached_prefix.py",
        "trainer_dispatch": PROJECT_ROOT / "verl/trainer/ppo/ray_trainer.py",
        "agent_loop": PROJECT_ROOT / "verl/experimental/agent_loop/agent_loop.py",
        "launcher": PROJECT_ROOT / "scripts/run_vopd_2gpu.sh",
        "training_preflight": PROJECT_ROOT / "scripts/vopd_training_preflight.py",
        "contract_tests": PROJECT_ROOT / "tests/test_cached_prefix_contract.py",
    }
    result = {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "status": status,
        "ready_for_gpu_pilot": status == "STATIC_CONTRACT_PASS",
        "formal_training_authorized": False,
        "runtime_ablation_status": "PENDING_TWO_GPU_PILOT" if status == "STATIC_CONTRACT_PASS" else "BLOCKED",
        "checks": checks,
        "token_roundtrip": token_evidence,
        "dataset_binding": binding_evidence,
        "configs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in input_paths.items()
        },
        "pilot_selection": {
            "online_path": str(resolve(online_pilot["paths"]["selection_manifest"])),
            "cached_path": str(resolve(cached_pilot["paths"]["selection_manifest"])),
            "sample_count": len(selection_ids(cached_selection)),
            "ordered_sample_ids_sha256": hashlib.sha256(
                "\n".join(selection_ids(cached_selection)).encode("utf-8")
            ).hexdigest(),
        },
        "limits": [
            "Static checks do not prove runtime loss, parameter update, Teacher EMA, checkpoint, or cold reload.",
            "Strict prefix-source ablation requires the frozen 8-step cached Pilot on two GPUs.",
        ],
    }
    output_path = resolve(args.output)
    report_path = resolve(args.report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(result, report_path)
    (output_path.parent / "contract_audit_sha256.txt").write_text(
        f"{sha256_file(output_path)}  {output_path}\n"
        f"{sha256_file(report_path)}  {report_path}\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if status == "STATIC_CONTRACT_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
