#!/usr/bin/env python3
"""Build and validate the Day 6 comparability preflight artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from eval.score_smoke import base_score, call_judge, normalize_judge_output


TEXT_EXTENSIONS = {".md", ".yaml", ".json", ".jsonl", ".txt"}
MODEL_SHARDS = {
    "model.safetensors-00001-of-00002.safetensors":
        "26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61",
    "model.safetensors-00002-of-00002.safetensors":
        "cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def raw_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_lf_sha256(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.replace("\r\n", "\n").replace("\r", "\n"), encoding="utf-8")


def write_json(path: Path, value: Any) -> None:
    write_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def git_commit(repo: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()


def build_lock(repo: Path) -> Path:
    project = yaml.safe_load((repo / "configs/project_1024.yaml").read_text(encoding="utf-8"))
    benchmark = yaml.safe_load((repo / "configs/benchmark_eval.yaml").read_text(encoding="utf-8"))
    data_root = Path("/root/autodl-tmp/data/vision_opd_1024")
    model_root = Path(project["model"]["path"])
    lock = {
        "schema_version": 1,
        "experiment_id": "E-D6-001",
        "created_at_utc": utc_now(),
        "git_commit_at_lock": git_commit(repo),
        "status": "frozen_before_day6_full_external_scores",
        "model": {
            "name": project["model"]["name"],
            "path": str(model_root),
            "role": "shared_untrained_base_for_vision_opd_and_cached_prefix",
            "weight_shards_sha256": MODEL_SHARDS,
            "forbid_official_vision_opd_weights_as_base": True,
            "forbid_serial_branch_inheritance": True,
        },
        "data": {
            "source_revision": project["data"]["source_revision"],
            "splits": {
                "train_1024": {
                    "path": str(data_root / "train_1024.parquet"),
                    "sha256_raw_bytes": raw_sha256(data_root / "train_1024.parquet"),
                },
                "eval_128": {
                    "path": str(data_root / "eval_128.parquet"),
                    "sha256_raw_bytes": raw_sha256(data_root / "eval_128.parquet"),
                    "usage": "internal_checkpoint_selection_only",
                },
                "retention_64": {
                    "path": str(data_root / "retention_64.parquet"),
                    "sha256_raw_bytes": raw_sha256(data_root / "retention_64.parquet"),
                    "usage": "internal_retention_and_format_guardrail",
                },
            },
        },
        "reproducibility": {
            "master_seed": 42,
            "data_split_seed": 42,
            "dataloader_seed": 42,
            "training_seed": 42,
            "rollout_seed": 42,
            "require_explicit_seed_in_every_experiment": True,
        },
        "chat_and_views": {
            "chat_template": "chat_templates/perception_chat_template_qwen35.jinja",
            "chat_template_sha256_raw_bytes": raw_sha256(
                repo / "chat_templates/perception_chat_template_qwen35.jinja"
            ),
            "student_view": "full_red_box_image",
            "teacher_view": "crop_image",
            "deployment_view": "student_full_red_box_image_only",
        },
        "shared_training_design": {
            "epoch": 1,
            "global_batch": 8,
            "estimated_optimizer_steps": 128,
            "rollout_n": 1,
            "max_prompt_length": {
                "starting_candidate": 8192,
                "final_value_requires_day7_processor_p99_and_max_gate": True,
                "forbid_default_4096": True,
                "forbid_silent_truncation": True,
            },
            "max_response_length": 256,
            "learning_rate": 2e-6,
            "top_k": 100,
            "jsd_alpha": 0.5,
            "jsd_beta": 0.5,
            "ema_update_rate": 0.05,
            "gpu_per_node": 2,
            "node_count": 1,
        },
        "single_variable_ablation": {
            "vision_opd": {"prefix_source": "online"},
            "cached_prefix": {"prefix_source": "cached"},
            "all_other_model_data_loss_and_eval_fields_must_match": True,
            "cached_label_requires_day14_contract_pass": True,
        },
        "internal_evaluation": {
            "evaluator_version_file": "artifacts/eval/evaluator_version.json",
            "evaluator_id": "e40cc751a2b732b5cd8eaeb4f4ca61754c75a3286800ab44bf4a2e72dbe7c689",
            "generation_temperature": 0,
            "unsupported_policy": "mark_unsupported",
            "invalid_prediction_policy": "count_as_incorrect_in_full_denominator",
        },
        "checkpoint_selection": {
            "vision_opd_day12": "select_and_freeze_using_only_internal_eval_128_and_retention_64",
            "cached_prefix_day18": "select_and_freeze_using_only_internal_eval_128_and_retention_64",
            "forbid_external_benchmark_score_for_selection_or_retraining": True,
            "external_evaluation_after_both_branches_frozen": "E-D18-002",
        },
        "external_result_firewall": {
            "lock_and_sha256_required_before_opening_day6_full_summary": True,
            "day6_base_full_external_allowed_after_all_preflight_gates_pass": True,
            "vision_opd_external_results_forbidden_on_day12": True,
            "vision_opd_and_cached_external_results_deferred_to_day18": True,
            "external_scores_must_not_change_learning_rate_epoch_prompt_loss_or_checkpoint": True,
        },
        "allowed_post_lock_changes": {
            "only_after_real_training_smoke": [
                "micro_batch_size",
                "gradient_accumulation_preserving_global_batch",
                "memory_offload",
                "max_prompt_length_from_measured_processor_distribution",
                "throughput_and_oom_parameters",
            ],
            "require_amendment_with_smoke_evidence": True,
            "forbid_external_accuracy_as_evidence": True,
        },
        "day6_external_protocol": {
            "config": "configs/benchmark_eval.yaml",
            "protocol_revision": benchmark["protocol"]["protocol_revision"],
            "config_sha256_raw_bytes": raw_sha256(repo / "configs/benchmark_eval.yaml"),
            "benchmarks": ["zoombench", "mmstar", "vstar"],
        },
    }
    path = repo / "artifacts/runs/E-D6-001/preflight/training_design_lock.yaml"
    write_text(path, yaml.safe_dump(lock, allow_unicode=True, sort_keys=False))
    digest = canonical_lf_sha256(path)
    write_text(
        path.with_suffix(".sha256"),
        f"{digest}  canonical_lf  {path.relative_to(repo)}\n",
    )
    return path


def controlled_cases(dataset: list[dict[str, Any]]) -> list[dict[str, Any]]:
    numeric = [
        item for item in dataset
        if item.get("question_format") == "open_question" and str(item.get("response", "")).isdigit()
    ][:8]
    if len(numeric) < 8:
        raise ValueError("need at least eight numeric ZoomBench open questions")
    cases: list[dict[str, Any]] = []
    for index, item in enumerate(numeric):
        answer = int(item["response"])
        variants = [
            ("deterministic_numeric", str(answer), True),
            ("clear_numeric_error", str(answer + 1), False),
            ("semantic_equivalent", f"There are {answer} in total.", True),
            (
                "boundary_expression" if index % 2 == 0 else "clear_semantic_error",
                f"Approximately {answer}." if index % 2 == 0 else f"There are {answer + 2} in total.",
                True if index % 2 == 0 else False,
            ),
        ]
        for variant, prediction, label in variants:
            cases.append({
                "schema_version": 1,
                "calibration_id": f"JC-{len(cases) + 1:03d}",
                "sample_uid": item["sample_uid"],
                "question": item["query"],
                "reference_answer": item["response"],
                "model_answer": prediction,
                "coverage_type": variant,
                "provisional_reference_label": label,
                "provisional_reviewer": "codex_semantic_review",
                "human_label": None,
                "human_reviewer": None,
                "human_reviewed_at_utc": None,
                "human_notes": None,
            })
    return cases


def score_calibration(repo: Path, api_base: str | None, workers: int) -> Path:
    output = repo / "artifacts/runs/E-D6-001/preflight/judge_calibration.jsonl"
    if output.is_file():
        rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line]
    else:
        dataset_path = Path("/root/autodl-tmp/benchmark_data/converted/zoombench/zoombench.json")
        rows = controlled_cases(json.loads(dataset_path.read_text(encoding="utf-8")))
    config = yaml.safe_load((repo / "configs/benchmark_eval.yaml").read_text(encoding="utf-8"))
    pending: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for row in rows:
        prediction = {
            "benchmark": "zoombench",
            "view": "calibration",
            "sample_uid": row["calibration_id"],
            "question_format": "open_question",
            "official_category": "unavailable_official",
            "reference_answer": row["reference_answer"],
            "raw_model_answer": f"<answer>{row['model_answer']}</answer>",
            "prompt": row["question"],
            "error": None,
        }
        score = base_score(prediction)
        if score["score_status"] == "pending_judge":
            pending.append((prediction, score))
        row["pipeline_score"] = score
    if pending and api_base:
        call_judge(
            pending,
            config=config,
            api_base=api_base,
            api_key="EMPTY",
            model_id=config["judge"]["model"]["served_model_name"],
            workers=workers,
        )
    write_text(output, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))
    summarize_calibration(repo, rows)
    return output


def summarize_calibration(repo: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    human = [row for row in rows if isinstance(row.get("human_label"), bool)]
    provisional_decided = [
        row for row in rows
        if row.get("pipeline_score", {}).get("score_status") == "scored"
    ]
    human_agree = sum(
        row["pipeline_score"].get("is_correct") == row["human_label"]
        for row in human
        if row.get("pipeline_score", {}).get("score_status") == "scored"
    )
    human_comparable = sum(
        row.get("pipeline_score", {}).get("score_status") == "scored" for row in human
    )
    provisional_agree = sum(
        row["pipeline_score"].get("is_correct") == row["provisional_reference_label"]
        for row in provisional_decided
    )
    systematic_bias = any(
        row.get("human_notes") == "systematic_base_expression_bias" for row in human
    )
    human_rate = human_agree / human_comparable if human_comparable else None
    status = (
        "pass"
        if len(human) >= 32 and human_comparable == len(human) and human_rate is not None
        and human_rate >= 0.90 and not systematic_bias
        else "pending_human_review"
    )
    summary = {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "status": status,
        "minimum_human_labeled_pairs": 32,
        "minimum_agreement": 0.90,
        "record_count": len(rows),
        "human_labeled_count": len(human),
        "human_pipeline_comparable_count": human_comparable,
        "human_agreement_count": human_agree,
        "human_agreement_rate": human_rate,
        "provisional_codex_review_count": len(rows),
        "provisional_pipeline_decided_count": len(provisional_decided),
        "provisional_agreement_rate": (
            provisional_agree / len(provisional_decided) if provisional_decided else None
        ),
        "pending_pipeline_count": len(rows) - len(provisional_decided),
        "systematic_base_expression_bias_found_by_human": systematic_bias,
        "coverage_counts": {
            key: sum(row["coverage_type"] == key for row in rows)
            for key in sorted({row["coverage_type"] for row in rows})
        },
        "decision_note": (
            "Codex provisional labels are not human labels. A human must review all 32 rows, "
            "populate human_label/human_reviewer, then rerun summarize before this Gate can PASS."
        ),
    }
    write_json(repo / "artifacts/runs/E-D6-001/preflight/judge_calibration_summary.json", summary)
    return summary


def build_hash_manifest(repo: Path) -> Path:
    preflight = repo / "artifacts/runs/E-D6-001/preflight"
    tracked = subprocess.check_output(["git", "ls-files"], cwd=repo, text=True).splitlines()
    rows = []
    for relative in tracked:
        path = repo / relative
        if path.suffix.casefold() in TEXT_EXTENSIONS:
            rows.append({
                "path": relative,
                "algorithm": "sha256",
                "content_rule": "canonical_lf_utf8",
                "sha256": canonical_lf_sha256(path),
            })
    for path in sorted(preflight.glob("*")):
        if not path.is_file() or path.name in {"hash_manifest.json", "hash_manifest.sha256"}:
            continue
        rule = "canonical_lf_utf8" if path.suffix.casefold() in TEXT_EXTENSIONS else "raw_bytes"
        rows.append({
            "path": str(path.relative_to(repo)),
            "algorithm": "sha256",
            "content_rule": rule,
            "sha256": canonical_lf_sha256(path) if rule == "canonical_lf_utf8" else raw_sha256(path),
        })
    manifest = preflight / "hash_manifest.json"
    write_json(manifest, {"schema_version": 1, "generated_at_utc": utc_now(), "files": rows})
    write_text(
        preflight / "hash_manifest.sha256",
        f"{canonical_lf_sha256(manifest)}  canonical_lf  {manifest.relative_to(repo)}\n",
    )
    return manifest


def validate(repo: Path) -> dict[str, Any]:
    preflight = repo / "artifacts/runs/E-D6-001/preflight"
    lock = preflight / "training_design_lock.yaml"
    lock_hash = preflight / "training_design_lock.sha256"
    calibration_summary_path = preflight / "judge_calibration_summary.json"
    environment_summary_path = preflight / "environment_summary.json"
    manifest = preflight / "hash_manifest.json"
    calibration = json.loads(calibration_summary_path.read_text(encoding="utf-8")) if calibration_summary_path.is_file() else {}
    environment = json.loads(environment_summary_path.read_text(encoding="utf-8")) if environment_summary_path.is_file() else {}
    gates = {
        "training_design_lock": lock.is_file() and lock_hash.is_file(),
        "judge_calibration": calibration.get("status") == "pass",
        "server_environment": environment.get("status") == "pass",
        "canonical_hash": manifest.is_file(),
    }
    result = {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "experiment_id": "E-D6-001",
        "gates": gates,
        "status": "pass" if all(gates.values()) else "blocked",
        "blocking_gates": [key for key, passed in gates.items() if not passed],
        "day6_full_external_evaluation_authorized": all(gates.values()),
    }
    write_json(preflight / "preflight_summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["lock", "calibration", "hashes", "validate", "all"])
    parser.add_argument("--repo", default=".")
    parser.add_argument("--judge-api-base")
    parser.add_argument("--judge-workers", type=int, default=4)
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    if args.command in {"lock", "all"}:
        build_lock(repo)
    if args.command in {"calibration", "all"}:
        score_calibration(repo, args.judge_api_base, args.judge_workers)
    if args.command in {"hashes", "all"}:
        build_hash_manifest(repo)
    if args.command in {"validate", "all"}:
        print(json.dumps(validate(repo), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
