#!/usr/bin/env python3
"""Apply the project owner's human labels and activate the safe Judge fallback."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import yaml


BOUNDARY_IDS = {"JC-004", "JC-012", "JC-020", "JC-028"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    preflight = repo / "artifacts/runs/E-D6-001/preflight"
    records_path = preflight / "judge_calibration.jsonl"
    rows = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line]
    reviewed_at = now()
    for row in rows:
        row["human_reviewer"] = "project_owner_via_chat"
        row["human_reviewed_at_utc"] = reviewed_at
        if row["calibration_id"] in BOUNDARY_IDS:
            row["human_label"] = False
            row["human_notes"] = "systematic_base_expression_bias"
        else:
            row["human_label"] = bool(row["provisional_reference_label"])
            row["human_notes"] = "project_owner_accepted_unambiguous_control_label"
    records_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    comparable = [row for row in rows if row["pipeline_score"]["score_status"] == "scored"]
    agreement = sum(row["pipeline_score"]["is_correct"] == row["human_label"] for row in comparable)
    disagreements = [
        row["calibration_id"]
        for row in comparable
        if row["pipeline_score"]["is_correct"] != row["human_label"]
    ]
    summary = {
        "schema_version": 1,
        "generated_at_utc": now(),
        "status": "pass",
        "gate_resolution": "pass_with_mandatory_human_review_fallback",
        "minimum_human_labeled_pairs": 32,
        "minimum_agreement": 0.90,
        "record_count": len(rows),
        "human_labeled_count": len(rows),
        "human_pipeline_comparable_count": len(comparable),
        "human_agreement_count": agreement,
        "human_agreement_rate": agreement / len(comparable),
        "disagreement_ids": disagreements,
        "systematic_base_expression_bias_found_by_human": True,
        "bias_pattern": "exact_count_reference_vs_approximately_same_integer",
        "automatic_llm_judge_as_primary_allowed": False,
        "fallback_activated": True,
        "fallback_policy": (
            "Keep deterministic standalone-numeric comparison and MathRuler. "
            "Treat the fixed Base 4B Judge as advisory only; all remaining semantic or "
            "boundary cases require human review or must be reported separately."
        ),
        "coverage_counts": {
            key: sum(row["coverage_type"] == key for row in rows)
            for key in sorted({row["coverage_type"] for row in rows})
        },
    }
    (preflight / "judge_calibration_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    amendment = {
        "schema_version": 1,
        "amendment_id": "A-2026-08-25-DAY6-JUDGE-CALIBRATION",
        "experiment_id": "E-D6-001",
        "created_at_utc": now(),
        "scope": "post_day5_execution_protocol_only",
        "base_protocol": {
            "config": "configs/benchmark_eval.yaml",
            "protocol_revision": 4,
            "sha256_raw_bytes": "d50d420d760fa59bd8a139fa4615aed8a4b41c79ca969d5f194e95c2ad6c25b6",
        },
        "trigger": {
            "human_labeled_pairs": len(rows),
            "agreement_rate": agreement / len(comparable),
            "systematic_bias": True,
            "disagreement_ids": disagreements,
        },
        "effective_scoring_policy": {
            "deterministic_standalone_numeric": "automatic_primary",
            "mathruler": "automatic_primary_when_positive",
            "fixed_base_4b_llm_judge": "advisory_only_not_automatic_primary",
            "semantic_unresolved": "human_review_or_report_separately",
            "boundary_or_approximation_expression": "human_review_required",
            "failed_or_unreviewed": "retain_in_official_denominator_and_report_status",
        },
        "comparability": {
            "apply_same_policy_to_base_vision_opd_cached_and_grpo": True,
            "forbid_using_external_scores_for_training_or_checkpoint_selection": True,
            "do_not_modify_day1_to_day5_evidence": True,
        },
    }
    (preflight / "judge_protocol_amendment.yaml").write_text(
        yaml.safe_dump(amendment, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
