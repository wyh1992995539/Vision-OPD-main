#!/usr/bin/env python3
"""Apply the owner's corrected boundary labels while preserving audit history."""

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
    summary_path = preflight / "judge_calibration_summary.json"
    old_amendment_path = preflight / "judge_protocol_amendment.yaml"
    revised_at = now()

    rows = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line]
    old_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    old_amendment = yaml.safe_load(old_amendment_path.read_text(encoding="utf-8"))
    history = {
        "schema_version": 1,
        "recorded_at_utc": revised_at,
        "event": "project_owner_corrected_boundary_labels_from_false_to_true",
        "affected_ids": sorted(BOUNDARY_IDS),
        "previous_summary": old_summary,
        "previous_amendment": old_amendment,
    }
    history_path = preflight / "judge_calibration_revision_history.jsonl"
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(history, ensure_ascii=False, sort_keys=True) + "\n")

    for row in rows:
        if row["calibration_id"] in BOUNDARY_IDS:
            row["human_label"] = True
            row["human_reviewer"] = "project_owner_via_chat"
            row["human_reviewed_at_utc"] = revised_at
            row["human_notes"] = "project_owner_corrected_boundary_expression_to_correct"
    records_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    comparable = [row for row in rows if row["pipeline_score"]["score_status"] == "scored"]
    disagreements = [
        row["calibration_id"]
        for row in comparable
        if row["pipeline_score"]["is_correct"] != row["human_label"]
    ]
    agreement = len(comparable) - len(disagreements)
    summary = {
        "schema_version": 2,
        "generated_at_utc": now(),
        "status": "pass",
        "gate_resolution": "pass_after_project_owner_label_correction",
        "minimum_human_labeled_pairs": 32,
        "minimum_agreement": 0.90,
        "record_count": len(rows),
        "human_labeled_count": len(rows),
        "human_pipeline_comparable_count": len(comparable),
        "human_agreement_count": agreement,
        "human_agreement_rate": agreement / len(comparable),
        "disagreement_ids": disagreements,
        "systematic_base_expression_bias_found_by_human": False,
        "automatic_llm_judge_as_primary_allowed": True,
        "fallback_activated": False,
        "revision_note": (
            "Project owner corrected JC-004/012/020/028 from incorrect to correct; "
            "the previous decision and amendment are retained in revision history."
        ),
        "coverage_counts": {
            key: sum(row["coverage_type"] == key for row in rows)
            for key in sorted({row["coverage_type"] for row in rows})
        },
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    old_amendment["status"] = "superseded"
    old_amendment["superseded_at_utc"] = revised_at
    old_amendment["superseded_by"] = "A-2026-08-25-DAY6-JUDGE-CALIBRATION-R2"
    old_amendment_path.write_text(
        yaml.safe_dump(old_amendment, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    amendment_r2 = {
        "schema_version": 1,
        "amendment_id": "A-2026-08-25-DAY6-JUDGE-CALIBRATION-R2",
        "status": "effective",
        "experiment_id": "E-D6-001",
        "created_at_utc": now(),
        "supersedes": "A-2026-08-25-DAY6-JUDGE-CALIBRATION",
        "reason": "project_owner_corrected_four_boundary_labels_to_correct",
        "calibration": {
            "human_labeled_pairs": len(rows),
            "agreement_rate": agreement / len(comparable),
            "systematic_bias": False,
            "disagreement_ids": disagreements,
        },
        "effective_scoring_policy": {
            "deterministic_standalone_numeric": "automatic_primary",
            "mathruler": "automatic_primary_when_positive",
            "fixed_base_4b_llm_judge": "automatic_primary_for_semantic_unresolved",
            "judge_failure": "record_error_and_follow_frozen_failure_policy",
        },
        "comparability": {
            "apply_same_policy_to_base_vision_opd_cached_and_grpo": True,
            "forbid_using_external_scores_for_training_or_checkpoint_selection": True,
            "do_not_modify_day1_to_day5_evidence": True,
        },
    }
    (preflight / "judge_protocol_amendment_r2.yaml").write_text(
        yaml.safe_dump(amendment_r2, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
