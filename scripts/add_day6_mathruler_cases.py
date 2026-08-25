#!/usr/bin/env python3
"""Append actual MathRuler-resolvable cases to the Day 6 calibration set."""

from __future__ import annotations

import json
from pathlib import Path

from eval.score_smoke import base_score
from scripts.day6_preflight import summarize_calibration


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    path = repo / "artifacts/runs/E-D6-001/preflight/judge_calibration.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if any(row["coverage_type"] == "mathruler_resolvable" for row in rows):
        summarize_calibration(repo, rows)
        return
    sources = [row for row in rows if row["coverage_type"] == "deterministic_numeric"]
    for source in sources:
        calibration_id = f"JC-{len(rows) + 1:03d}"
        model_answer = f"\\boxed{{{source['reference_answer']}}}"
        prediction = {
            "benchmark": "zoombench",
            "view": "calibration",
            "sample_uid": calibration_id,
            "question_format": "open_question",
            "official_category": "unavailable_official",
            "reference_answer": source["reference_answer"],
            "raw_model_answer": f"<answer>{model_answer}</answer>",
            "prompt": source["question"],
            "error": None,
        }
        score = base_score(prediction)
        if score["score_source"] != "mathruler" or not score["is_correct"]:
            raise RuntimeError(f"expected MathRuler resolution for {calibration_id}: {score}")
        rows.append({
            "schema_version": 1,
            "calibration_id": calibration_id,
            "sample_uid": source["sample_uid"],
            "question": source["question"],
            "reference_answer": source["reference_answer"],
            "model_answer": model_answer,
            "coverage_type": "mathruler_resolvable",
            "provisional_reference_label": True,
            "provisional_reviewer": "codex_semantic_review",
            "human_label": None,
            "human_reviewer": None,
            "human_reviewed_at_utc": None,
            "human_notes": None,
            "pipeline_score": score,
        })
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    summarize_calibration(repo, rows)


if __name__ == "__main__":
    main()
