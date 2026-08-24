#!/usr/bin/env python3
"""Build a portable technical report artifact from overlap audit outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_artifact(report: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    benchmark_rows = []
    for name, values in sorted(report["per_benchmark"].items()):
        benchmark_rows.append({
            "benchmark": name,
            "samples": values["benchmark_sample_count"],
            "candidates": values["candidate_pair_count"],
            "confirmed": values["confirmed_overlap_pair_count"],
            "dismissed": values["dismissed_pair_count"],
            "unresolved": values["unresolved_pair_count"],
            "impacted_rate": values["confirmed_impacted_rate"],
        })

    candidate_rows = []
    for item in candidates:
        review = item.get("manual_review", {})
        candidate_rows.append({
            "benchmark": item["benchmark"],
            "project_split": item["project_split"],
            "benchmark_sample": item["benchmark_sample_uid"],
            "match": ", ".join(item["match_types"]),
            "phash_distance": item["minimum_phash_distance"],
            "status": item["review_status"],
            "evidence": review.get("review_note", "Automatic exact-image confirmation."),
        })

    generated_at = report["generated_at_utc"]
    status = report["decision_status"]
    summary = (
        "## Decision\n\n"
        f"The audit status is **{status}**. Four VStar test images occur in the training split "
        "(4 of 191; 2.094%). MMStar has no detected candidates. The single ZoomBench exact-question "
        "candidate was dismissed because the answers differ and all image pHash distances are 32–36.\n\n"
        "**Implication:** retain official full-set scores, but do not describe VStar as fully independent. "
        "Report a VStar score excluding the four affected samples as a separate diagnostic."
    )
    method = (
        "## Definitions and method\n\n"
        "- **Exact image:** byte-identical SHA-256 across a project and benchmark image reference.\n"
        "- **Exact question:** NFKC normalization, Unicode casefold, and collapsed whitespace produce identical text.\n"
        "- **Perceptual candidate:** 64-bit DCT pHash Hamming distance at most 5 after EXIF transpose.\n"
        "- Both project full images and project crops were checked against benchmark full images and crops.\n"
        "- pHash-only and question-only candidates require a recorded manual decision.\n\n"
        "The audit covered 1,216 project samples, 2,536 benchmark samples, and 5,813 image references. "
        "There were no missing images, empty questions, duplicate sample IDs, or fingerprint errors."
    )
    limitations = (
        "## Limitations and next steps\n\n"
        "This is a strong practical overlap audit, not a proof that no semantic near-duplicate exists: pHash can miss "
        "heavy crops, viewpoint changes, composites, or materially edited copies. Candidate review used pixel statistics "
        "and source metadata; four VStar pairs share dimensions and pHash and differ only in sparse localized annotation regions.\n\n"
        "Next: add a four-sample VStar exclusion list to the diagnostic evaluator, keep the official 191-sample result, "
        "and rerun this audit whenever training data or frozen benchmark revisions change.\n\n"
        "The chart supports quick cross-benchmark comparison; the tables remain the source for exact counts and evidence."
    )

    source_id = "overlap_audit_outputs"
    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "Vision-OPD Benchmark Overlap Audit",
            "description": "Training-data overlap review for ZoomBench, MMStar, and VStar.",
            "generatedAt": generated_at,
            "charts": [{
                "id": "confirmed_overlap_chart",
                "title": "Confirmed benchmark overlap",
                "subtitle": "Confirmed affected test samples; official converted set sizes are 845, 1,500, and 191.",
                "type": "bar",
                "dataset": "benchmark_summary",
                "sourceId": source_id,
                "valueFormat": "number",
                "encodings": {
                    "x": {"field": "benchmark", "type": "nominal", "label": "Benchmark"},
                    "y": {"field": "confirmed", "type": "quantitative", "label": "Confirmed samples"},
                },
            }],
            "tables": [
                {
                    "id": "benchmark_summary",
                    "title": "Result by benchmark",
                    "subtitle": "Confirmed impact is measured against each official converted benchmark set.",
                    "dataset": "benchmark_summary",
                    "sourceId": source_id,
                    "columns": [
                        {"field": "benchmark", "label": "Benchmark", "type": "text"},
                        {"field": "samples", "label": "Samples", "format": "number"},
                        {"field": "candidates", "label": "Candidates", "format": "number"},
                        {"field": "confirmed", "label": "Confirmed", "format": "number"},
                        {"field": "dismissed", "label": "Dismissed", "format": "number"},
                        {"field": "unresolved", "label": "Unresolved", "format": "number"},
                        {"field": "impacted_rate", "label": "Confirmed impact", "format": "percent"},
                    ],
                },
                {
                    "id": "candidate_detail",
                    "title": "Candidate review evidence",
                    "subtitle": "Every candidate has a final decision; unresolved count is zero.",
                    "dataset": "candidate_detail",
                    "sourceId": source_id,
                    "columns": [
                        {"field": "benchmark", "label": "Benchmark", "type": "text"},
                        {"field": "project_split", "label": "Project split", "type": "text"},
                        {"field": "benchmark_sample", "label": "Benchmark sample", "type": "text"},
                        {"field": "match", "label": "Trigger", "type": "text"},
                        {"field": "phash_distance", "label": "pHash distance", "format": "number"},
                        {"field": "status", "label": "Decision", "type": "text"},
                        {"field": "evidence", "label": "Review evidence", "type": "text"},
                    ],
                },
            ],
            "sources": [{"id": source_id, "label": "Frozen local overlap audit outputs"}],
            "blocks": [
                {"id": "decision", "type": "markdown", "body": summary, "sourceId": source_id},
                {"id": "confirmed_chart", "type": "chart", "chartId": "confirmed_overlap_chart"},
                {"id": "benchmark_table", "type": "table", "tableId": "benchmark_summary"},
                {"id": "candidate_table", "type": "table", "tableId": "candidate_detail"},
                {"id": "method", "type": "markdown", "body": method, "sourceId": source_id},
                {"id": "limitations", "type": "markdown", "body": limitations, "sourceId": source_id},
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "benchmark_summary": benchmark_rows,
                "candidate_detail": candidate_rows,
            },
        },
        "sources": [{
            "id": source_id,
            "query": {
                "engine": "local_audit",
                "sql": "SELECT benchmark, samples, candidates, confirmed, dismissed, unresolved, confirmed / samples AS impacted_rate FROM overlap_audit_summary",
                "description": "SHA-256, normalized-question, and 64-bit DCT pHash audit with recorded manual decisions.",
                "executed_at": generated_at,
                "metric_definitions": {
                    "confirmed impact": "Confirmed benchmark samples divided by the official converted benchmark sample count."
                },
            },
        }],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    report = json.loads((input_dir / "overlap_report.json").read_text(encoding="utf-8"))
    candidates = [
        json.loads(line)
        for line in (input_dir / "overlap_candidates.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    output = Path(args.output) if args.output else input_dir / "report_artifact.json"
    output.write_text(
        json.dumps(build_artifact(report, candidates), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
