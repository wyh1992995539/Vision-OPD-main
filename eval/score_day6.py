#!/usr/bin/env python3
"""Score E-D6-001 predictions and emit official-only V* 191-sample summaries."""
from __future__ import annotations

import argparse, json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from eval.run_day6 import EXPECTED, load_amendment, read_jsonl, write_json
from eval.run_smoke import record_key, resolve_path, sha256_file
from eval.score_smoke import base_score, call_judge, score_key, write_jsonl


def summarize(scores: list[dict[str, Any]], amendment: dict[str, Any]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    categories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for score in scores:
        group = f"{score['benchmark']}/{score['view']}"; groups[group].append(score)
        if score["benchmark"] in {"mmstar", "vstar"}: categories[f"{group}/{score['official_category']}"].append(score)
    def stat(items: list[dict[str, Any]]) -> dict[str, Any]:
        pending = sum(x["score_status"] != "scored" for x in items); correct = sum(bool(x["is_correct"]) for x in items)
        return {"total": len(items), "correct": correct, "incorrect": len(items)-correct-pending, "pending_judge": pending, "accuracy": correct/len(items) if items and not pending else None}
    result = {"schema_version": 1, "experiment_id": "E-D6-001", "generated_at_utc": datetime.now(timezone.utc).isoformat(), "decision_status": "complete" if all(x["score_status"] == "scored" for x in scores) else "pending_judge", "request_count": len(scores), "groups": {key: stat(value) for key,value in sorted(groups.items())}, "official_category_groups": {key: stat(value) for key,value in sorted(categories.items())}, "vstar_reporting_policy": amendment["effective_reporting_policy"], "deduplicated_diagnostic": None}
    if result["groups"].get("vstar/full", {}).get("total") != 191:
        raise ValueError("V* official summary denominator must be exactly 191")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Explicit legacy E-D5/E-D6 config path; R3 uses the paper-aligned entrypoints")
    parser.add_argument("--amendment", default="artifacts/runs/E-D6-001/preflight/vstar_reporting_amendment.yaml")
    parser.add_argument("--input-dir", default="artifacts/runs/E-D6-001/base")
    parser.add_argument("--judge-api-base"); parser.add_argument("--judge-api-key", default="EMPTY"); parser.add_argument("--judge-model-id"); parser.add_argument("--judge-workers", type=int, default=4)
    args = parser.parse_args(); repo = Path(__file__).resolve().parent.parent
    config_path = resolve_path(args.config, repo); amendment_path = resolve_path(args.amendment, repo); out = resolve_path(args.input_dir, repo)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")); amendment = load_amendment(amendment_path)
    predictions = list(read_jsonl(out / "predictions.jsonl").values())
    expected = sum(EXPECTED.values())
    if len(predictions) != expected: raise ValueError(f"expected {expected} unique predictions, found {len(predictions)}")
    scores = [base_score(item) for item in predictions]; by_key = {record_key(item): item for item in predictions}
    pending = [(by_key[score_key(score)], score) for score in scores if score["score_status"] == "pending_judge"]
    if pending and args.judge_api_base:
        model = args.judge_model_id or str(config["judge"]["model"]["served_model_name"])
        if model != str(config["judge"]["model"]["served_model_name"]): raise ValueError("judge model does not match frozen protocol")
        call_judge(pending, config=config, api_base=args.judge_api_base, api_key=args.judge_api_key, model_id=model, workers=args.judge_workers)
    write_jsonl(out / "scores.jsonl", scores)
    summary = summarize(scores, amendment); summary.update({"config_sha256_raw_bytes": sha256_file(config_path), "amendment_sha256_raw_bytes": sha256_file(amendment_path), "resume_gate": {"unique_request_keys": len(predictions), "duplicate_request_keys": 0}})
    write_json(out / "summary.json", summary); print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__": main()
