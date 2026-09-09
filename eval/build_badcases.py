#!/usr/bin/env python3
"""Select deterministic Day15 bad cases from frozen, completed R3 scores."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def request_key(item: dict[str, Any]) -> str:
    return "\0".join((str(item["benchmark"]), str(item.get("view", "full")), str(item["sample_uid"])))


def read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        key = request_key(item)
        if key in result:
            raise ValueError(f"duplicate request key in {path}: {key!r}")
        result[key] = item
    return result


def stratum(base: bool, vision: bool, cached: bool) -> str | None:
    mapping = {
        (False, True, True): "both_trained_fix",
        (False, True, False): "vision_only_fix_cached_regression",
        (False, False, True): "cached_only_fix",
        (True, False, False): "both_trained_regression",
        (True, False, True): "vision_regression_cached_recovers",
        (True, True, False): "cached_regression",
    }
    return mapping.get((base, vision, cached))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/day15_final_eval_freeze.yaml")
    args = parser.parse_args()
    config_path = resolve(args.config)
    config = yaml.safe_load(config_path.read_text())
    paths = {
        "base": resolve(config["base_run"]),
        "vision_opd": resolve(config["models"]["vision_opd"]["formal_output"]),
        "cached_prefix": resolve(config["models"]["cached_prefix"]["formal_output"]),
    }
    scores = {role: read_jsonl(path / "scores.jsonl") for role, path in paths.items()}
    predictions = {role: read_jsonl(path / "predictions.jsonl") for role, path in paths.items()}
    keys = set(scores["base"])
    if any(set(rows) != keys for rows in (*scores.values(), *predictions.values())):
        raise ValueError("all score and prediction key sets must be identical")
    expected = int(config["output_schema"]["expected_total"])
    if len(keys) != expected:
        raise ValueError(f"expected {expected} keys, found {len(keys)}")

    policy = config["badcase_sampling"]
    seed = str(policy["seed"])
    candidates: dict[str, list[tuple[str, str]]] = {
        name: [] for name in policy["strata_priority"]
    }
    for key in keys:
        flags = tuple(bool(scores[role][key]["final_is_correct"]) for role in ("base", "vision_opd", "cached_prefix"))
        name = stratum(*flags)
        if name:
            rank = hashlib.sha256((seed + "\0" + name + "\0" + key).encode()).hexdigest()
            candidates[name].append((rank, key))

    maximum_per = int(policy["maximum_per_stratum"])
    maximum_total = int(policy["maximum_total"])
    selected: list[tuple[str, str]] = []
    for name in policy["strata_priority"]:
        for _, key in sorted(candidates[name])[:maximum_per]:
            if len(selected) < maximum_total:
                selected.append((name, key))

    rows = []
    for name, key in selected:
        base_score, vision_score, cached_score = (scores[role][key] for role in ("base", "vision_opd", "cached_prefix"))
        base_pred, vision_pred, cached_pred = (predictions[role][key] for role in ("base", "vision_opd", "cached_prefix"))
        rows.append({
            "schema_version": 1,
            "request_key": key.replace("\0", "\\0"),
            "stratum": name,
            "benchmark": base_score["benchmark"],
            "view": base_score.get("view", "full"),
            "sample_uid": base_score["sample_uid"],
            "official_category": base_score.get("official_category"),
            "official_l2_category": base_score.get("official_l2_category"),
            "prompt": base_pred["prompt"],
            "reference_answer": base_score["reference_answer"],
            "base_answer": base_pred["raw_model_answer"],
            "vision_opd_answer": vision_pred["raw_model_answer"],
            "cached_prefix_answer": cached_pred["raw_model_answer"],
            "base_correct": bool(base_score["final_is_correct"]),
            "vision_opd_correct": bool(vision_score["final_is_correct"]),
            "cached_prefix_correct": bool(cached_score["final_is_correct"]),
            "rule_and_judge_sources": {
                role: {
                    "rule_source": scores[role][key].get("rule_source"),
                    "judge_source": scores[role][key].get("judge_source"),
                    "judge_error": scores[role][key].get("judge_error"),
                }
                for role in ("base", "vision_opd", "cached_prefix")
            },
        })

    output = resolve(config["outputs"]["badcases_jsonl"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    selected_counts = Counter(row["stratum"] for row in rows)
    summary = {
        "schema_version": 1,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "PASS",
        "experiment_id": config["experiment_id"],
        "selection_policy": policy,
        "freeze_config_sha256": sha256(config_path),
        "freeze_receipt_sha256": sha256(resolve(config["outputs"]["freeze_receipt"])),
        "candidate_counts": {name: len(candidates[name]) for name in policy["strata_priority"]},
        "selected_counts": dict(selected_counts),
        "selected_total": len(rows),
        "output": str(output.resolve()),
        "output_sha256": sha256(output),
        "manual_review_status": "pending",
        "score_mutation_performed": False,
        "checkpoint_reselection_performed": False,
    }
    target = resolve(config["outputs"]["badcases_summary"])
    target.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(f"DAY15_BADCASES=PASS selected={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
