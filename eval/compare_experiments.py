#!/usr/bin/env python3
"""Compare frozen Base, Vision-OPD, and Cached R3 results without changing scores."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
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


def key(item: dict[str, Any]) -> str:
    return "\0".join((str(item["benchmark"]), str(item.get("view", "full")), str(item["sample_uid"])))


def read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        item_key = key(item)
        if item_key in rows:
            raise ValueError(f"duplicate key in {path}:{number}: {item_key!r}")
        rows[item_key] = item
    return rows


def load_run(path: Path, expected_role: str, expected_total: int) -> dict[str, Any]:
    required = {
        "run_manifest.json", "predictions.jsonl", "judge_results.jsonl",
        "scores.jsonl", "summary.json", "validation.json", "metrics.json",
        "cost.json", "artifact_sha256.txt",
    }
    missing = sorted(name for name in required if not (path / name).is_file())
    if missing:
        raise FileNotFoundError(f"{path}: missing {missing}")
    manifest = json.loads((path / "run_manifest.json").read_text())
    summary = json.loads((path / "summary.json").read_text())
    validation = json.loads((path / "validation.json").read_text())
    scores = read_jsonl(path / "scores.jsonl")
    predictions = read_jsonl(path / "predictions.jsonl")
    if manifest.get("model_role") != expected_role or summary.get("model_role") != expected_role:
        raise ValueError(f"{path}: expected model role {expected_role}")
    if manifest.get("run_mode") != "formal" or summary.get("decision_status") != "complete":
        raise ValueError(f"{path}: formal scoring is not complete")
    if validation.get("status") != "pass" or validation.get("run", {}).get("status") != "pass":
        raise ValueError(f"{path}: validation is not pass")
    if len(scores) != expected_total or len(predictions) != expected_total or scores.keys() != predictions.keys():
        raise ValueError(f"{path}: expected {expected_total} matching prediction/score keys")
    return {
        "path": path, "manifest": manifest, "summary": summary,
        "validation": validation, "scores": scores, "predictions": predictions,
    }


def stat(scores: dict[str, dict[str, Any]], benchmark: str | None = None) -> dict[str, Any]:
    selected = [
        row for row in scores.values()
        if benchmark is None or row["benchmark"] == benchmark
    ]
    correct = sum(bool(row["final_is_correct"]) for row in selected)
    return {
        "total": len(selected),
        "correct": correct,
        "accuracy": correct / len(selected) if selected else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/day15_final_eval_freeze.yaml")
    args = parser.parse_args()
    config_path = resolve(args.config)
    config = yaml.safe_load(config_path.read_text())
    expected_total = int(config["output_schema"]["expected_total"])
    paths = {
        "base": resolve(config["base_run"]),
        "vision_opd": resolve(config["models"]["vision_opd"]["formal_output"]),
        "cached_prefix": resolve(config["models"]["cached_prefix"]["formal_output"]),
    }
    runs = {
        role: load_run(path, role, expected_total)
        for role, path in paths.items()
    }
    key_sets = [set(run["scores"]) for run in runs.values()]
    if not all(item == key_sets[0] for item in key_sets[1:]):
        raise ValueError("model runs do not have identical request keys")
    base_manifest = runs["base"]["manifest"]
    for role in ("vision_opd", "cached_prefix"):
        manifest = runs[role]["manifest"]
        for field in (
            "config_sha256_raw_bytes", "amendment_sha256_raw_bytes",
            "dataset_files", "request_contract", "expected_requests",
            "expected_request_count", "resume_key",
        ):
            if manifest.get(field) != base_manifest.get(field):
                raise ValueError(f"{role}: manifest field differs from Base: {field}")

    rows = []
    for benchmark in ("zoombench", "mmstar", "vstar", None):
        stats = {role: stat(run["scores"], benchmark) for role, run in runs.items()}
        rows.append({
            "benchmark": benchmark or "overall_micro",
            "total": stats["base"]["total"],
            "base_correct": stats["base"]["correct"],
            "base_accuracy": stats["base"]["accuracy"],
            "vision_opd_correct": stats["vision_opd"]["correct"],
            "vision_opd_accuracy": stats["vision_opd"]["accuracy"],
            "vision_opd_delta_vs_base": stats["vision_opd"]["accuracy"] - stats["base"]["accuracy"],
            "cached_prefix_correct": stats["cached_prefix"]["correct"],
            "cached_prefix_accuracy": stats["cached_prefix"]["accuracy"],
            "cached_prefix_delta_vs_base": stats["cached_prefix"]["accuracy"] - stats["base"]["accuracy"],
            "cached_prefix_delta_vs_vision_opd": stats["cached_prefix"]["accuracy"] - stats["vision_opd"]["accuracy"],
        })

    transitions = {}
    for left, right in (
        ("base", "vision_opd"), ("base", "cached_prefix"), ("vision_opd", "cached_prefix")
    ):
        transitions[f"{left}_to_{right}"] = {
            "incorrect_to_correct": sum(
                not bool(runs[left]["scores"][k]["final_is_correct"])
                and bool(runs[right]["scores"][k]["final_is_correct"])
                for k in key_sets[0]
            ),
            "correct_to_incorrect": sum(
                bool(runs[left]["scores"][k]["final_is_correct"])
                and not bool(runs[right]["scores"][k]["final_is_correct"])
                for k in key_sets[0]
            ),
        }

    result = {
        "schema_version": 1,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "PASS",
        "experiment_id": config["experiment_id"],
        "freeze_config": str(config_path.resolve()),
        "freeze_config_sha256": sha256(config_path),
        "freeze_receipt": str(resolve(config["outputs"]["freeze_receipt"]).resolve()),
        "freeze_receipt_sha256": sha256(resolve(config["outputs"]["freeze_receipt"])),
        "runs": {
            role: {
                "path": str(run["path"].resolve()),
                "run_manifest_sha256": sha256(run["path"] / "run_manifest.json"),
                "scores_sha256": sha256(run["path"] / "scores.jsonl"),
                "summary_sha256": sha256(run["path"] / "summary.json"),
                "validation_sha256": sha256(run["path"] / "validation.json"),
            }
            for role, run in runs.items()
        },
        "rows": rows,
        "paired_transitions": transitions,
        "notes": [
            "All failures and invalid outputs remain in the frozen denominator.",
            "V* primary denominator is 191.",
            "Comparison does not mutate scores or select checkpoints.",
        ],
    }
    output_json = resolve(config["outputs"]["comparison_json"])
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")

    headings = [
        "Benchmark", "N", "Base", "Vision-OPD", "Delta V-B",
        "Cached", "Delta C-B", "Delta C-V",
    ]
    lines = [
        "# Base / Vision-OPD / Cached Prefix R3 Comparison", "",
        "| " + " | ".join(headings) + " |",
        "|" + "|".join(["---"] + ["---:"] * (len(headings) - 1)) + "|",
    ]
    for row in rows:
        pct = lambda value: f"{100 * value:.2f}%"
        lines.append(
            f"| {row['benchmark']} | {row['total']} | {pct(row['base_accuracy'])} | "
            f"{pct(row['vision_opd_accuracy'])} | {pct(row['vision_opd_delta_vs_base'])} | "
            f"{pct(row['cached_prefix_accuracy'])} | {pct(row['cached_prefix_delta_vs_base'])} | "
            f"{pct(row['cached_prefix_delta_vs_vision_opd'])} |"
        )
    lines.extend(["", "All results use the frozen single-GPU R3 protocol and fixed Base Judge."])
    output_md = resolve(config["outputs"]["comparison_markdown"])
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n")
    print(f"DAY15_COMPARISON=PASS rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
