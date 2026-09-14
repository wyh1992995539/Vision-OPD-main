#!/usr/bin/env python3
"""Validate the Vision-OPD MCQ reward against all converted GRPO rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import sys
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "verl/utils/reward_score"))
from vision_opd_mcq_grpo import (
    DATA_SOURCE,
    DEFAULT_VALID_OPTIONS,
    REWARD_ROUTE,
    compute_score,
)


DEFAULT_DATA = Path("/root/autodl-tmp/data/vision_opd_6241/grpo_train_6241.parquet")
DEFAULT_RUN_DIR = ROOT / "artifacts/runs/E-D17-GRPO-DATA-001"
REWARD_FILE = ROOT / "verl/utils/reward_score/vision_opd_mcq_grpo.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full 6,241-row GRPO reward gate.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--expected-rows", type=int, default=6241)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_reward_dataset(
    data_path: Path,
    run_dir: Path,
    *,
    expected_rows: int = 6241,
    overwrite: bool = False,
) -> dict[str, Any]:
    data_path = data_path.resolve()
    run_dir = run_dir.resolve()
    report_path = run_dir / "reward_validation.json"
    hash_path = run_dir / "reward_function_sha256.txt"
    if not data_path.is_file():
        raise FileNotFoundError(f"GRPO Parquet not found: {data_path}")
    existing = [path for path in (report_path, hash_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"reward validation output exists; use --overwrite: {existing}")

    table = pq.read_table(data_path)
    if table.num_rows != expected_rows:
        raise ValueError(f"expected {expected_rows} rows, found {table.num_rows}")

    failures: list[dict[str, Any]] = []
    correct_trials = 0
    incorrect_trials = 0
    parse_valid_trials = 0
    seen_ids: set[str] = set()
    gold_distribution = {option: 0 for option in DEFAULT_VALID_OPTIONS}
    for row_index, row in enumerate(table.to_pylist()):
        extra_info = row["extra_info"]
        sample_id = str(extra_info["sample_id"])
        if sample_id in seen_ids:
            failures.append({"index": row_index, "sample_id": sample_id, "reason": "duplicate_sample_id"})
            continue
        seen_ids.add(sample_id)
        gold = str(row["reward_model"]["ground_truth"])
        if gold in gold_distribution:
            gold_distribution[gold] += 1
        wrong = next(option for option in DEFAULT_VALID_OPTIONS if option != gold)
        try:
            correct = compute_score(row["data_source"], f"Answer: {gold}", gold, extra_info)
            incorrect = compute_score(row["data_source"], f"Answer: {wrong}", gold, extra_info)
        except Exception as exc:  # pragma: no cover - reported with the production dataset
            failures.append(
                {
                    "index": row_index,
                    "sample_id": sample_id,
                    "reason": "reward_exception",
                    "detail": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        correct_trials += int(correct["score"] == 1.0)
        incorrect_trials += int(incorrect["score"] == 0.0)
        parse_valid_trials += int(correct["parse_valid"]) + int(incorrect["parse_valid"])
        if correct["score"] != 1.0 or incorrect["score"] != 0.0:
            failures.append(
                {
                    "index": row_index,
                    "sample_id": sample_id,
                    "reason": "unexpected_reward",
                    "correct_result": correct,
                    "incorrect_result": incorrect,
                }
            )

    reward_sha256 = sha256_file(REWARD_FILE)
    data_sha256 = sha256_file(data_path)
    passed = (
        not failures
        and len(seen_ids) == expected_rows
        and correct_trials == expected_rows
        and incorrect_trials == expected_rows
        and parse_valid_trials == 2 * expected_rows
    )
    report = {
        "schema_version": 1,
        "experiment_id": "E-D17-GRPO-DATA-001",
        "gate": "vision_opd_mcq_grpo_v1_reward",
        "status": "PASS" if passed else "FAIL",
        "data_path": data_path.as_posix(),
        "data_sha256": data_sha256,
        "reward_file": REWARD_FILE.as_posix(),
        "reward_file_sha256": reward_sha256,
        "data_source": DATA_SOURCE,
        "reward_route": REWARD_ROUTE,
        "rows": table.num_rows,
        "unique_sample_ids": len(seen_ids),
        "gold_distribution": gold_distribution,
        "trials": {
            "synthetic_correct": expected_rows,
            "synthetic_incorrect": expected_rows,
            "total": 2 * expected_rows,
        },
        "checks": {
            "all_correct_templates_score_one": correct_trials == expected_rows,
            "all_wrong_templates_score_zero": incorrect_trials == expected_rows,
            "all_synthetic_answers_parse_valid": parse_valid_trials == 2 * expected_rows,
            "no_reward_exceptions": not any(
                failure["reason"] == "reward_exception" for failure in failures
            ),
            "unique_sample_ids": len(seen_ids) == expected_rows,
            "binary_outcome_only": True,
            "learned_reward_model_used": False,
            "benchmark_base_judge_used": False,
        },
        "failure_count": len(failures),
        "failure_examples": failures[:20],
        "reward_gate_passed": passed,
        "pilot_authorized": False,
        "pending_gates": [
            "load_grpo_parquet_through_training_process_with_sufficient_cgroup_memory",
            "freeze_grpo_training_config_and_guarded_launcher",
            "run_32_prompt_real_policy_gradient_pilot",
        ],
    }
    atomic_json(report_path, report)
    atomic_text(hash_path, f"{reward_sha256}  {REWARD_FILE.as_posix()}\n")
    if not passed:
        raise AssertionError(f"GRPO reward validation failed: {failures[:3]}")
    return report


def main() -> None:
    args = parse_args()
    report = validate_reward_dataset(
        args.data,
        args.run_dir,
        expected_rows=args.expected_rows,
        overwrite=args.overwrite,
    )
    print("PASS: vision_opd_mcq_grpo_v1 reward validation completed")
    print(f"  rows: {report['rows']}")
    print(f"  trials: {report['trials']['total']}")
    print(f"  reward sha256: {report['reward_file_sha256']}")
    print(f"  report: {(args.run_dir / 'reward_validation.json').resolve()}")
    print("  pilot_authorized: false (trainer gates remain)")


if __name__ == "__main__":
    main()
