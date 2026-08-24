#!/usr/bin/env python3
"""Derive three full-evaluation time and cost scenarios from frozen Smoke inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_COST_CAP_CNY = 2000.0
RECOMMENDED_EXECUTION_CAP_CNY = 100.0
JUDGE_SECONDS_PER_INSTANCE = 5.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def make_scenario(
    *,
    name: str,
    effective_concurrency: float,
    overhead_fraction: float,
    individual_inference_seconds: float,
    judge_instances: float,
    price_cny_per_hour: float,
) -> dict[str, Any]:
    if effective_concurrency <= 0 or not 0 <= overhead_fraction < 1:
        raise ValueError("invalid scenario parameters")
    inference_seconds = individual_inference_seconds / effective_concurrency
    judge_seconds = judge_instances * JUDGE_SECONDS_PER_INSTANCE
    total_seconds = (inference_seconds + judge_seconds) * (1 + overhead_fraction)
    wall_hours = total_seconds / 3600
    return {
        "name": name,
        "effective_inference_concurrency": effective_concurrency,
        "judge_instances": judge_instances,
        "judge_seconds_per_instance": JUDGE_SECONDS_PER_INSTANCE,
        "overhead_fraction": overhead_fraction,
        "inference_wall_seconds_before_buffer": inference_seconds,
        "judge_wall_seconds_before_buffer": judge_seconds,
        "estimated_wall_seconds": total_seconds,
        "estimated_wall_hours": wall_hours,
        "estimated_cost_cny": wall_hours * price_cny_per_hour,
    }


def projected_workload(inputs: dict[str, Any]) -> tuple[dict[str, Any], float]:
    observations = inputs["smoke_artifacts"]["observations"]["by_benchmark"]
    workload = inputs["dataset_manifest"]["workload"]["by_benchmark"]
    result: dict[str, Any] = {}
    total_individual_seconds = 0.0
    for benchmark, frozen in sorted(workload.items()):
        observed = observations[benchmark]
        request_count = int(frozen["full_request_count"])
        individual_seconds = float(observed["mean_latency_seconds"]) * request_count
        total_individual_seconds += individual_seconds
        result[benchmark] = {
            "full_request_count": request_count,
            "projected_prompt_tokens": observed["mean_prompt_tokens"] * request_count,
            "projected_completion_tokens": observed["mean_completion_tokens"] * request_count,
            "projected_individual_latency_seconds": individual_seconds,
            "maximum_semantic_judge_instances": int(frozen["maximum_semantic_judge_instances"]),
        }
    return result, total_individual_seconds


def build_cost_document(
    budget_inputs: dict[str, Any],
    *,
    budget_inputs_path: Path,
    scores_path: Path,
    dual_gpu_hourly_cny: float,
    price_source: str,
) -> dict[str, Any]:
    if dual_gpu_hourly_cny <= 0:
        raise ValueError("--dual-gpu-hourly-cny must be positive")
    if budget_inputs["inputs"]["smoke_artifacts"]["summary"]["decision_status"] != "complete":
        raise ValueError("budget inputs do not contain a completed Smoke")
    scores = load_jsonl(scores_path)
    prediction_rows = load_jsonl(Path(budget_inputs["inputs"]["smoke_artifacts"]["predictions_path"]))
    open_instances = sum(item["question_format"] == "open_question" for item in prediction_rows)
    judged_instances = sum(item["score_source"] == "fixed_base_4b_judge" for item in scores)
    if open_instances == 0:
        raise ValueError("Smoke has no open-question instances")
    judge_rate = judged_instances / open_instances
    projected, individual_seconds = projected_workload(budget_inputs["inputs"])
    max_judge = int(budget_inputs["inputs"]["dataset_manifest"]["workload"]["maximum_semantic_judge_instances"])
    expected_judge = max_judge * judge_rate

    observations = budget_inputs["inputs"]["smoke_artifacts"]["observations"]
    observed_concurrency = observations["sum_request_latency_seconds"] / observations["completion_span_seconds"]
    scenarios = [
        make_scenario(
            name="measured_throughput",
            effective_concurrency=observed_concurrency,
            overhead_fraction=0.15,
            individual_inference_seconds=individual_seconds,
            judge_instances=expected_judge,
            price_cny_per_hour=dual_gpu_hourly_cny,
        ),
        make_scenario(
            name="conservative_execution_budget",
            effective_concurrency=5.0,
            overhead_fraction=0.30,
            individual_inference_seconds=individual_seconds,
            judge_instances=expected_judge,
            price_cny_per_hour=dual_gpu_hourly_cny,
        ),
        make_scenario(
            name="worst_case_guardrail",
            effective_concurrency=4.0,
            overhead_fraction=0.50,
            individual_inference_seconds=individual_seconds,
            judge_instances=max_judge,
            price_cny_per_hour=dual_gpu_hourly_cny,
        ),
    ]
    return {
        "schema_version": 1,
        "purpose": "day5_task6_full_external_evaluation_time_and_cost_budget",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": budget_inputs["experiment_id"],
        "protocol_revision": budget_inputs["protocol_revision"],
        "budget_inputs": {
            "path": str(budget_inputs_path.resolve()),
            "sha256": sha256_file(budget_inputs_path),
        },
        "pricing": {
            "dual_gpu_hourly_cny": dual_gpu_hourly_cny,
            "source": price_source,
            "billing_unit": "dual_gpu_instance_hour",
        },
        "frozen_measurement": {
            "smoke_scores_path": str(scores_path.resolve()),
            "smoke_scores_sha256": sha256_file(scores_path),
            "smoke_open_question_instances": open_instances,
            "smoke_semantic_judge_instances": judged_instances,
            "observed_semantic_judge_rate": judge_rate,
            "observed_effective_inference_concurrency": observed_concurrency,
            "judge_seconds_per_instance_assumption": JUDGE_SECONDS_PER_INSTANCE,
        },
        "full_workload_projection": {
            "by_benchmark": projected,
            "full_request_count": sum(item["full_request_count"] for item in projected.values()),
            "projected_prompt_tokens": sum(item["projected_prompt_tokens"] for item in projected.values()),
            "projected_completion_tokens": sum(item["projected_completion_tokens"] for item in projected.values()),
            "individual_request_latency_seconds": individual_seconds,
            "expected_semantic_judge_instances": expected_judge,
            "maximum_semantic_judge_instances": max_judge,
        },
        "scenarios": scenarios,
        "guardrails": {
            "project_cost_cap_cny": PROJECT_COST_CAP_CNY,
            "recommended_execution_cap_cny": RECOMMENDED_EXECUTION_CAP_CNY,
            "worst_case_within_project_cap": scenarios[-1]["estimated_cost_cny"] <= PROJECT_COST_CAP_CNY,
            "worst_case_within_recommended_cap": scenarios[-1]["estimated_cost_cny"] <= RECOMMENDED_EXECUTION_CAP_CNY,
            "run_policy": "Start Day 6 only under the conservative budget; stop and investigate before exceeding the recommended cap.",
        },
    }


def render_markdown(document: dict[str, Any]) -> str:
    workload = document["full_workload_projection"]
    rows = [
        "# Day 5 Task 6: Full external-evaluation budget",
        "",
        f"- Experiment: {document['experiment_id']}, protocol revision {document['protocol_revision']}",
        f"- Pricing: {document['pricing']['dual_gpu_hourly_cny']:.2f} CNY per dual-GPU instance hour ({document['pricing']['source']})",
        f"- Full workload: {workload['full_request_count']} requests; expected semantic Judge {workload['expected_semantic_judge_instances']:.0f}, maximum {workload['maximum_semantic_judge_instances']}",
        f"- Projected tokens: prompt {workload['projected_prompt_tokens']:.0f}, completion {workload['projected_completion_tokens']:.0f}",
        "",
        "| Scenario | Wall time | Cost (CNY) | Judge instances |",
        "|---|---:|---:|---:|",
    ]
    for scenario in document["scenarios"]:
        rows.append(
            f"| {scenario['name']} | {scenario['estimated_wall_hours']:.2f} h | "
            f"{scenario['estimated_cost_cny']:.2f} | {scenario['judge_instances']:.0f} |"
        )
    guardrails = document["guardrails"]
    rows.extend(
        [
            "",
            f"Recommended execution cap: {guardrails['recommended_execution_cap_cny']:.2f} CNY; project hard cap: {guardrails['project_cost_cap_cny']:.2f} CNY.",
            "",
            "Method: benchmark-specific mean Smoke latency and token counts are multiplied by frozen full request counts. The measured scenario uses observed completion-span concurrency; the conservative and worst-case scenarios apply fixed lower concurrency and larger buffers. Judge calls are sequentially budgeted at 5 seconds each.",
        ]
    )
    return "\n".join(rows) + "\n"


def write_artifact(path: Path, content: str, *, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite frozen artifact: {path}; use --force only for an amendment")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget-inputs", default="artifacts/runs/E-D5-001/budget_inputs.json")
    parser.add_argument("--dual-gpu-hourly-cny", type=float, required=True)
    parser.add_argument("--price-source", required=True)
    parser.add_argument("--output", default="artifacts/runs/E-D5-001/cost.json")
    parser.add_argument("--report", default="artifacts/runs/E-D5-001/full_eval_budget.md")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    inputs_path = Path(args.budget_inputs)
    frozen_inputs = json.loads(inputs_path.read_text(encoding="utf-8"))
    scores_path = Path(frozen_inputs["inputs"]["smoke_artifacts"]["predictions_path"]).with_name("scores.jsonl")
    cost_path = Path(args.output)
    report_path = Path(args.report)
    document = build_cost_document(
        frozen_inputs,
        budget_inputs_path=inputs_path,
        scores_path=scores_path,
        dual_gpu_hourly_cny=args.dual_gpu_hourly_cny,
        price_source=args.price_source,
    )
    write_artifact(cost_path, json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n", force=args.force)
    write_artifact(report_path, render_markdown(document), force=args.force)
    hash_path = cost_path.with_name("budget_artifacts.sha256")
    hashes = [
        (sha256_file(cost_path), cost_path.resolve()),
        (sha256_file(report_path), report_path.resolve()),
    ]
    write_artifact(hash_path, "".join(f"{digest}  {path}\n" for digest, path in hashes), force=args.force)
    print(json.dumps({"cost_json": str(cost_path.resolve()), "report": str(report_path.resolve()), "hashes": str(hash_path.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()

