#!/usr/bin/env python3
"""Postflight audit for the 16/64-row Vision-OPD 6K training Pilots."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.monitor_vopd_training import (
    ANSI_RE,
    METRIC_RE,
    parse_scalar,
    parse_training_metric_line,
    scan_fatal_log_line,
    validate_checkpoint,
    write_json,
)
from scripts.run_vopd_6241_pilot_guarded import (
    DEFAULT_POLICY,
    load_pilot_policy,
    resolve,
    sha256_file,
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            values.append(value)
    return values


def parse_steps(text: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    steps: dict[int, dict[str, Any]] = {}
    duplicates = 0
    for line in text.splitlines():
        parsed = parse_training_metric_line(line)
        if parsed is None:
            continue
        clean = ANSI_RE.sub("", line).replace("\r", "")
        raw = {key: parse_scalar(value) for key, value in METRIC_RE.findall(clean)}
        step = int(parsed["step"])
        if step in steps:
            duplicates += 1
            continue
        parsed.update(
            {
                "learning_rate": raw.get("actor/lr"),
                "step_seconds": raw.get("timing_s/step"),
                "generation_seconds": raw.get("timing_s/gen"),
                "checkpoint_save_seconds": raw.get("timing_s/save_checkpoint"),
                "prompt_max_tokens": raw.get("prompt_length/max"),
                "prompt_clip_ratio": raw.get("prompt_length/clip_ratio"),
                "response_mean_tokens": raw.get("response_length/mean"),
                "response_max_tokens": raw.get("response_length/max"),
                "response_clip_ratio": raw.get("response_length/clip_ratio"),
                "teacher_always_on_fraction": raw.get(
                    "self_distillation/teacher_always_on_fraction"
                ),
                "teacher_image_swap_fraction": raw.get(
                    "self_distillation/teacher_image_swap_fraction"
                ),
            }
        )
        steps[step] = parsed
    lower = text.lower()
    signals = {
        "duplicate_metric_steps": duplicates,
        "traceback_count": text.count("Traceback (most recent call last):"),
        "cuda_oom_count": lower.count("cuda out of memory"),
        "out_of_memory_error_count": lower.count("outofmemoryerror"),
        "dataloader_worker_killed_count": lower.count("is killed by signal: killed"),
        "checkpoint_save_failure_count": len(
            [rule for line in text.splitlines() for rule in scan_fatal_log_line(line)
             if rule == "checkpoint_save_failure"]
        ),
    }
    return [steps[key] for key in sorted(steps)], signals


def finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def evaluate_steps(
    rows: list[dict[str, Any]], expected_steps: int, max_response_tokens: int = 1024,
    warmup_steps: int = 0,
) -> dict[str, bool]:
    expected = list(range(1, expected_steps + 1))
    required_finite = (
        "loss", "grad_norm", "learning_rate", "student_optimizer_delta", "teacher_optimizer_delta",
        "teacher_grad_non_none_count", "teacher_ema_delta", "ema_update_applied",
        "aborted_ratio", "prompt_max_tokens", "prompt_clip_ratio",
        "response_mean_tokens", "response_max_tokens", "response_clip_ratio",
        "teacher_always_on_fraction", "teacher_image_swap_fraction",
    )
    complete = bool(rows) and all(
        all(finite(row.get(key)) for key in required_finite) for row in rows
    )
    return {
        "exact_contiguous_steps": [int(row["step"]) for row in rows] == expected,
        "required_metrics_present_and_finite": complete,
        "jsd_loss_finite": bool(rows) and all(finite(row.get("loss")) for row in rows),
        "learning_rate_nonnegative": complete
        and all(float(row["learning_rate"]) >= 0 for row in rows),
        "zero_lr_only_within_warmup": complete
        and all(
            float(row["learning_rate"]) > 0 or int(row["step"]) <= warmup_steps
            for row in rows
        ),
        "positive_learning_rate_observed": complete
        and any(float(row["learning_rate"]) > 0 for row in rows),
        "student_update_matches_learning_rate": complete
        and all(
            (
                float(row["student_optimizer_delta"]) >= 0
                if float(row["learning_rate"]) == 0
                else float(row["student_optimizer_delta"]) > 0
            )
            for row in rows
        ),
        "teacher_optimizer_unchanged": complete
        and all(float(row["teacher_optimizer_delta"]) == 0 for row in rows),
        "teacher_direct_gradient_absent": complete
        and all(float(row["teacher_grad_non_none_count"]) == 0 for row in rows),
        "teacher_ema_updated_each_step": complete
        and all(
            float(row["teacher_ema_delta"]) > 0
            and int(row["ema_update_applied"]) == 1
            for row in rows
        ),
        "crop_teacher_active_each_step": complete
        and all(
            float(row["teacher_always_on_fraction"]) == 1
            and float(row["teacher_image_swap_fraction"]) == 1
            for row in rows
        ),
        "generation_errors_zero": complete
        and all(float(row["aborted_ratio"]) == 0 for row in rows),
        "prompt_truncation_zero": complete
        and all(float(row["prompt_clip_ratio"]) == 0 for row in rows),
        "response_within_frozen_limit": complete
        and all(float(row["response_max_tokens"]) <= max_response_tokens for row in rows),
    }


def telemetry_summary(output_dir: Path) -> dict[str, Any]:
    paths = {
        "gpu": output_dir / "evidence/telemetry/gpu.jsonl",
        "process_rss": output_dir / "evidence/telemetry/process_rss.jsonl",
        "cgroup": output_dir / "evidence/telemetry/cgroup_memory.jsonl",
        "disk": output_dir / "evidence/telemetry/disk.jsonl",
    }
    rows: dict[str, list[dict[str, Any]]] = {}
    errors: list[str] = []
    for name, path in paths.items():
        if not path.is_file():
            errors.append(f"missing telemetry: {path}")
            rows[name] = []
            continue
        try:
            rows[name] = read_jsonl(path)
        except ValueError as exc:
            errors.append(str(exc))
            rows[name] = []
        if not rows[name]:
            errors.append(f"empty telemetry: {path}")

    gpu_rows = [gpu for sample in rows["gpu"] for gpu in sample.get("gpus", [])]
    gpu_ids = sorted({int(gpu["index"]) for gpu in gpu_rows if "index" in gpu})
    peak_by_gpu: dict[str, dict[str, Any]] = {}
    for gpu in gpu_rows:
        index = str(gpu.get("index"))
        used = int(gpu.get("memory_used_bytes", 0))
        total = int(gpu.get("memory_total_bytes", 0))
        previous = peak_by_gpu.get(index)
        if previous is None or used > previous["memory_used_bytes"]:
            peak_by_gpu[index] = {
                "memory_used_bytes": used,
                "memory_total_bytes": total,
                "used_ratio": used / total if total else None,
            }
    elapsed = [float(row.get("elapsed_seconds", 0)) for row in rows["gpu"]]
    return {
        "status": "PASS" if not errors and len(gpu_ids) == 2 else "FAIL",
        "paths": {name: str(path) for name, path in paths.items()},
        "row_counts": {name: len(value) for name, value in rows.items()},
        "gpu_ids": gpu_ids,
        "peak_by_gpu": peak_by_gpu,
        "max_observed_elapsed_seconds": max(elapsed, default=0),
        "errors": errors,
    }


def projection(rows: list[dict[str, Any]], telemetry: dict[str, Any], rate: float) -> dict[str, Any] | None:
    if len(rows) != 8 or not all(finite(row.get("step_seconds")) for row in rows):
        return None
    steady = [float(row["step_seconds"]) for row in rows[1:]]
    checkpoint_seconds = float(rows[-1].get("checkpoint_save_seconds") or 0)
    observed = float(telemetry.get("max_observed_elapsed_seconds", 0))
    measured = sum(float(row["step_seconds"]) for row in rows) + checkpoint_seconds
    startup = max(0.0, observed - measured)
    scenarios = {}
    for name, seconds in {
        "median": statistics.median(steady),
        "mean": statistics.mean(steady),
        "conservative_max": max(steady),
    }.items():
        total = startup + float(rows[0]["step_seconds"]) + 779 * seconds + 2 * checkpoint_seconds
        scenarios[name] = {
            "steady_step_seconds": seconds,
            "projected_total_seconds": total,
            "projected_dual_gpu_hours": total / 3600,
            "projected_cost_cny": total / 3600 * rate,
        }
    return {
        "target_optimizer_steps": 780,
        "checkpoint_count": 2,
        "startup_seconds_estimate": startup,
        "pilot_final_checkpoint_seconds": checkpoint_seconds,
        "scenarios": scenarios,
        "status": "MEASURED_PROJECTION_NOT_YET_FROZEN",
    }


def render_markdown(report: dict[str, Any]) -> str:
    checks = "\n".join(
        f"| `{name}` | {'PASS' if passed else 'FAIL'} |"
        for name, passed in report["checks"].items()
    )
    return f"""# Vision-OPD 6241 Pilot {report['stage']} Postflight

- 状态：**{report['status']}**
- 训练链路通过：`{str(report['training_gate_pass']).lower()}`
- 阶段 Gate 通过：`{str(report['stage_gate_pass']).lower()}`
- 正式训练授权：`false`

| 检查 | 结果 |
| --- | --- |
{checks}

此报告只证明 Pilot 工程与机制状态，不是模型能力结论，也不能作为正式训练 checkpoint。
"""


def audit(stage: str, policy_path: Path, reload_report: Path | None = None) -> dict[str, Any]:
    policy, contract = load_pilot_policy(policy_path, stage)
    config_path = resolve(contract["config"])
    output_dir = resolve(contract["output_dir"])
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    expected_steps = int(contract["expected_optimizer_steps"])
    paths = {
        "train_log": output_dir / "logs/train.log",
        "guard_summary": output_dir / "evidence/guard_summary.json",
        "run_invocation": output_dir / "preflight/run_invocation.json",
        "preflight": output_dir / "preflight/preflight_summary.json",
        "selection_manifest": resolve(config["paths"]["selection_manifest"]),
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        return {
            "schema_version": 1, "stage": stage, "experiment_id": contract["experiment_id"],
            "status": "FAIL", "training_gate_pass": False, "stage_gate_pass": False,
            "formal_training_authorized": False, "missing_inputs": missing, "checks": {},
        }

    log_text = paths["train_log"].read_text(encoding="utf-8", errors="replace")
    rows, signals = parse_steps(log_text)
    warmup_steps = int(config["actor"]["lr_warmup_steps"])
    step_checks = evaluate_steps(
        rows,
        expected_steps,
        int(config["data"]["max_response_length"]),
        warmup_steps,
    )
    guard = read_json(paths["guard_summary"])
    invocation = read_json(paths["run_invocation"])
    preflight = read_json(paths["preflight"])
    selection = read_json(paths["selection_manifest"])
    checkpoint = validate_checkpoint(output_dir, policy)
    telemetry = telemetry_summary(output_dir)
    log_prob_files = [
        path for path in (output_dir / "evidence/log_probs").rglob("*.pt")
        if path.is_file() and path.stat().st_size > 0
    ] if (output_dir / "evidence/log_probs").is_dir() else []

    from scripts.checkpoint_io_contract import checkpoint_io_matches

    checks = {
        "checkpoint_io_revision_matches": checkpoint_io_matches(invocation),
        "guard_pass": guard.get("status") == "PASS",
        "training_preflight_pass": preflight.get("status") == "PASS",
        "run_invocation_matches": (
            invocation.get("experiment_id") == contract["experiment_id"]
            and invocation.get("config_sha256") == sha256_file(config_path)
            and invocation.get("train_file_sha256") == selection.get("output", {}).get("sha256")
        ),
        "no_duplicate_metric_steps": signals["duplicate_metric_steps"] == 0,
        "no_traceback_or_oom": (
            signals["traceback_count"] == 0
            and signals["cuda_oom_count"] == 0
            and signals["out_of_memory_error_count"] == 0
            and signals["dataloader_worker_killed_count"] == 0
            and signals["checkpoint_save_failure_count"] == 0
        ),
        "log_prob_evidence_complete": len(log_prob_files) >= expected_steps * 2,
        "checkpoint_complete": checkpoint["status"] == "PASS",
        "telemetry_complete_two_gpus": telemetry["status"] == "PASS",
        **step_checks,
    }
    training_gate_pass = all(checks.values())
    reload_required = bool(contract.get("require_cold_reload"))
    reload_value = None
    reload_pass = not reload_required
    if reload_report:
        reload_value = read_json(reload_report)
        reload_pass = (
            reload_value.get("status") == "PASS"
            and reload_value.get("source_checkpoint_unchanged") is True
            and reload_value.get("verification", {}).get("status") == "PASS"
        )
    stage_gate_pass = training_gate_pass and reload_pass
    if stage_gate_pass:
        status = "PASS"
    elif training_gate_pass and reload_required and reload_report is None:
        status = "PASS_TRAINING_PENDING_RELOAD"
    else:
        status = "FAIL"
    failed = sorted(name for name, passed in checks.items() if not passed)
    report = {
        "schema_version": 1, "stage": stage, "experiment_id": contract["experiment_id"],
        "status": status, "training_gate_pass": training_gate_pass,
        "stage_gate_pass": stage_gate_pass, "formal_training_authorized": False,
        "checks": checks, "failed_checks": failed, "signals": signals,
        "observed_steps": len(rows), "expected_steps": expected_steps, "steps": rows,
        "warmup_contract": {
            "lr_warmup_steps": warmup_steps,
            "zero_lr_steps": [
                int(row["step"]) for row in rows
                if finite(row.get("learning_rate")) and float(row["learning_rate"]) == 0
            ],
            "positive_lr_steps": [
                int(row["step"]) for row in rows
                if finite(row.get("learning_rate")) and float(row["learning_rate"]) > 0
            ],
            "student_update_rule": "delta>=0 when lr=0; delta>0 when lr>0",
        },
        "checkpoint": checkpoint, "telemetry": telemetry,
        "log_prob_file_count": len(log_prob_files),
        "checkpoint_io_contract": invocation.get("checkpoint_io_contract"),
        "reload_required": reload_required,
        "reload_report": str(reload_report.resolve()) if reload_report else None,
        "reload": reload_value,
        "projection_780": projection(
            rows, telemetry, float(policy["budget"]["hourly_dual_gpu_rate_cny"])
        ) if stage == "64" and training_gate_pass else None,
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("16", "64"), required=True)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--reload-report", type=Path)
    args = parser.parse_args()
    policy_path = resolve(args.policy)
    _policy, contract = load_pilot_policy(policy_path, args.stage)
    output_dir = resolve(contract["output_dir"])
    report = audit(
        args.stage, policy_path,
        resolve(args.reload_report) if args.reload_report else None,
    )
    evidence = output_dir / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    json_path = evidence / "postflight.json"
    markdown_path = evidence / "postflight.md"
    sha_path = evidence / "postflight_sha256.txt"
    write_json(json_path, report)
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    sha_path.write_text(
        f"{sha256_file(json_path)}  {json_path.resolve()}\n"
        f"{sha256_file(markdown_path)}  {markdown_path.resolve()}\n",
        encoding="utf-8",
    )
    print(f"PILOT_POSTFLIGHT={report['status']}")
    print(f"TRAINING_GATE_PASS={report['training_gate_pass']}")
    print(f"STAGE_GATE_PASS={report['stage_gate_pass']}")
    print(f"OUTPUT={json_path.resolve()}")
    return 0 if report["training_gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
