#!/usr/bin/env python3
"""Postflight audit for the 128-row, 16-step formal-candidate validation."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_vopd_6241_pilot import (
    evaluate_steps,
    finite,
    parse_steps,
    read_json,
    read_jsonl,
    telemetry_summary,
)
from scripts.checkpoint_io_contract import checkpoint_io_matches
from scripts.monitor_vopd_training import validate_checkpoint, write_json
from scripts.run_vopd_6241_candidate_128_guarded import (
    DEFAULT_POLICY,
    load_candidate_policy,
    resolve,
    sha256_file,
)


def source_bindings_current(live_gate: dict[str, Any]) -> bool:
    bindings = live_gate.get("source_bindings")
    if not isinstance(bindings, dict) or not bindings:
        return False
    return all(
        isinstance(entry, dict)
        and Path(str(entry.get("path", ""))).is_file()
        and sha256_file(Path(entry["path"])) == entry.get("sha256")
        for entry in bindings.values()
    )


def cgroup_checks(output_dir: Path, threshold: float) -> tuple[bool, bool, dict[str, Any]]:
    path = output_dir / "evidence/telemetry/cgroup_memory.jsonl"
    if not path.is_file():
        return False, False, {"path": str(path), "rows": 0}
    try:
        rows = read_jsonl(path)
    except ValueError as exc:
        return False, False, {"path": str(path), "rows": 0, "error": str(exc)}
    ratios: list[float] = []
    for row in rows:
        current = row.get("memory_current_bytes")
        maximum = row.get("memory_max_bytes")
        if isinstance(current, int) and isinstance(maximum, int) and maximum > 0:
            ratios.append(current / maximum)
    events = [row.get("memory_events") for row in rows if isinstance(row.get("memory_events"), dict)]
    oom_stable = bool(events) and all(
        int(row.get(key, 0)) == int(events[0].get(key, 0))
        for row in events
        for key in ("oom", "oom_kill")
    )
    below = bool(rows) and len(ratios) == len(rows) and max(ratios, default=1.0) < threshold
    return below, oom_stable, {
        "path": str(path),
        "rows": len(rows),
        "peak_used_ratio": max(ratios, default=None),
        "abort_ratio": threshold,
        "memory_events_first": events[0] if events else None,
        "memory_events_last": events[-1] if events else None,
    }


def failed_report(
    experiment_id: str, missing: list[str], checks: dict[str, bool] | None = None
) -> dict[str, Any]:
    checks = checks or {}
    return {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "status": "FAIL",
        "training_gate_pass": False,
        "validation_gate_pass": False,
        "formal_training_authorized": False,
        "missing_inputs": missing,
        "checks": checks,
        "failed_checks": sorted(name for name, passed in checks.items() if not passed),
    }


def audit(policy_path: Path = DEFAULT_POLICY) -> dict[str, Any]:
    policy_path = resolve(policy_path)
    policy, contract = load_candidate_policy(policy_path)
    config_path = resolve(contract["config"])
    output_dir = resolve(contract["output_dir"])
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    expected_steps = int(contract["expected_optimizer_steps"])
    max_response_tokens = int(config["data"]["max_response_length"])
    warmup_steps = int(config["actor"]["lr_warmup_steps"])

    paths = {
        "train_log": output_dir / "logs/train.log",
        "guard_summary": output_dir / "evidence/guard_summary.json",
        "live_launch_gate": output_dir / "preflight/live_launch_gate.json",
        "run_invocation": output_dir / "preflight/run_invocation.json",
        "training_preflight": output_dir / "preflight/preflight_summary.json",
        "selection_manifest": resolve(contract["selection_manifest"]),
        "config": config_path,
        "policy": policy_path,
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        return failed_report(policy["experiment_id"], missing)

    log_text = paths["train_log"].read_text(encoding="utf-8", errors="replace")
    rows, signals = parse_steps(log_text)
    step_checks = evaluate_steps(
        rows, expected_steps, max_response_tokens, warmup_steps
    )
    guard = read_json(paths["guard_summary"])
    live_gate = read_json(paths["live_launch_gate"])
    invocation = read_json(paths["run_invocation"])
    preflight = read_json(paths["training_preflight"])
    selection = read_json(paths["selection_manifest"])
    checkpoint = validate_checkpoint(output_dir, policy)
    telemetry = telemetry_summary(output_dir)
    cgroup_below, cgroup_oom_stable, cgroup = cgroup_checks(
        output_dir, float(policy["memory"]["cgroup_used_ratio_abort"])
    )

    log_prob_files = (
        [
            path
            for path in (output_dir / "evidence/log_probs").rglob("*.pt")
            if path.is_file() and path.stat().st_size > 0
        ]
        if (output_dir / "evidence/log_probs").is_dir()
        else []
    )
    manifest_ids = [str(item.get("sample_id", "")) for item in selection.get("samples", [])]
    post_warmup_rows = [row for row in rows if int(row["step"]) > warmup_steps]
    expected_post_warmup = int(contract["required_post_warmup_steps"])
    response_rows_complete = bool(rows) and all(
        finite(row.get("response_max_tokens"))
        and finite(row.get("response_mean_tokens"))
        and finite(row.get("response_clip_ratio"))
        for row in rows
    )
    peak_gpu_ratios = [
        float(value["used_ratio"])
        for value in telemetry.get("peak_by_gpu", {}).values()
        if finite(value.get("used_ratio"))
    ]

    checks = {
        "guard_pass": guard.get("status") == "PASS",
        "live_launch_gate_pass": (
            live_gate.get("status") == "PASS"
            and all(live_gate.get("live_checks", {}).values())
        ),
        "source_bindings_current": source_bindings_current(live_gate),
        "training_preflight_pass": preflight.get("status") == "PASS",
        "run_invocation_matches": (
            invocation.get("experiment_id") == policy["experiment_id"]
            and invocation.get("config_sha256") == contract["expected_config_sha256"]
            and invocation.get("train_file_sha256")
            == selection.get("output", {}).get("sha256")
            and invocation.get("sample_ids") == manifest_ids
            and invocation.get("training_contract", {}).get("total_optimizer_steps")
            == expected_steps
        ),
        "checkpoint_io_revision_matches": checkpoint_io_matches(invocation),
        "no_unapproved_hydra_overrides": invocation.get("hydra_overrides") == [],
        "no_duplicate_metric_steps": signals["duplicate_metric_steps"] == 0,
        "no_traceback_or_oom": (
            signals["traceback_count"] == 0
            and signals["cuda_oom_count"] == 0
            and signals["out_of_memory_error_count"] == 0
            and signals["dataloader_worker_killed_count"] == 0
            and signals["checkpoint_save_failure_count"] == 0
        ),
        "log_prob_evidence_complete": len(log_prob_files) >= expected_steps * 2,
        "checkpoint_complete_at_step_16": checkpoint.get("status") == "PASS",
        "telemetry_complete_two_gpus": telemetry.get("status") == "PASS",
        "gpu_peak_below_abort_line": (
            len(peak_gpu_ratios) == 2
            and max(peak_gpu_ratios)
            < float(policy["memory"]["gpu_used_ratio_abort"])
        ),
        "cgroup_peak_below_abort_line": cgroup_below,
        "cgroup_oom_counters_stable": cgroup_oom_stable,
        "post_warmup_steps_complete": (
            len(post_warmup_rows) == expected_post_warmup == 6
            and [int(row["step"]) for row in post_warmup_rows]
            == list(range(warmup_steps + 1, expected_steps + 1))
            and all(
                finite(row.get("learning_rate"))
                and float(row["learning_rate"]) > 0
                and finite(row.get("student_optimizer_delta"))
                and float(row["student_optimizer_delta"]) > 0
                for row in post_warmup_rows
            )
        ),
        "normal_eos_config_bound": (
            config["rollout"]["ignore_eos"] is False
            and config["validation"]["normal_eos_required"] is True
            and config["validation"]["forced_min_response_tokens"] is None
            and "diagnostic_generation" not in config
        ),
        "natural_eos_observed": (
            response_rows_complete
            and all(float(row["response_clip_ratio"]) == 0 for row in rows)
            and any(float(row["response_max_tokens"]) < max_response_tokens for row in rows)
        ),
        "formal_training_not_authorized": (
            config["validation"]["formal_training_authorized"] is False
            and contract["formal_training_authorized"] is False
        ),
        **step_checks,
    }
    training_gate_pass = all(
        passed
        for name, passed in checks.items()
        if name not in {"formal_training_not_authorized"}
    )
    validation_gate_pass = training_gate_pass and checks["formal_training_not_authorized"]
    failed = sorted(name for name, passed in checks.items() if not passed)
    status = "PASS_CANDIDATE_VALIDATION" if validation_gate_pass else "FAIL"

    response_summary = {
        "limit_tokens": max_response_tokens,
        "mean_of_step_means": (
            sum(float(row["response_mean_tokens"]) for row in rows) / len(rows)
            if response_rows_complete
            else None
        ),
        "observed_max_tokens": (
            max(float(row["response_max_tokens"]) for row in rows)
            if response_rows_complete
            else None
        ),
        "steps_with_clip_ratio_nonzero": (
            [
                int(row["step"])
                for row in rows
                if finite(row.get("response_clip_ratio"))
                and float(row["response_clip_ratio"]) != 0
            ]
            if rows
            else []
        ),
    }
    return {
        "schema_version": 1,
        "experiment_id": policy["experiment_id"],
        "status": status,
        "training_gate_pass": training_gate_pass,
        "validation_gate_pass": validation_gate_pass,
        "formal_training_authorized": False,
        "checks": checks,
        "failed_checks": failed,
        "signals": signals,
        "observed_steps": len(rows),
        "expected_steps": expected_steps,
        "steps": rows,
        "warmup_contract": {
            "lr_warmup_steps": warmup_steps,
            "required_post_warmup_steps": expected_post_warmup,
            "observed_post_warmup_steps": [
                int(row["step"]) for row in post_warmup_rows
            ],
        },
        "response_length": response_summary,
        "checkpoint": checkpoint,
        "telemetry": telemetry,
        "cgroup": cgroup,
        "log_prob_file_count": len(log_prob_files),
        "checkpoint_io_contract": invocation.get("checkpoint_io_contract"),
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    checks = "\n".join(
        f"| `{name}` | {'PASS' if passed else 'FAIL'} |"
        for name, passed in report.get("checks", {}).items()
    )
    response = report.get("response_length", {})
    return f"""# 128×16 Formal Candidate Validation Postflight

- 状态：**{report['status']}**
- 训练 Gate：`{str(report['training_gate_pass']).lower()}`
- 候选验证 Gate：`{str(report['validation_gate_pass']).lower()}`
- 正式训练授权：`false`
- 观测步骤：`{report.get('observed_steps', 0)}/{report.get('expected_steps', 16)}`
- 回复最大长度：`{response.get('observed_max_tokens')} / {response.get('limit_tokens', 1024)}`

| 检查 | 结果 |
| --- | --- |
{checks}

此报告只决定 128×16 候选验证是否通过，不会自行放行 6,241 条正式训练。
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    args = parser.parse_args()
    policy_path = resolve(args.policy)
    _policy, contract = load_candidate_policy(policy_path)
    output_dir = resolve(contract["output_dir"])
    report = audit(policy_path)
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
    print(f"CANDIDATE_128_POSTFLIGHT={report['status']}")
    print(f"TRAINING_GATE_PASS={report['training_gate_pass']}")
    print(f"VALIDATION_GATE_PASS={report['validation_gate_pass']}")
    print("FORMAL_TRAINING_AUTHORIZED=false")
    print(f"OUTPUT={json_path.resolve()}")
    return 0 if report["validation_gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
