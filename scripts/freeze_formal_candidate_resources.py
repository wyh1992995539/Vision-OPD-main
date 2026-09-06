#!/usr/bin/env python3
"""Freeze candidate-validation, disk, and budget evidence for the formal Gate.

This is a CPU-only, fail-closed evidence builder.  It never promotes a config
and never authorizes or starts training.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path
import shutil
import statistics
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "artifacts/runs/E-D11-6K-GATE-001"
DEFAULT_POSTFLIGHT = RUN_ROOT / "formal_candidate_validation_v1/run/evidence/postflight.json"
DEFAULT_VALIDATION_POLICY = ROOT / "configs/vopd_6241_candidate_128_abort_policy.yaml"
DEFAULT_CANDIDATE = ROOT / "configs/vopd_6241_candidate.yaml"
DEFAULT_FORMAL_POLICY = ROOT / "configs/vopd_6241_abort_policy.yaml"
DEFAULT_OUTPUT = RUN_ROOT / "formal_candidate_validation_v1/formal_gate_freeze.json"
GIB = 1024**3


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def tree_apparent_bytes(path: Path) -> int:
    """Match ``du --apparent-size --bytes`` without invoking another process."""
    require(path.is_dir(), f"directory is missing: {path}")
    total = path.lstat().st_size
    for entry in path.rglob("*"):
        require(not entry.is_symlink(), f"symlink is not allowed in evidence tree: {entry}")
        require(entry.is_file() or entry.is_dir(), f"unsupported evidence entry: {entry}")
        total += entry.lstat().st_size
    return total


def file_payload_bytes(path: Path) -> int:
    require(path.is_dir(), f"directory is missing: {path}")
    files = [entry for entry in path.rglob("*") if entry.is_file()]
    require(files, f"evidence tree has no files: {path}")
    return sum(entry.stat().st_size for entry in files)


def source_is_current(entry: dict[str, Any]) -> bool:
    path = Path(str(entry.get("path", "")))
    return path.is_file() and entry.get("sha256") == sha256_file(path)


def _scenario(total_seconds: float, steady_seconds: float, rate: float) -> dict[str, float]:
    hours = total_seconds / 3600
    return {
        "steady_step_seconds": steady_seconds,
        "projected_total_seconds": total_seconds,
        "projected_dual_gpu_hours": hours,
        "projected_cost_cny": hours * rate,
    }


def build_freeze(
    *,
    postflight_path: Path = DEFAULT_POSTFLIGHT,
    validation_policy_path: Path = DEFAULT_VALIDATION_POLICY,
    candidate_path: Path = DEFAULT_CANDIDATE,
    formal_policy_path: Path = DEFAULT_FORMAL_POLICY,
    disk_free_bytes: int,
    generated_at: str | None = None,
) -> dict[str, Any]:
    paths = {
        "candidate_postflight": postflight_path.resolve(),
        "candidate_validation_policy": validation_policy_path.resolve(),
        "formal_candidate_config": candidate_path.resolve(),
        "formal_abort_policy": formal_policy_path.resolve(),
    }
    require(all(path.is_file() for path in paths.values()), "candidate Gate input is missing")
    report = load_json(paths["candidate_postflight"])
    validation_policy = yaml.safe_load(paths["candidate_validation_policy"].read_text(encoding="utf-8"))
    candidate = yaml.safe_load(paths["formal_candidate_config"].read_text(encoding="utf-8"))
    formal_policy = yaml.safe_load(paths["formal_abort_policy"].read_text(encoding="utf-8"))

    require(report.get("status") == "PASS_CANDIDATE_VALIDATION", "candidate postflight did not pass")
    require(report.get("training_gate_pass") is True, "candidate training Gate did not pass")
    require(report.get("validation_gate_pass") is True, "candidate validation Gate did not pass")
    require(report.get("formal_training_authorized") is False, "candidate run self-authorized training")
    require(report.get("checks") and all(report["checks"].values()), "candidate checks are incomplete")
    require(not report.get("failed_checks"), "candidate postflight contains failed checks")
    require(report.get("warmup_contract", {}).get("observed_post_warmup_steps") == list(range(11, 17)),
            "candidate lacks the six required post-warmup steps")
    require(report.get("response_length", {}).get("steps_with_clip_ratio_nonzero") == [],
            "candidate contains clipped responses")

    for entry in report.get("inputs", {}).values():
        require(source_is_current(entry), f"candidate postflight input changed: {entry.get('path')}")
    live_path = Path(report["inputs"]["live_launch_gate"]["path"])
    live = load_json(live_path)
    require(live.get("status") == "PASS" and live.get("live_checks")
            and all(live["live_checks"].values()), "candidate live launch Gate did not pass")
    require(live.get("source_bindings") and all(source_is_current(entry)
            for entry in live["source_bindings"].values()), "candidate source binding changed")

    contract = validation_policy["candidate_validation"]
    candidate_hash = sha256_file(paths["formal_candidate_config"])
    require(candidate_hash == contract["expected_source_candidate_sha256"],
            "formal candidate no longer matches the GPU-validated source")
    source_binding = live["source_bindings"]["source_candidate"]
    require(Path(source_binding["path"]).resolve() == paths["formal_candidate_config"]
            and source_binding["sha256"] == candidate_hash,
            "live Gate did not bind the formal candidate")
    require(candidate["experiment"]["id"] == formal_policy["experiment_id"] == "E-D12-6K-VOPD-001",
            "formal experiment identity changed")
    require(candidate["rollout"]["ignore_eos"] is False
            and candidate["actor"]["defer_optimizer_state_load"] is True,
            "validated deferred/normal-EOS candidate contract changed")
    require(candidate["training"]["total_optimizer_steps"] == 780
            and formal_policy["checkpoint"]["allowed_save_steps"] == [390, 780]
            and formal_policy["checkpoint"]["expected_final_step"] == 780,
            "formal step/checkpoint schedule changed")

    checkpoint = report["checkpoint"]
    checkpoint_dir = Path(checkpoint["step_directory"])
    required_files = [checkpoint_dir / relative for relative in validation_policy["checkpoint"]["required_relative_files"]]
    require(checkpoint.get("status") == "PASS" and checkpoint.get("marker_value") == "16",
            "candidate checkpoint receipt did not pass")
    require(all(path.is_file() and path.stat().st_size > 0 for path in required_files),
            "candidate checkpoint is missing a required nonempty file")
    require(validation_policy["checkpoint"]["required_relative_files"]
            == formal_policy["checkpoint"]["required_relative_files"],
            "candidate and formal checkpoint file contracts differ")
    checkpoint_payload = file_payload_bytes(checkpoint_dir)
    checkpoint_apparent = tree_apparent_bytes(checkpoint_dir)

    steps = report["steps"]
    require(len(steps) == 16 and [int(row["step"]) for row in steps] == list(range(1, 17)),
            "candidate optimizer steps are incomplete")
    step_seconds = [float(row["step_seconds"]) for row in steps]
    require(all(math.isfinite(value) and value > 0 for value in step_seconds), "invalid step timing")
    steady = step_seconds[1:]
    checkpoint_seconds = float(steps[-1]["checkpoint_save_seconds"])
    observed_seconds = float(report["telemetry"]["max_observed_elapsed_seconds"])
    startup_seconds = observed_seconds - sum(step_seconds) - checkpoint_seconds
    require(math.isfinite(checkpoint_seconds) and checkpoint_seconds > 0, "invalid checkpoint timing")
    require(math.isfinite(startup_seconds) and startup_seconds >= 0, "invalid startup residual")

    hourly_rate = float(formal_policy["budget"]["hourly_dual_gpu_rate_cny"])
    target_steps = int(candidate["training"]["total_optimizer_steps"])
    checkpoint_count = len(formal_policy["checkpoint"]["allowed_save_steps"])
    scenario_steps = {
        "median": statistics.median(steady),
        "mean": statistics.mean(steady),
        "conservative_max": max(steady),
    }
    scenarios = {}
    for name, seconds in scenario_steps.items():
        total = startup_seconds + step_seconds[0] + (target_steps - 1) * seconds + checkpoint_count * checkpoint_seconds
        scenarios[name] = _scenario(total, seconds, hourly_rate)
    require(scenarios["median"]["projected_cost_cny"]
            <= scenarios["mean"]["projected_cost_cny"]
            <= scenarios["conservative_max"]["projected_cost_cny"], "budget scenarios are not ordered")

    billing = live["billing"]
    historical_cumulative = float(billing["current_autodl_cost_cny"])
    candidate_runtime_estimate = observed_seconds / 3600 * hourly_rate
    hard_hours = float(formal_policy["runtime"]["max_wall_time_hours"])
    hard_cost = hard_hours * hourly_rate
    project_limit = float(formal_policy["budget"]["project_hard_limit_cny"])
    estimated_post_candidate = historical_cumulative + candidate_runtime_estimate

    run_dir = checkpoint_dir.parents[1]
    checkpoints_dir = checkpoint_dir.parent
    run_apparent = tree_apparent_bytes(run_dir)
    checkpoints_apparent = tree_apparent_bytes(checkpoints_dir)
    noncheckpoint_observed = run_apparent - checkpoints_apparent
    projected_noncheckpoint = math.ceil(noncheckpoint_observed / len(steps) * target_steps)
    configured_checkpoint = int(formal_policy["disk"]["checkpoint_estimate_bytes"])
    checkpoint_budget = max(checkpoint_payload, configured_checkpoint)
    reserve = int(formal_policy["disk"]["reserve_bytes"])
    formula_required = 2 * checkpoint_budget + reserve
    policy_floor = int(formal_policy["disk"]["prelaunch_required_bytes"])
    required_disk = max(formula_required, policy_floor)
    disk_checks = {
        "measured_payload_covered_by_checkpoint_budget": checkpoint_payload <= checkpoint_budget,
        "measured_apparent_two_copy_peak_below_rounded_floor": 2 * checkpoint_apparent + reserve <= policy_floor,
        "projected_noncheckpoint_evidence_covered_by_reserve": projected_noncheckpoint <= reserve,
        "policy_formula_field_consistent": int(formal_policy["disk"]["formula_required_bytes"])
        == 2 * configured_checkpoint + reserve,
        "policy_floor_covers_recomputed_formula": policy_floor >= formula_required,
        "current_disk_meets_refrozen_floor": int(disk_free_bytes) >= required_disk,
    }
    require(all(disk_checks[name] for name in disk_checks if name != "current_disk_meets_refrozen_floor"),
            "formal disk policy does not cover the candidate measurement")

    budget_pass = estimated_post_candidate + hard_cost <= project_limit
    status = "PASS_CANDIDATE_GATE_FREEZE" if disk_checks["current_disk_meets_refrozen_floor"] and budget_pass \
        else "BLOCKED_CANDIDATE_RESOURCE_GATE"
    result = {
        "schema_version": 1,
        "freeze_id": "E-D11-6K-CANDIDATE-GATE-FREEZE-001",
        "generated_at_utc": generated_at or utc_now(),
        "status": status,
        "artifact_status": "COMPLETE",
        "formal_training_authorized": False,
        "candidate_validation": {
            "status": report["status"],
            "training_gate_pass": True,
            "validation_gate_pass": True,
            "normal_eos_observed": bool(report["checks"]["natural_eos_observed"]),
            "observed_steps": len(steps),
            "post_warmup_steps": report["warmup_contract"]["observed_post_warmup_steps"],
            "maximum_response_tokens": report["response_length"]["observed_max_tokens"],
            "response_limit_tokens": report["response_length"]["limit_tokens"],
            "gpu_peak_ratio": max(float(row["used_ratio"])
                                  for row in report["telemetry"]["peak_by_gpu"].values()),
            "cpu_peak_ratio": float(report["cgroup"]["peak_used_ratio"]),
            "checkpoint_pass": checkpoint["status"] == "PASS",
            "validated_candidate_sha256": candidate_hash,
        },
        "disk": {
            "status": "PASS" if disk_checks["current_disk_meets_refrozen_floor"] else "BLOCKED",
            "checks": disk_checks,
            "checkpoint_payload_bytes": checkpoint_payload,
            "checkpoint_apparent_bytes": checkpoint_apparent,
            "configured_checkpoint_estimate_bytes": configured_checkpoint,
            "checkpoint_budget_bytes": checkpoint_budget,
            "checkpoint_copies_at_write_peak": 2,
            "reserve_bytes": reserve,
            "candidate_noncheckpoint_observed_bytes": noncheckpoint_observed,
            "formal_noncheckpoint_projected_bytes": projected_noncheckpoint,
            "formula_required_bytes": formula_required,
            "policy_rounded_floor_bytes": policy_floor,
            "refrozen_prelaunch_required_bytes": required_disk,
            "snapshot_free_bytes": int(disk_free_bytes),
            "snapshot_headroom_bytes": int(disk_free_bytes) - required_disk,
            "formula": "max(policy_floor, 2 * checkpoint_budget + reserve)",
        },
        "budget": {
            "status": "PASS_BUDGET_REFROZEN_FROM_NATURAL_CANDIDATE" if budget_pass else "BLOCKED_PROJECT_CAP",
            "currency": "CNY",
            "target_optimizer_steps": target_steps,
            "checkpoint_count": checkpoint_count,
            "hourly_dual_gpu_rate_cny": hourly_rate,
            "startup_seconds_estimate": startup_seconds,
            "first_step_seconds": step_seconds[0],
            "candidate_checkpoint_seconds": checkpoint_seconds,
            "steady_step_sample_count": len(steady),
            "steady_step_source": "candidate steps 2-16 under natural EOS",
            "formula": "startup + first_step + 779 * steady_step + 2 * checkpoint_save",
            "scenarios": scenarios,
            "selected": {
                "planning_scenario": "mean",
                "planning_hours": scenarios["mean"]["projected_dual_gpu_hours"],
                "planning_incremental_cost_cny": scenarios["mean"]["projected_cost_cny"],
                "reservation_scenario": "conservative_max",
                "reservation_hours": scenarios["conservative_max"]["projected_dual_gpu_hours"],
                "reservation_incremental_cost_cny": scenarios["conservative_max"]["projected_cost_cny"],
                "hard_abort_ceiling_hours": hard_hours,
                "hard_abort_ceiling_incremental_cost_cny": hard_cost,
            },
            "project_cap": {
                "hard_limit_cny": project_limit,
                "historical_pre_candidate_cumulative_cny": historical_cumulative,
                "historical_observation_at_utc": billing["observed_at_utc"],
                "candidate_runtime_cost_estimate_cny": candidate_runtime_estimate,
                "estimated_post_candidate_cumulative_cny": estimated_post_candidate,
                "estimated_plus_reservation_cny": estimated_post_candidate
                + scenarios["conservative_max"]["projected_cost_cny"],
                "estimated_plus_hard_abort_ceiling_cny": estimated_post_candidate + hard_cost,
                "budget_pass": budget_pass,
                "launch_value_fresh": False,
                "launch_requirement": "Refresh cumulative AutoDL charge and UTC time immediately before formal --run.",
            },
        },
        "input_paths": {name: str(path) for name, path in paths.items()},
        "sources": {
            **{name: {"path": str(path), "sha256": sha256_file(path)} for name, path in paths.items()},
            "candidate_live_launch_gate": {"path": str(live_path.resolve()), "sha256": sha256_file(live_path)},
            "freeze_builder": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__))},
        },
        "limits": [
            "The 340 CNY observation predates the candidate run and is not reusable as formal-launch billing evidence.",
            "The candidate runtime cost is a rate-based estimate, not a provider invoice.",
            "The 120 GiB disk floor is a launch floor; the guarded launcher must recheck actual free bytes.",
            "This freeze validates the candidate Gate input but does not promote a formal config or authorize training.",
        ],
    }
    return result


def verify(path: Path = DEFAULT_OUTPUT) -> bool:
    try:
        path = Path(path).resolve()
        value = load_json(path)
        require(value.get("artifact_status") == "COMPLETE", "incomplete candidate freeze")
        require(value.get("formal_training_authorized") is False, "candidate freeze self-authorized training")
        rebuilt = build_freeze(
            postflight_path=Path(value["input_paths"]["candidate_postflight"]),
            validation_policy_path=Path(value["input_paths"]["candidate_validation_policy"]),
            candidate_path=Path(value["input_paths"]["formal_candidate_config"]),
            formal_policy_path=Path(value["input_paths"]["formal_abort_policy"]),
            disk_free_bytes=int(value["disk"]["snapshot_free_bytes"]),
            generated_at=value["generated_at_utc"],
        )
        return rebuilt == value
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False


def render_markdown(value: dict[str, Any]) -> str:
    disk = value["disk"]
    budget = value["budget"]
    candidate = value["candidate_validation"]
    scenarios = "\n".join(
        f"| {name} | {row['steady_step_seconds']:.2f} | {row['projected_dual_gpu_hours']:.2f} | {row['projected_cost_cny']:.2f} |"
        for name, row in budget["scenarios"].items()
    )
    return f"""# 128×16 正式候选 Gate、磁盘与预算冻结

- 状态：**{value['status']}**
- 候选验证：`{candidate['status']}`；16/16 步，正常 EOS，最长回复 {candidate['maximum_response_tokens']:.0f}/1024。
- GPU / CPU 峰值：{candidate['gpu_peak_ratio']:.2%} / {candidate['cpu_peak_ratio']:.2%}。
- 正式训练授权：`false`

## 正式训练预算重算

| 情景 | 稳态秒/步 | 双卡小时 | 增量费用（元） |
| --- | ---: | ---: | ---: |
{scenarios}

- 计划：{budget['selected']['planning_hours']:.2f} 小时 / {budget['selected']['planning_incremental_cost_cny']:.2f} 元。
- 保守预留：{budget['selected']['reservation_hours']:.2f} 小时 / {budget['selected']['reservation_incremental_cost_cny']:.2f} 元。
- 38 小时硬中止费用：{budget['selected']['hard_abort_ceiling_incremental_cost_cny']:.2f} 元。
- 340 元是候选启动前历史值；正式启动必须重新读取平台累计费用。

## 磁盘重算

- checkpoint payload / apparent size：{disk['checkpoint_payload_bytes']} / {disk['checkpoint_apparent_bytes']} bytes。
- 原始双份公式：{disk['formula_required_bytes'] / GIB:.2f} GiB。
- 圆整后的正式启动门槛：{disk['refrozen_prelaunch_required_bytes'] / GIB:.2f} GiB。
- 冻结时可用：{disk['snapshot_free_bytes'] / GIB:.2f} GiB；余量：{disk['snapshot_headroom_bytes'] / GIB:.2f} GiB。

此冻结只提供正式 Gate 输入；不会自动修改或放行正式配置。
"""


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--postflight", type=Path, default=DEFAULT_POSTFLIGHT)
    parser.add_argument("--validation-policy", type=Path, default=DEFAULT_VALIDATION_POLICY)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--formal-policy", type=Path, default=DEFAULT_FORMAL_POLICY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--disk-path", type=Path, default=ROOT.parent)
    args = parser.parse_args()
    value = build_freeze(
        postflight_path=args.postflight,
        validation_policy_path=args.validation_policy,
        candidate_path=args.candidate,
        formal_policy_path=args.formal_policy,
        disk_free_bytes=shutil.disk_usage(args.disk_path.resolve()).free,
    )
    output = args.output.resolve()
    write_atomic(output, json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    write_atomic(output.with_suffix(".md"), render_markdown(value))
    write_atomic(output.with_suffix(".sha256"),
                 f"{sha256_file(output)}  {output}\n{sha256_file(output.with_suffix('.md'))}  {output.with_suffix('.md')}\n")
    print(f"CANDIDATE_GATE_FREEZE={value['status']}")
    print(f"DISK_REQUIRED_GIB={value['disk']['refrozen_prelaunch_required_bytes'] / GIB:.2f}")
    print(f"BUDGET_RESERVATION_CNY={value['budget']['selected']['reservation_incremental_cost_cny']:.2f}")
    print("FORMAL_TRAINING_AUTHORIZED=false")
    print(f"OUTPUT={output}")
    return 0 if value["artifact_status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
