#!/usr/bin/env python3
"""Freeze the Pilot-64-derived budget for the 6,241-row formal run."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "artifacts/runs/E-D11-6K-GATE-001"
DEFAULT_POSTFLIGHT = RUN_ROOT / "pilot/64/evidence/postflight.json"
DEFAULT_POLICY = ROOT / "configs/vopd_6241_abort_policy.yaml"
DEFAULT_BILLING = RUN_ROOT / "pilot/billing_observation.json"
DEFAULT_OUTPUT = RUN_ROOT / "budget_freeze.json"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def validate_scenario(name: str, source: dict[str, Any], hourly_rate: float) -> dict[str, float]:
    seconds = float(source["projected_total_seconds"])
    hours = float(source["projected_dual_gpu_hours"])
    cost = float(source["projected_cost_cny"])
    if min(seconds, hours, cost) <= 0:
        raise ValueError(f"{name}: projection values must be positive")
    if not math.isclose(hours, seconds / 3600, rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError(f"{name}: hours do not match seconds")
    if not math.isclose(cost, hours * hourly_rate, rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError(f"{name}: cost does not match the policy hourly rate")
    return {
        "steady_step_seconds": float(source["steady_step_seconds"]),
        "projected_total_seconds": seconds,
        "projected_dual_gpu_hours": hours,
        "projected_cost_cny": cost,
    }


def build_budget(
    postflight: dict[str, Any],
    policy: dict[str, Any],
    billing: dict[str, Any],
    *,
    postflight_path: Path,
    policy_path: Path,
    billing_path: Path,
    generated_at: str | None = None,
) -> dict[str, Any]:
    if postflight.get("experiment_id") != "E-D11-6K-VOPD-PILOT-064":
        raise ValueError("wrong Pilot-64 experiment ID")
    if not (postflight.get("status") == "PASS" and postflight.get("stage_gate_pass") is True):
        raise ValueError("Pilot-64 stage Gate has not passed")
    if int(postflight.get("observed_steps", 0)) != 8:
        raise ValueError("Pilot-64 evidence must contain exactly 8 optimizer steps")

    projection = postflight.get("projection_780", {})
    if (projection.get("status") != "MEASURED_PROJECTION_NOT_YET_FROZEN"
            or int(projection.get("target_optimizer_steps", 0)) != 780
            or int(projection.get("checkpoint_count", 0)) != 2):
        raise ValueError("invalid Pilot-64 780-step projection")
    hourly_rate = float(policy["budget"]["hourly_dual_gpu_rate_cny"])
    names = ("median", "mean", "conservative_max")
    scenarios = {
        name: validate_scenario(name, projection["scenarios"][name], hourly_rate)
        for name in names
    }
    if not (
        scenarios["median"]["projected_cost_cny"]
        <= scenarios["mean"]["projected_cost_cny"]
        <= scenarios["conservative_max"]["projected_cost_cny"]
    ):
        raise ValueError("projection scenarios are not ordered conservatively")

    steps = postflight.get("steps", [])
    max_response = max((float(row.get("response_max_tokens", 0)) for row in steps), default=0.0)
    warmup_steps = int(postflight.get("warmup_contract", {}).get("lr_warmup_steps", 0))
    observed_step_max = max((int(row.get("step", 0)) for row in steps), default=0)
    gpu_peak = max(
        (float(row.get("used_ratio", 0)) for row in postflight["telemetry"]["peak_by_gpu"].values()),
        default=0.0,
    )
    abort_ratio = float(policy["memory"]["gpu_used_ratio_abort"])
    hard_hours = float(policy["runtime"]["max_wall_time_hours"])
    hard_cost = hard_hours * hourly_rate
    cap = float(policy["budget"]["project_hard_limit_cny"])
    historical_current = float(billing["current_autodl_cumulative_charge_cny"])

    return {
        "schema_version": 1,
        "budget_id": "E-D11-6K-VOPD-BUDGET-001",
        "generated_at_utc": generated_at or utc_now(),
        "experiment_id": "E-D12-6K-VOPD-001",
        "status": "PASS_BUDGET_FROZEN_WITH_RESOURCE_CAVEATS",
        "formal_training_authorized": False,
        "currency": "CNY",
        "target_optimizer_steps": 780,
        "dual_gpu_hourly_rate_cny": hourly_rate,
        "scenarios": scenarios,
        "selected_budget": {
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
            "hard_limit_cny": cap,
            "historical_user_reported_cumulative_cny": historical_current,
            "historical_observation_at_utc": billing["recorded_at_utc"],
            "historical_plus_reservation_cny": historical_current
            + scenarios["conservative_max"]["projected_cost_cny"],
            "historical_plus_hard_abort_ceiling_cny": historical_current + hard_cost,
            "budget_pass": historical_current + hard_cost <= cap,
            "launch_value_fresh": False,
            "launch_requirement": "Refresh cumulative AutoDL charge and UTC time immediately before formal --run.",
        },
        "coverage": {
            "pilot_optimizer_steps": observed_step_max,
            "warmup_steps": warmup_steps,
            "post_warmup_steps_observed": max(0, observed_step_max - warmup_steps),
            "configured_response_limit_tokens": 1024,
            "maximum_response_tokens_observed": max_response,
            "maximum_gpu_used_ratio_observed": gpu_peak,
            "gpu_abort_ratio": abort_ratio,
            "gpu_peak_below_abort_ratio": gpu_peak < abort_ratio,
        },
        "caveats": [
            "This freezes a measured extrapolation, not a provider billing guarantee.",
            "Pilot-64 ended before step 10 warmup completed.",
            "The longest observed response was shorter than the configured 1024-token limit.",
            "At least one GPU peak exceeded the 98% runtime abort ratio, although the configured three-sample streak did not trigger.",
            "The historical cumulative charge is expired launch evidence and is not reusable for formal launch authorization.",
        ],
        "sources": {
            "pilot_64_postflight": {"path": str(postflight_path), "sha256": sha256_file(postflight_path)},
            "formal_abort_policy": {"path": str(policy_path), "sha256": sha256_file(policy_path)},
            "historical_billing_observation": {"path": str(billing_path), "sha256": sha256_file(billing_path)},
        },
    }


def render_markdown(value: dict[str, Any]) -> str:
    selected = value["selected_budget"]
    coverage = value["coverage"]
    scenarios = "\n".join(
        f"| {name} | {row['projected_dual_gpu_hours']:.2f} | {row['projected_cost_cny']:.2f} |"
        for name, row in value["scenarios"].items()
    )
    caveats = "\n".join(f"- {item}" for item in value["caveats"])
    return f"""# Vision-OPD 6241 正式训练预算冻结

- 状态：**{value['status']}**
- 计划值：{selected['planning_hours']:.2f} 双卡小时 / {selected['planning_incremental_cost_cny']:.2f} 元
- 资源预留：{selected['reservation_hours']:.2f} 双卡小时 / {selected['reservation_incremental_cost_cny']:.2f} 元
- 38 小时中止上限：{selected['hard_abort_ceiling_incremental_cost_cny']:.2f} 元
- 正式训练授权：`false`

| 情景 | 双卡小时 | 增量费用（元） |
| --- | ---: | ---: |
{scenarios}

## 覆盖边界

- Pilot steps / warmup steps：{coverage['pilot_optimizer_steps']} / {coverage['warmup_steps']}
- 实测最长响应 / 配置上限：{coverage['maximum_response_tokens_observed']:.0f} / 1024 tokens
- 实测最高 GPU 比例 / 中止线：{coverage['maximum_gpu_used_ratio_observed']:.4%} / {coverage['gpu_abort_ratio']:.2%}

## 限制

{caveats}
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--postflight", type=Path, default=DEFAULT_POSTFLIGHT)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--billing", type=Path, default=DEFAULT_BILLING)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    paths = [args.postflight.resolve(), args.policy.resolve(), args.billing.resolve()]
    postflight, billing = load_json(paths[0]), load_json(paths[2])
    policy = yaml.safe_load(paths[1].read_text(encoding="utf-8"))
    value = build_budget(postflight, policy, billing, postflight_path=paths[0], policy_path=paths[1], billing_path=paths[2])
    output = args.output.resolve()
    write_atomic(output, json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    write_atomic(output.with_suffix(".md"), render_markdown(value))
    write_atomic(output.with_suffix(".sha256"), f"{sha256_file(output)}  {output}\n{sha256_file(output.with_suffix('.md'))}  {output.with_suffix('.md')}\n")
    print(f"BUDGET_FREEZE={value['status']}")
    print(f"RESERVATION_CNY={value['selected_budget']['reservation_incremental_cost_cny']:.2f}")
    print(f"FORMAL_TRAINING_AUTHORIZED={value['formal_training_authorized']}")
    print(f"OUTPUT={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
