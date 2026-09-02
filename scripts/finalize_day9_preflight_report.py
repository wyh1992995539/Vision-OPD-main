#!/usr/bin/env python3
"""Build the auditable Day 9 Task 4 formal-training preflight report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.day9_formal_training_readiness import GIB, sha256_file, utc_now, write_json, write_text


EXPERIMENT_ID = "E-D10-001"
SOURCE_PATHS = {
    "budget": "artifacts/runs/E-D10-001/preflight/budget_projection.json",
    "data": "artifacts/runs/E-D10-001/preflight/data_manifest.json",
    "base": "artifacts/runs/E-D10-001/preflight/base_model_manifest.json",
    "launcher": "artifacts/runs/E-D10-001/preflight/preflight_summary.json",
    "readiness": "artifacts/runs/E-D10-001/preflight/task2_readiness.json",
    "config_freeze": "artifacts/runs/E-D10-001/preflight/task3_config_freeze.json",
    "day8_stability": "artifacts/runs/E-D8-001/evidence/stability_summary.json",
}


def load_sources(project_root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, str]]]:
    values: dict[str, dict[str, Any]] = {}
    provenance: dict[str, dict[str, str]] = {}
    for name, relative in SOURCE_PATHS.items():
        path = project_root / relative
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"{relative}: expected a JSON object")
        values[name] = value
        provenance[name] = {"path": relative, "sha256": sha256_file(path)}
    return values, provenance


def build_payload(
    sources: dict[str, dict[str, Any]],
    provenance: dict[str, dict[str, str]],
    generated_at: str | None = None,
) -> dict[str, Any]:
    budget = sources["budget"]
    data = sources["data"]
    base = sources["base"]
    launcher = sources["launcher"]
    readiness = sources["readiness"]
    freeze = sources["config_freeze"]
    day8 = sources["day8_stability"]

    config_hashes = {
        "task3_freeze": freeze["sources"]["formal_config"]["sha256"],
        "launcher_preflight": launcher["config_sha256"],
    }
    source_gates = {
        "readiness_pass": readiness["readiness_status"] == "PASS" and not readiness["blocking_gates"],
        "all_readiness_gates_pass": all(gate["status"] == "PASS" for gate in readiness["gates"].values()),
        "task3_config_pass": freeze["config_gate_status"] == "PASS" and freeze["task3_completed"],
        "config_hashes_match": len(set(config_hashes.values())) == 1,
        "launcher_preflight_pass": launcher["status"] == "PASS" and all(launcher["checks"].values()),
        "launcher_used_no_gpu": launcher["gpu_used"] is False,
        "data_pass": data["status"] == "PASS",
        "base_pass": base["status"] == "PASS",
        "budget_pass": budget["gate_status"] == "PASS",
        "day8_cold_reload_pass": day8["cold_reload"]["status"] == "PASS",
    }
    failed = [name for name, passed in source_gates.items() if not passed]
    if failed:
        raise ValueError(f"Task 4 source gates failed: {', '.join(failed)}")

    storage = readiness["storage"]
    storage_margin = int(storage["available_bytes"]) - int(storage["required_bytes"])
    storage_class = "PASS_WITH_LOW_HEADROOM" if storage_margin < 5 * GIB else "PASS"
    storage_risk = (
        {
            "severity": "HIGH",
            "status": storage_class,
            "risk": "Storage passes the frozen formula with limited additional headroom.",
            "evidence": f"margin={storage_margin} bytes after 2 x checkpoint + 5 GiB",
            "handoff": "Task 5 must monitor filesystem free space and abort before the reserve is consumed.",
        }
        if storage_class == "PASS_WITH_LOW_HEADROOM"
        else {
            "severity": "INFO",
            "status": storage_class,
            "risk": "Storage exceeds the frozen checkpoint-retention formula with additional headroom.",
            "evidence": f"margin={storage_margin} bytes after 2 x checkpoint + 5 GiB",
            "handoff": "Retain filesystem monitoring in Task 5 because checkpoint writes are still large.",
        }
    )
    planning = budget["selected_budget"]["planning"]
    reservation = budget["selected_budget"]["reservation"]
    cap = budget["project_cap"]
    contract = freeze["formal_contract"]

    gate_rows = [
        {"gate": name, "status": gate["status"], "evidence": gate["evidence"]}
        for name, gate in readiness["gates"].items()
    ]
    gate_rows.extend(
        [
            {
                "gate": "launcher_preflight",
                "status": "PASS",
                "evidence": f"{launcher['train_rows']} rows; missing images={len(launcher['missing_image_paths'])}; gpu_used=false",
            },
            {
                "gate": "config_hash_identity",
                "status": "PASS",
                "evidence": config_hashes["task3_freeze"],
            },
            {
                "gate": "day8_cold_reload",
                "status": "PASS",
                "evidence": f"{day8['cold_reload']['prediction_count']} predictions; {day8['cold_reload']['inference_error_count']} inference errors",
            },
        ]
    )

    risks = [
        storage_risk,
        {
            "severity": "MEDIUM",
            "status": "MITIGATED_REQUIRES_MONITORING",
            "risk": "Day 8 recorded one DataLoader worker Killed event after checkpoint save.",
            "evidence": "Formal config sets data.dataloader_num_workers=0.",
            "handoff": "Task 5 must capture trainer RSS and cgroup memory events.",
        },
        {
            "severity": "MEDIUM",
            "status": "REFRESH_BEFORE_LAUNCH",
            "risk": "The controlling AutoDL cumulative charge is a user-reported point-in-time value.",
            "evidence": f"recorded current={cap['current_platform_cumulative_charge_cny']} CNY",
            "handoff": "Refresh the console value immediately before Day 10 launch.",
        },
        {
            "severity": "HIGH",
            "status": "OPEN_TASK5",
            "risk": "Training abort conditions and monitoring are not yet frozen as executable controls.",
            "evidence": "Day 9 Task 5 remains incomplete.",
            "handoff": "Complete Task 5 before authorizing the --run command.",
        },
    ]
    return {
        "schema_version": 1,
        "generated_at_utc": generated_at or utc_now(),
        "experiment_id": EXPERIMENT_ID,
        "purpose": "day9_task4_formal_training_preflight_report",
        "artifact_status": "COMPLETE",
        "report_status": "PASS_TO_TASK5",
        "task4_completed": True,
        "advance_to_task5": True,
        "advance_to_day10": False,
        "day10_block_reason": "Task 5 executable abort conditions and monitoring remain to be frozen.",
        "audited_source_commit": readiness["git"]["commit"],
        "audited_source_worktree_clean": readiness["git"]["clean"],
        "source_gates": source_gates,
        "config_hashes": config_hashes,
        "training_contract": contract,
        "gate_rows": gate_rows,
        "time_and_cost": {
            "currency": "CNY",
            "planning_dual_gpu_hours": planning["dual_gpu_hours"],
            "planning_cost_cny": planning["estimated_cost_cny"],
            "reservation_dual_gpu_hours": reservation["dual_gpu_hours"],
            "reservation_cost_cny": reservation["estimated_cost_cny"],
            "current_autodl_cumulative_charge_cny": cap["current_platform_cumulative_charge_cny"],
            "projected_total_after_reservation_cny": cap["projected_total_after_e_d10_reservation_cny"],
            "remaining_after_reservation_cny": cap["remaining_after_e_d10_reservation_cny"],
        },
        "storage": {
            **storage,
            "margin_above_required_bytes": storage_margin,
            "report_classification": storage_class,
        },
        "risks": risks,
        "day8_caveats": day8["caveats"],
        "commands": {
            "preflight_only": freeze["commands"]["preflight_only"],
            "formal_training_after_task5_and_launch_refresh": freeze["commands"]["formal_training_after_all_gates_pass"],
        },
        "sources": provenance,
        "visual_omission_reason": "Readiness evidence is a point-in-time set of discrete gates, not a time series; tables are more faithful than a chart.",
    }


def build_markdown(payload: dict[str, Any]) -> str:
    contract = payload["training_contract"]
    cost = payload["time_and_cost"]
    storage = payload["storage"]
    gates = "\n".join(
        f"| {row['gate']} | {row['status']} | {row['evidence']} |"
        for row in payload["gate_rows"]
    )
    risks = "\n".join(
        f"| {row['severity']} | {row['status']} | {row['risk']} | {row['handoff']} |"
        for row in payload["risks"]
    )
    sources = "\n".join(
        f"- `{name}`：`{value['path']}`，SHA256 `{value['sha256']}`"
        for name, value in payload["sources"].items()
    )
    if storage["report_classification"] == "PASS_WITH_LOW_HEADROOM":
        storage_summary = (
            f"公式之外仅剩 {storage['margin_above_required_bytes']} bytes（约 "
            f"{storage['margin_above_required_bytes'] / GIB:.2f} GiB），因此标记为 "
            "`PASS_WITH_LOW_HEADROOM`。这不是容量 FAIL，但要求训练期持续监控。"
        )
    else:
        storage_summary = (
            f"公式之外另有 {storage['margin_above_required_bytes']} bytes（约 "
            f"{storage['margin_above_required_bytes'] / GIB:.2f} GiB）余量，容量 Gate 为 `PASS`。"
            "checkpoint 写入量仍然较大，因此任务 5 继续保留磁盘监控。"
        )
    return f"""# E-D10-001 基础 Gate 已通过，Day 10 仍等待中止条件冻结

> 生成时间：{payload['generated_at_utc']}
> Task 4：**{payload['artifact_status']}**
> 决策：**{payload['report_status']}**
> GPU 使用：**false**

## 结论

E-D10-001 的数据、Base、正式配置、预算、Git、输出目录、日志路径、磁盘和 CPU-only launcher preflight 均已通过。任务 4 已完成，可以进入任务 5；当前仍不得执行正式训练，因为训练中止条件和观测性控制尚未冻结。

磁盘满足项目定义的 `2 × 最终 checkpoint 估算 + 5 GiB`。{storage_summary}

## 正式训练合同

| 项目 | 冻结值 |
|---|---:|
| 实验 | {payload['experiment_id']} |
| 样本数 | {contract['expected_samples']} |
| global batch | {contract['global_batch_size']} |
| optimizer steps | {contract['optimizer_steps']} |
| epochs | {contract['total_epochs']} |
| 完整 epoch | {contract['require_full_epoch']} |
| 配置 SHA256 | `{payload['config_hashes']['task3_freeze']}` |

配置从冻结 Base 冷启动，`resume_mode=disable`；`dataloader_num_workers=0`；只保留最终 checkpoint。

## Gate 证据

| Gate | 状态 | 证据 |
|---|---|---|
{gates}

审计源提交为 `{payload['audited_source_commit']}`，审计开始时工作树 clean=`{payload['audited_source_worktree_clean']}`。本报告生成后产生的新文件需要单独提交，最终 clean 状态在提交后复核。

## 时间与预算

| 口径 | 双卡小时 | 费用 |
|---|---:|---:|
| 均值规划 | {cost['planning_dual_gpu_hours']:.4f} | ¥{cost['planning_cost_cny']:.2f} |
| 保守预留 | {cost['reservation_dual_gpu_hours']:.4f} | ¥{cost['reservation_cost_cny']:.2f} |

- 用户报告的 AutoDL 累计费用：¥{cost['current_autodl_cumulative_charge_cny']:.2f}。
- 加入保守预留后的预计累计费用：¥{cost['projected_total_after_reservation_cny']:.2f}。
- 项目预算预计剩余：¥{cost['remaining_after_reservation_cny']:.2f}。
- 该累计费用是点时值；Day 10 启动前必须刷新 AutoDL 控制台。

## 磁盘口径

- Day 8 checkpoint：{storage['day8_checkpoint_size_bytes']} bytes。
- 冻结公式要求：{storage['required_bytes']} bytes。
- 当前可用：{storage['available_bytes']} bytes。
- 高于最低要求：{storage['margin_above_required_bytes']} bytes。

## 风险与任务 5 交接

| 严重度 | 状态 | 风险 | 交接动作 |
|---|---|---|---|
{risks}

Day 8 的 checkpoint 已完成 5/5 冷重载推理且没有 inference error。DataLoader worker `Killed` 和不可审计的逐卡显存峰值仍作为观测性 caveat 保留，不改写为已解决。

## 可复制命令

允许重复执行的 CPU-only preflight：

```bash
{payload['commands']['preflight_only']}
```

正式训练命令已经冻结，但只有 Task 5 完成、AutoDL 费用刷新且所有 Gate 仍为 PASS 后才允许执行：

```bash
{payload['commands']['formal_training_after_task5_and_launch_refresh']}
```

## 证据来源

{sources}

未绘制趋势图：这些证据是离散的单次 readiness Gate，不是连续时间序列；表格能更准确地保留状态、单位和来源。
"""


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--generated-at")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    output = (args.output or project_root / "artifacts/runs/E-D10-001/preflight.md").resolve()
    json_output = (args.json_output or project_root / "artifacts/runs/E-D10-001/preflight/task4_preflight_report.json").resolve()
    sources, provenance = load_sources(project_root)
    payload = build_payload(sources, provenance, args.generated_at)
    write_json(json_output, payload)
    write_text(output, build_markdown(payload))
    print(f"ARTIFACT_STATUS={payload['artifact_status']}")
    print(f"REPORT_STATUS={payload['report_status']}")
    print(f"REPORT={output}")
    print(f"REPORT_JSON={json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
