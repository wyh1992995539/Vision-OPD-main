#!/usr/bin/env python3
"""Build the auditable Day 9 budget projection for E-D10-001."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
EXPERIMENT_ID = "E-D10-001"
TARGET_SAMPLES = 1024
GLOBAL_BATCH_SIZE = 8
TARGET_STEPS = TARGET_SAMPLES // GLOBAL_BATCH_SIZE


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    write_text_atomic(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def relative_source(project_root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(project_root)),
        "sha256": sha256_file(path),
    }


def assert_close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError(f"{label}: expected {expected}, found {actual}")


def validate_day8_projection(cost: dict[str, Any]) -> dict[str, Any]:
    if cost.get("experiment_id") != "E-D8-001":
        raise ValueError("Day 8 cost evidence has the wrong experiment ID")
    projection = cost["projection_1024"]
    if int(projection["target_samples"]) != TARGET_SAMPLES:
        raise ValueError("Day 8 projection target_samples is not 1024")
    if int(projection["global_batch_size"]) != GLOBAL_BATCH_SIZE:
        raise ValueError("Day 8 projection global_batch_size is not 8")
    if int(projection["target_optimizer_steps"]) != TARGET_STEPS:
        raise ValueError("Day 8 projection target_optimizer_steps is not 128")

    startup = float(projection["startup_seconds"])
    first_step = float(projection["first_step_seconds"])
    checkpoint_save = float(projection["checkpoint_save_seconds"])
    hourly_cost = float(projection["cost_per_dual_gpu_hour_cny"])
    if hourly_cost != 11.96:
        raise ValueError("dual-GPU hourly cost is not the frozen 11.96 CNY")

    scenarios: dict[str, dict[str, float]] = {}
    for name in ("median", "mean", "conservative_max"):
        source = projection["scenarios"][name]
        steady = float(source["steady_step_seconds"])
        total_seconds = startup + first_step + (TARGET_STEPS - 1) * steady + checkpoint_save
        hours = total_seconds / 3600.0
        estimated_cost = hours * hourly_cost
        assert_close(float(source["total_seconds"]), total_seconds, f"{name}.total_seconds")
        assert_close(float(source["dual_gpu_hours"]), hours, f"{name}.dual_gpu_hours")
        assert_close(
            float(source["estimated_cost_cny"]),
            estimated_cost,
            f"{name}.estimated_cost_cny",
        )
        scenarios[name] = {
            "steady_step_seconds": steady,
            "total_seconds": total_seconds,
            "dual_gpu_hours": hours,
            "estimated_cost_cny": estimated_cost,
        }

    if projection.get("planning_scenario") != "mean":
        raise ValueError("Day 8 planning scenario is not mean")
    if projection.get("resource_reservation_scenario") != "conservative_max":
        raise ValueError("Day 8 reservation scenario is not conservative_max")
    return {
        "startup_seconds": startup,
        "first_step_seconds": first_step,
        "steady_steps": TARGET_STEPS - 1,
        "checkpoint_save_seconds": checkpoint_save,
        "cost_per_dual_gpu_hour_cny": hourly_cost,
        "scenarios": scenarios,
        "planning_scenario": "mean",
        "resource_reservation_scenario": "conservative_max",
    }


def historical_ledger(project_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    paths = {
        "day5_projection": project_root / "artifacts/runs/E-D5-001/cost.json",
        "day6_base": project_root / "artifacts/runs/E-D6-001/base/cost.json",
        "paper_smoke_r1": project_root
        / "artifacts/runs/E-PAPER-BASEJUDGE-001/smoke/base/cost.json",
        "paper_smoke_r2": project_root
        / "artifacts/runs/E-PAPER-BASEJUDGE-001/smoke_r2/base/cost.json",
        "paper_base_r3": project_root
        / "artifacts/runs/E-PAPER-BASEJUDGE-001/base/cost.json",
        "day8": project_root / "artifacts/runs/E-D8-001/cost.json",
    }
    values = {name: load_json(path) for name, path in paths.items()}
    entries = [
        {
            "sort_order": 1,
            "record_id": "E-D5-001-projection",
            "experiment_id": "E-D5-001",
            "record_type": "projection",
            "cost_cny": float(values["day5_projection"]["scenarios"][1]["estimated_cost_cny"]),
            "included_in_documented_subtotal": False,
            "coverage": "Conservative full-evaluation projection; not observed spend.",
            "source": relative_source(project_root, paths["day5_projection"]),
        },
        {
            "sort_order": 2,
            "record_id": "E-D6-001-base",
            "experiment_id": "E-D6-001",
            "record_type": "observed_service_window",
            "cost_cny": float(values["day6_base"]["costs"]["total_cny"]),
            "included_in_documented_subtotal": True,
            "coverage": str(values["day6_base"]["basis"]),
            "source": relative_source(project_root, paths["day6_base"]),
        },
        {
            "sort_order": 3,
            "record_id": "E-PAPER-BASEJUDGE-001-smoke-r1",
            "experiment_id": "E-PAPER-BASEJUDGE-001",
            "record_type": "observed_client_window",
            "cost_cny": float(values["paper_smoke_r1"]["estimated_cost_cny"]),
            "included_in_documented_subtotal": True,
            "coverage": str(values["paper_smoke_r1"]["measurement_scope"]),
            "source": relative_source(project_root, paths["paper_smoke_r1"]),
        },
        {
            "sort_order": 4,
            "record_id": "E-PAPER-BASEJUDGE-001-smoke-r2",
            "experiment_id": "E-PAPER-BASEJUDGE-001",
            "record_type": "observed_client_window",
            "cost_cny": float(values["paper_smoke_r2"]["estimated_cost_cny"]),
            "included_in_documented_subtotal": True,
            "coverage": str(values["paper_smoke_r2"]["measurement_scope"]),
            "source": relative_source(project_root, paths["paper_smoke_r2"]),
        },
        {
            "sort_order": 5,
            "record_id": "E-PAPER-BASEJUDGE-001-base-r3",
            "experiment_id": "E-PAPER-BASEJUDGE-001",
            "record_type": "observed_client_window",
            "cost_cny": float(values["paper_base_r3"]["estimated_cost_cny"]),
            "included_in_documented_subtotal": True,
            "coverage": str(values["paper_base_r3"]["measurement_scope"]),
            "source": relative_source(project_root, paths["paper_base_r3"]),
        },
        {
            "sort_order": 6,
            "record_id": "E-D8-001-training-window",
            "experiment_id": "E-D8-001",
            "record_type": "observed_evidence_window",
            "cost_cny": float(values["day8"]["observed_windows"]["training_window_cost_cny"]),
            "included_in_documented_subtotal": True,
            "coverage": str(values["day8"]["observed_windows"]["coverage"]),
            "source": relative_source(project_root, paths["day8"]),
        },
        {
            "sort_order": 7,
            "record_id": "E-D8-001-reload-window",
            "experiment_id": "E-D8-001",
            "record_type": "observed_evidence_window",
            "cost_cny": float(values["day8"]["observed_windows"]["reload_window_cost_cny"]),
            "included_in_documented_subtotal": True,
            "coverage": str(values["day8"]["observed_windows"]["coverage"]),
            "source": relative_source(project_root, paths["day8"]),
        },
    ]
    gaps = [
        {
            "severity": "HIGH",
            "scope": "platform_total",
            "finding": "No AutoDL billing export or authoritative current cumulative charge is archived.",
            "impact": "The 2000 CNY project-cap gate cannot be marked PASS from repository evidence alone.",
            "remediation": "Record the current platform cumulative charge before Day 10 launch.",
        },
        {
            "severity": "MEDIUM",
            "scope": "E-D4-001_E-D5-001_E-D7-001",
            "finding": "Observed GPU cost files are absent for Day 4, Day 5 Smoke, and Day 7.",
            "impact": "The documented subtotal is incomplete and must not be presented as the cloud bill.",
            "remediation": "Reconcile these runs against the platform billing export.",
        },
        {
            "severity": "MEDIUM",
            "scope": "measurement_coverage",
            "finding": "Several cost records exclude startup, shutdown, idle time, failed attempts, or billing rounding.",
            "impact": "Summed records represent documented workload windows, not invoice-level spend.",
            "remediation": "Use the platform cumulative charge as the controlling budget source.",
        },
        {
            "severity": "LOW",
            "scope": "window_overlap",
            "finding": "Some paper-aligned records have duration and update time but no exact service interval.",
            "impact": "Exact overlap cannot be proven from the cost files, although runs are chronologically distinct.",
            "remediation": "Do not treat the documented subtotal as an audited invoice total.",
        },
    ]
    return entries, gaps


def build_budget(
    project_root: Path,
    generated_at: str | None = None,
    current_autodl_cost_cny: float | None = None,
) -> dict[str, Any]:
    generated_at = generated_at or utc_now()
    day8_path = project_root / "artifacts/runs/E-D8-001/cost.json"
    day8_cost = load_json(day8_path)
    projection = validate_day8_projection(day8_cost)
    ledger, gaps = historical_ledger(project_root)
    documented_subtotal = sum(
        float(row["cost_cny"])
        for row in ledger
        if row["included_in_documented_subtotal"]
    )

    day5_cost_path = project_root / "artifacts/runs/E-D5-001/cost.json"
    day5_cost = load_json(day5_cost_path)
    project_cap = float(day5_cost["guardrails"]["project_cost_cap_cny"])
    planning = projection["scenarios"][projection["planning_scenario"]]
    reservation = projection["scenarios"][projection["resource_reservation_scenario"]]
    max_current_total = project_cap - reservation["estimated_cost_cny"]
    unreconciled_allowance = max_current_total - documented_subtotal
    budget_pass = (
        current_autodl_cost_cny is not None
        and current_autodl_cost_cny <= max_current_total
    )
    projected_total = (
        current_autodl_cost_cny + reservation["estimated_cost_cny"]
        if current_autodl_cost_cny is not None
        else None
    )

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": generated_at,
        "experiment_id": EXPERIMENT_ID,
        "purpose": "day9_task1_formal_training_time_and_cost_budget",
        "artifact_status": "COMPLETE",
        "gate_status": "PASS" if budget_pass else "PENDING_BUDGET_RECONCILIATION",
        "decision": (
            "Budget gate passes; other Day 9 readiness gates still control Day 10 launch."
            if budget_pass
            else "Do not launch Day 10 until the current AutoDL cumulative charge is recorded and is no greater than the computed launch threshold."
        ),
        "source_evidence": {
            "day8_cost": relative_source(project_root, day8_path),
            "day5_budget": relative_source(project_root, day5_cost_path),
        },
        "training_contract": {
            "target_samples": TARGET_SAMPLES,
            "global_batch_size": GLOBAL_BATCH_SIZE,
            "target_optimizer_steps": TARGET_STEPS,
            "first_step_count": 1,
            "steady_step_count": TARGET_STEPS - 1,
        },
        "formula": "startup_seconds + first_step_seconds + 127 * steady_step_seconds + checkpoint_save_seconds",
        "projection_validation": {
            "status": "PASS",
            "all_source_values_recomputed": True,
            **projection,
        },
        "selected_budget": {
            "planning": {"scenario": "mean", **planning},
            "reservation": {"scenario": "conservative_max", **reservation},
        },
        "historical_cost_reconciliation": {
            "status": "PARTIAL",
            "currency": "CNY",
            "documented_cost_record_subtotal_cny": documented_subtotal,
            "subtotal_interpretation": "Sum of non-projection cost records found in the repository; not an invoice total or complete project spend.",
            "entries": ledger,
            "quality_findings": gaps,
        },
        "project_cap": {
            "hard_cap_cny": project_cap,
            "reserved_e_d10_cost_cny": reservation["estimated_cost_cny"],
            "maximum_current_platform_cumulative_charge_for_launch_cny": max_current_total,
            "maximum_unreconciled_cost_above_documented_subtotal_cny": unreconciled_allowance,
            "current_platform_cumulative_charge_cny": current_autodl_cost_cny,
            "projected_total_after_e_d10_reservation_cny": projected_total,
            "remaining_after_e_d10_reservation_cny": (
                project_cap - projected_total if projected_total is not None else None
            ),
            "status": "PASS" if budget_pass else "PENDING_PLATFORM_BILLING_INPUT",
            "input_provenance": (
                "User-reported AutoDL console cumulative charge in the current Codex session."
                if current_autodl_cost_cny is not None
                else None
            ),
        },
        "gates": [
            {
                "gate": "Day 8 projection identity and arithmetic",
                "status": "PASS",
                "evidence": "1024 samples / batch 8 = 128 steps; all three scenarios recomputed exactly.",
            },
            {
                "gate": "Frozen dual-GPU price",
                "status": "PASS",
                "evidence": "11.96 CNY per dual-GPU hour.",
            },
            {
                "gate": "E-D10-001 planning and reservation",
                "status": "PASS",
                "evidence": "Mean scenario is planning baseline; conservative maximum is reservation.",
            },
            {
                "gate": "Project cumulative spend reconciliation",
                "status": "PASS" if budget_pass else "PENDING",
                "evidence": (
                    f"User-reported AutoDL cumulative charge is {current_autodl_cost_cny:.2f} CNY; conservative projected total is {projected_total:.2f} CNY."
                    if budget_pass
                    else "Repository cost records are incomplete and exclude some billable windows."
                ),
            },
        ],
        "required_next_input": None if current_autodl_cost_cny is not None else {
            "name": "current_autodl_cumulative_charge_cny",
            "pass_condition": f"value <= {max_current_total:.6f}",
            "record_before": "E-D10-001 launch",
        },
        "report_design": {
            "audience": "technical",
            "delivery_mode": "portable_html",
            "chart": "Single-series bar comparison of the three discrete cost scenarios; no color grouping.",
            "table": "Exact historical cost-record ledger; projections excluded from the documented subtotal.",
            "omission": "No time-series chart because the evidence is a set of discrete experiment windows, not a continuous billing series.",
        },
    }
    return payload


def build_markdown(payload: dict[str, Any]) -> str:
    selected = payload["selected_budget"]
    reconciliation = payload["historical_cost_reconciliation"]
    cap = payload["project_cap"]
    scenario_rows = []
    labels = {"median": "中位稳态", "mean": "均值稳态", "conservative_max": "稳态最大值"}
    for name, values in payload["projection_validation"]["scenarios"].items():
        scenario_rows.append(
            f"| {labels[name]} | {values['steady_step_seconds']:.2f} | "
            f"{values['dual_gpu_hours']:.4f} | ¥{values['estimated_cost_cny']:.4f} |"
        )
    ledger_rows = []
    for row in reconciliation["entries"]:
        included = "是" if row["included_in_documented_subtotal"] else "否"
        ledger_rows.append(
            f"| {row['record_id']} | {row['record_type']} | ¥{row['cost_cny']:.4f} | {included} |"
        )
    findings = "\n".join(
        f"- **{item['severity']} / {item['scope']}**：{item['finding']} {item['impact']}"
        for item in reconciliation["quality_findings"]
    )
    billing_record = (
        f"用户报告的 AutoDL 控制台累计费用为 **¥{cap['current_platform_cumulative_charge_cny']:.2f}**；"
        f"加上保守预留后的预计累计费用为 **¥{cap['projected_total_after_e_d10_reservation_cny']:.2f}**，"
        f"剩余 **¥{cap['remaining_after_e_d10_reservation_cny']:.2f}**，因此预算 Gate 为 `PASS`。"
        if payload["gate_status"] == "PASS"
        else "尚未取得 AutoDL 控制台累计费用，因此预算 Gate 仍待对账。"
    )
    return f"""# Day 9 Task 1：E-D10-001 正式训练时间与费用预算

> 生成时间：{payload['generated_at_utc']}  
> 产物状态：**{payload['artifact_status']}**  
> Budget Gate：**{payload['gate_status']}**

## 技术摘要

Day 8 的 1024 条训练外推已完成独立复算：正式训练为 128 个 optimizer steps，规划基线采用均值场景 **{selected['planning']['dual_gpu_hours']:.2f} 双卡小时 / ¥{selected['planning']['estimated_cost_cny']:.2f}**，启动预留采用保守场景 **{selected['reservation']['dual_gpu_hours']:.2f} 双卡小时 / ¥{selected['reservation']['estimated_cost_cny']:.2f}**。外推计算与源文件逐项一致，projection Gate 为 PASS。

仓库内非预测成本记录的可见小计为 **¥{reconciliation['documented_cost_record_subtotal_cny']:.2f}**，但它遗漏 Day 4、Day 5 Smoke、Day 7，以及部分启动、空闲、失败尝试和计费舍入。因此该数字不是云平台累计账单。{billing_record}

## 训练范围与费用定义

- 样本：1024 条；global batch：8；optimizer steps：128。
- 首步单独计入预热，后续使用 127 个稳态 step。
- 总时间：启动 + 首步 + `127 × 稳态 step` + 最终 checkpoint 保存。
- 费用：双卡小时 × `11.96 CNY/双卡小时`。
- 规划值用于预期；保守值用于启动前资源与费用预留，不是运行时 SLA。

## 三场景复算全部通过

| 场景 | 稳态 step 秒 | 双卡小时 | 预计费用 |
|---|---:|---:|---:|
{chr(10).join(scenario_rows)}

规划冻结值：`{selected['planning']['dual_gpu_hours']:.6f}` 双卡小时、`¥{selected['planning']['estimated_cost_cny']:.6f}`。资源预留值：`{selected['reservation']['dual_gpu_hours']:.6f}` 双卡小时、`¥{selected['reservation']['estimated_cost_cny']:.6f}`。

## 历史成本记录只能形成不完整小计

| 成本记录 | 类型 | 金额 | 计入可见小计 |
|---|---|---:|:---:|
{chr(10).join(ledger_rows)}

Day 5 的 `cost.json` 是完整外评预算预测，不是已发生费用，因此不计入可见小计。其余行按文件中记录的观测窗口相加，但不能替代 AutoDL 账单。

## 数据质量发现与限制

{findings}

这些缺口不会改变 E-D10-001 自身的 ¥{selected['reservation']['estimated_cost_cny']:.2f} 预留，也不改变本次基于 AutoDL 平台控制值的预算判断；它们只意味着仓库小计仍不能冒充完整账单。

## 启动阈值与下一步

在为 E-D10-001 预留 ¥{selected['reservation']['estimated_cost_cny']:.2f} 后，Day 10 启动前 AutoDL 平台显示的当前累计费用必须满足：

```text
current_autodl_cumulative_charge_cny <= {cap['maximum_current_platform_cumulative_charge_for_launch_cny']:.6f}
```

当前仓库可见小计与该阈值之间还有 ¥{cap['maximum_unreconciled_cost_above_documented_subtotal_cny']:.2f}，但这不是可直接支配的余额，只表示需要由平台账单解释的最大未对账空间。

当前控制值已写入 Day 9 preflight。预算 Gate 只代表费用条件满足；Day 10 仍受配置、Git、磁盘等其他 readiness Gate 控制。

## 进一步问题

- 正式训练启动前，AutoDL 控制台累计费用是否仍接近本次记录的 ¥{cap['current_platform_cumulative_charge_cny'] or 0:.2f}？
- 是否存在仓库外的失败训练、空闲占用或已删除实验？
- 平台账单是否按整分钟、整小时或其他规则舍入？
"""


def build_artifact(payload: dict[str, Any]) -> dict[str, Any]:
    generated_at = payload["generated_at_utc"]
    projection = payload["projection_validation"]["scenarios"]
    labels = {"median": "中位稳态", "mean": "均值规划", "conservative_max": "保守预留"}
    projection_rows = [
        {
            "scenario_key": name,
            "scenario": labels[name],
            "sort_order": index,
            "steady_step_seconds": values["steady_step_seconds"],
            "total_seconds": values["total_seconds"],
            "dual_gpu_hours": values["dual_gpu_hours"],
            "estimated_cost_cny": values["estimated_cost_cny"],
            "target_samples": TARGET_SAMPLES,
            "optimizer_steps": TARGET_STEPS,
            "hourly_cost_cny": payload["projection_validation"]["cost_per_dual_gpu_hour_cny"],
        }
        for index, (name, values) in enumerate(projection.items(), start=1)
    ]
    ledger_rows = [
        {
            "sort_order": row["sort_order"],
            "record_id": row["record_id"],
            "record_type": row["record_type"],
            "cost_cny": row["cost_cny"],
            "included": "是" if row["included_in_documented_subtotal"] else "否",
            "coverage": row["coverage"],
        }
        for row in payload["historical_cost_reconciliation"]["entries"]
    ]
    planning = payload["selected_budget"]["planning"]
    reservation = payload["selected_budget"]["reservation"]
    subtotal = payload["historical_cost_reconciliation"]["documented_cost_record_subtotal_cny"]
    cap = payload["project_cap"]
    budget_gate_summary = (
        f"用户报告 AutoDL 累计费用为 ¥{cap['current_platform_cumulative_charge_cny']:.2f}；"
        f"加上 ¥{reservation['estimated_cost_cny']:.2f} 保守预留后为 "
        f"¥{cap['projected_total_after_e_d10_reservation_cny']:.2f}，预算 Gate 为 PASS。"
        if payload["gate_status"] == "PASS"
        else "仓库缺少完整平台账单，因此预算 Gate 仍待对账。"
    )
    headline = [
        {
            "planning_cost": f"¥{planning['estimated_cost_cny']:.2f}",
            "planning_hours": f"{planning['dual_gpu_hours']:.2f} h",
            "reservation_cost": f"¥{reservation['estimated_cost_cny']:.2f}",
            "reservation_hours": f"{reservation['dual_gpu_hours']:.2f} h",
            "documented_subtotal": f"¥{subtotal:.2f}",
            "ledger_status": "PARTIAL",
        }
    ]
    sources = [
        {
            "id": "day8_cost",
            "label": "Day 8 stability cost evidence",
            "path": payload["source_evidence"]["day8_cost"]["path"],
            "query": {
                "description": "Recompute the 1024-sample projection from fixed overheads and the three steady-step scenarios.",
                "language": "python",
                "tables_used": [payload["source_evidence"]["day8_cost"]["path"]],
                "filters": ["target_samples=1024", "global_batch_size=8"],
                "metric_definitions": [
                    "total_seconds = startup + first_step + 127 * steady_step + checkpoint_save",
                    "estimated_cost_cny = total_seconds / 3600 * 11.96",
                ],
            },
        },
        {
            "id": "budget_projection",
            "label": "Day 9 budget reconciliation output",
            "path": "artifacts/runs/E-D10-001/preflight/budget_projection.json",
            "query": {
                "description": "Classify repository cost files as projections or observed windows and sum only non-projection records.",
                "language": "python",
                "tables_used": [
                    "artifacts/runs/E-D5-001/cost.json",
                    "artifacts/runs/E-D6-001/base/cost.json",
                    "artifacts/runs/E-D8-001/cost.json",
                    "artifacts/runs/E-PAPER-BASEJUDGE-001/base/cost.json",
                    "artifacts/runs/E-PAPER-BASEJUDGE-001/smoke/base/cost.json",
                    "artifacts/runs/E-PAPER-BASEJUDGE-001/smoke_r2/base/cost.json",
                ],
                "metric_definitions": [
                    "documented subtotal = sum(cost records classified as observed); projections are excluded",
                    "launch threshold = 2000 CNY - conservative E-D10-001 reservation",
                ],
            },
        },
    ]
    manifest = {
        "version": 1,
        "surface": "report",
        "title": "Day 9 E-D10-001 Budget Gate",
        "description": "Auditable 1024-sample Vision-OPD training projection and project-budget reconciliation.",
        "generatedAt": generated_at,
        "sources": sources,
        "cards": [
            {
                "id": "planning_cost",
                "description": "Mean steady-step scenario selected as the formal planning baseline.",
                "dataset": "budget_headlines",
                "sourceId": "day8_cost",
                "metrics": [
                    {"label": "规划费用（CNY）", "field": "planning_cost"},
                    {"label": "双卡时间", "field": "planning_hours"},
                ],
            },
            {
                "id": "reservation_cost",
                "description": "Maximum observed steady-step scenario used for launch reservation.",
                "dataset": "budget_headlines",
                "sourceId": "day8_cost",
                "metrics": [
                    {"label": "保守预留（CNY）", "field": "reservation_cost"},
                    {"label": "双卡时间", "field": "reservation_hours"},
                ],
            },
            {
                "id": "documented_subtotal",
                "description": "Incomplete subtotal of non-projection cost records; not the platform bill.",
                "dataset": "budget_headlines",
                "sourceId": "budget_projection",
                "metrics": [
                    {"label": "仓库可见小计", "field": "documented_subtotal"},
                    {"label": "覆盖状态", "field": "ledger_status"},
                ],
            },
        ],
        "charts": [
            {
                "id": "projection_cost_chart",
                "title": "E-D10-001 三场景预计费用",
                "subtitle": "1024 条、128 steps；双卡单价 11.96 CNY/小时",
                "showDescription": True,
                "question": "How much dual-GPU cost should be planned and reserved for E-D10-001?",
                "rationale": "A bar chart makes the three discrete scenario magnitudes comparable without implying a time trend.",
                "type": "bar",
                "dataset": "projection_scenarios",
                "sourceId": "day8_cost",
                "encodings": {
                    "x": {"field": "scenario", "type": "nominal", "label": "场景"},
                    "y": {
                        "field": "estimated_cost_cny",
                        "type": "quantitative",
                        "format": "number",
                        "label": "预计费用",
                        "unit": "CNY",
                    },
                    "tooltip": [
                        {"field": "dual_gpu_hours", "type": "quantitative", "label": "双卡小时"},
                        {"field": "steady_step_seconds", "type": "quantitative", "label": "稳态 step 秒"},
                        {"field": "optimizer_steps", "type": "quantitative", "label": "optimizer steps"},
                    ],
                },
                "valueFormat": "number",
                "unit": "CNY",
                "layout": "full",
                "maxRows": 3,
            }
        ],
        "tables": [
            {
                "id": "historical_ledger",
                "title": "仓库成本记录清单",
                "subtitle": "精确列示预测与观测窗口；预测不计入可见小计",
                "showDescription": True,
                "dataset": "historical_cost_entries",
                "defaultSort": {"field": "sort_order", "direction": "asc"},
                "density": "dense",
                "sourceId": "budget_projection",
                "layout": "full",
                "columns": [
                    {"field": "record_id", "label": "成本记录", "type": "text"},
                    {"field": "record_type", "label": "类型", "type": "text"},
                    {"field": "cost_cny", "label": "金额（CNY）", "format": "number"},
                    {"field": "included", "label": "计入小计", "type": "text"},
                    {"field": "coverage", "label": "覆盖范围", "type": "text"},
                    {"field": "sort_order", "label": "顺序", "type": "number"},
                ],
            }
        ],
        "blocks": [
            {"id": "title", "type": "markdown", "body": "# Day 9 E-D10-001 Budget Gate"},
            {
                "id": "technical_summary",
                "type": "markdown",
                "body": (
                    "## 预算外推已通过，累计账单仍待核对\n\n"
                    f"1024 条正式训练需要 128 steps。规划基线为 **{planning['dual_gpu_hours']:.2f} 双卡小时 / ¥{planning['estimated_cost_cny']:.2f}**，"
                    f"保守预留为 **{reservation['dual_gpu_hours']:.2f} 双卡小时 / ¥{reservation['estimated_cost_cny']:.2f}**。"
                    "三种场景均由 Day 8 原始固定开销和稳态 step 重新计算，结果与源文件一致。\n\n"
                    f"{budget_gate_summary} 本报告产物完整；其他 readiness Gate 仍独立控制 Day 10 启动。"
                ),
            },
            {"id": "headline_metrics", "type": "metric-strip", "cardIds": ["planning_cost", "reservation_cost", "documented_subtotal"]},
            {
                "id": "projection_finding",
                "type": "markdown",
                "body": (
                    "## 保守预留比规划基线高约 70%\n\n"
                    "下图比较同一训练合同下的三个离散情景。均值场景用于正式规划，稳态最大值只作为资源预留；"
                    "差异来自 7 个 Day 8 稳态 step 的有限样本，不应解释为置信区间或 SLA。"
                ),
                "sourceId": "day8_cost",
            },
            {"id": "projection_chart", "type": "chart", "chartId": "projection_cost_chart", "layout": "full"},
            {
                "id": "scope_definitions",
                "type": "markdown",
                "body": (
                    "## 计算覆盖启动、首步、127 个稳态 step 和最终保存\n\n"
                    "测量对象是 E-D10-001 的 1024 条训练、global batch 8、128 个 optimizer steps。"
                    "总时间定义为 `startup + first_step + 127 × steady_step + checkpoint_save`；费用为双卡小时乘以 11.96 CNY。"
                ),
                "sourceId": "day8_cost",
            },
            {
                "id": "methodology",
                "type": "markdown",
                "body": (
                    "## 复算采用源值而非四舍五入后的报告数字\n\n"
                    "脚本读取 Day 8 `cost.json` 的完整精度字段，逐场景重算总秒数、小时和费用，并以严格数值容差核对。"
                    "Day 5 的外评成本文件被识别为 projection，明确排除在历史观测小计之外。"
                ),
            },
            {
                "id": "ledger_finding",
                "type": "markdown",
                "body": (
                    "## 历史成本记录只能形成不完整小计\n\n"
                    f"非预测成本记录的仓库可见小计为 **¥{subtotal:.2f}**。"
                    "由于 Day 4、Day 5 Smoke、Day 7 和若干非观测计费窗口缺失，这个小计不能替代 AutoDL 平台累计费用。"
                ),
                "sourceId": "budget_projection",
            },
            {"id": "ledger_table", "type": "table", "tableId": "historical_ledger", "layout": "full"},
            {
                "id": "limitations",
                "type": "markdown",
                "body": (
                    "## 仓库成本台账仍不完整，但平台控制值已补录\n\n"
                    "部分记录明确排除了模型服务启动、关闭、空闲、失败尝试或计费舍入；部分记录只有客户端观测时长和更新时间，"
                    f"无法证明精确服务窗口。报告不把仓库小计冒充账单；控制判断采用用户报告的 AutoDL 累计值 ¥{cap['current_platform_cumulative_charge_cny'] or 0:.2f}。"
                ),
            },
            {
                "id": "next_steps",
                "type": "markdown",
                "body": (
                    "## 预算 Gate 已关闭，继续检查其余 readiness Gate\n\n"
                    f"当前累计费用 **¥{cap['current_platform_cumulative_charge_cny'] or 0:.2f}** 低于启动阈值 "
                    f"**¥{cap['maximum_current_platform_cumulative_charge_for_launch_cny']:.2f}**。"
                    "正式启动前应刷新控制台数值；配置、Git 和磁盘条件仍需独立通过。"
                ),
            },
            {
                "id": "further_questions",
                "type": "markdown",
                "body": (
                    "## 仍需核对的问题\n\n"
                    "- 启动前累计费用是否仍低于已计算阈值？\n"
                    "- 是否存在仓库外的失败运行或空闲占用？\n"
                    "- 平台采用什么计费舍入规则？"
                ),
            },
        ],
    }
    return {
        "surface": "report",
        "manifest": manifest,
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "budget_headlines": headline,
                "projection_scenarios": projection_rows,
                "historical_cost_entries": ledger_rows,
            },
        },
        "sources": sources,
    }


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=default_root)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--generated-at")
    parser.add_argument("--current-autodl-cost-cny", type=float)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else project_root / "artifacts/runs/E-D10-001/preflight"
    )
    payload = build_budget(
        project_root,
        args.generated_at,
        args.current_autodl_cost_cny,
    )
    budget_path = output_dir / "budget_projection.json"
    markdown_path = output_dir / "budget_evidence.md"
    artifact_path = output_dir / "budget_report_artifact.json"
    billing_input_path = output_dir / "autodl_billing_input.json"
    write_json_atomic(budget_path, payload)
    write_text_atomic(markdown_path, build_markdown(payload))
    write_json_atomic(artifact_path, build_artifact(payload))
    if args.current_autodl_cost_cny is not None:
        write_json_atomic(
            billing_input_path,
            {
                "schema_version": 1,
                "recorded_at_utc": payload["generated_at_utc"],
                "current_autodl_cumulative_charge_cny": args.current_autodl_cost_cny,
                "currency": "CNY",
                "provenance": "User-reported AutoDL console value in the current Codex session.",
            },
        )
    print(f"ARTIFACT_STATUS={payload['artifact_status']}")
    print(f"GATE_STATUS={payload['gate_status']}")
    print(f"BUDGET_PROJECTION={budget_path}")
    print(f"BUDGET_EVIDENCE={markdown_path}")
    print(f"REPORT_ARTIFACT={artifact_path}")
    if args.current_autodl_cost_cny is not None:
        print(f"BILLING_INPUT={billing_input_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
