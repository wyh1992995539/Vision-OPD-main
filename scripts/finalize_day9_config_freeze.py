#!/usr/bin/env python3
"""Freeze and document the E-D10-001 formal Vision-OPD configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from scripts.day9_formal_training_readiness import (
    EXPERIMENT_ID,
    FORMAL_CONFIG_EXPECTED,
    audit_config,
    nested_get,
    sha256_file,
    utc_now,
    write_json,
    write_text,
)


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    return value


def build_freeze(
    project_root: Path,
    config_path: Path,
    baseline_path: Path,
    generated_at: str | None = None,
) -> dict[str, Any]:
    config = load_yaml(config_path)
    baseline = load_yaml(baseline_path)
    audit = audit_config(project_root, config_path)
    actual = {key: nested_get(config, key) for key in FORMAL_CONFIG_EXPECTED}
    changes = [
        {
            "field": key,
            "day8_value": nested_get(baseline, key),
            "formal_value": value,
            "changed": nested_get(baseline, key) != value,
        }
        for key, value in actual.items()
    ]
    contract = {
        "expected_samples": actual["training.expected_samples"],
        "global_batch_size": actual["data.train_batch_size"],
        "optimizer_steps": actual["training.total_optimizer_steps"],
        "total_epochs": actual["training.total_epochs"],
        "require_full_epoch": actual["training.require_full_epoch"],
    }
    invariants = {
        "formal_config_gate": audit["status"] == "PASS",
        "sample_budget_matches_steps": (
            contract["expected_samples"]
            == contract["global_batch_size"] * contract["optimizer_steps"]
        ),
        "legacy_smoke_removed": "smoke" not in config,
        "fresh_base_start": actual["training.resume_mode"] == "disable",
        "final_checkpoint_only": (
            actual["training.save_frequency"] == -1
            and actual["training.max_actor_ckpt_to_keep"] == 1
        ),
        "dataloader_child_processes_disabled": actual["data.dataloader_num_workers"] == 0,
    }
    status = "PASS" if all(invariants.values()) else "FAIL"
    return {
        "schema_version": 1,
        "generated_at_utc": generated_at or utc_now(),
        "experiment_id": EXPERIMENT_ID,
        "purpose": "day9_task3_formal_config_freeze",
        "artifact_status": "COMPLETE",
        "config_gate_status": status,
        "task3_completed": status == "PASS",
        "advance_to_task4": status == "PASS",
        "advance_to_day10": False,
        "day10_block_reason": "Storage and Git gates remain independently controlling.",
        "sources": {
            "formal_config": {
                "path": str(config_path.relative_to(project_root)),
                "sha256": sha256_file(config_path),
            },
            "day8_baseline": {
                "path": str(baseline_path.relative_to(project_root)),
                "sha256": sha256_file(baseline_path),
            },
            "task2_data_manifest": "artifacts/runs/E-D10-001/preflight/data_manifest.json",
            "task2_base_manifest": "artifacts/runs/E-D10-001/preflight/base_model_manifest.json",
        },
        "formal_contract": contract,
        "frozen_parameters": actual,
        "invariants": invariants,
        "day8_comparison": changes,
        "intentional_changes": [
            {
                "field": "data.dataloader_num_workers",
                "from": nested_get(baseline, "data.dataloader_num_workers"),
                "to": actual["data.dataloader_num_workers"],
                "reason": "Remove the DataLoader child-process failure mode observed after the Day 8 checkpoint save.",
                "model_math_effect": "None; loading runs in the trainer process and throughput may be lower.",
            },
            {
                "field": "data.shuffle",
                "from": nested_get(baseline, "data.shuffle"),
                "to": actual["data.shuffle"],
                "reason": "The formal full train split uses seeded shuffling; Day 8 used an already ordered audit subset.",
                "determinism_control": "experiment.seed=42 and actor data_loader_seed=42",
            },
        ],
        "commands": {
            "preflight_only": "bash scripts/run_vopd_2gpu.sh --config configs/vopd_1024.yaml --preflight-only",
            "formal_training_after_all_gates_pass": "bash scripts/run_vopd_2gpu.sh --config configs/vopd_1024.yaml --run",
        },
    }


def build_markdown(payload: dict[str, Any]) -> str:
    rows = "\n".join(
        f"| `{key}` | `{value}` |"
        for key, value in payload["frozen_parameters"].items()
    )
    changed = "\n".join(
        f"| `{item['field']}` | `{item['day8_value']}` | `{item['formal_value']}` |"
        for item in payload["day8_comparison"]
        if item["changed"]
    )
    invariants = "\n".join(
        f"- `{key}`：{'PASS' if value else 'FAIL'}"
        for key, value in payload["invariants"].items()
    )
    contract = payload["formal_contract"]
    return f"""# Day 9 Task 3：E-D10-001 正式配置冻结

> 产物状态：**{payload['artifact_status']}**  
> Config Gate：**{payload['config_gate_status']}**  
> 配置 SHA256：`{payload['sources']['formal_config']['sha256']}`

## 结论

`E-D10-001` 正式配置已冻结：1024 条、global batch 8、128 optimizer steps、完整 1 epoch，从冻结 Base 冷启动。任务 3 已完成，但磁盘与 Git Gate 仍独立阻止 Day 10。

## 正式训练合同

- 样本：{contract['expected_samples']}
- global batch：{contract['global_batch_size']}
- optimizer steps：{contract['optimizer_steps']}
- epoch：{contract['total_epochs']}
- 要求完整 epoch：{contract['require_full_epoch']}

## 冻结参数

| 字段 | 冻结值 |
|---|---|
{rows}

## 相对 Day 8 的变化

| 字段 | Day 8 | E-D10-001 |
|---|---|---|
{changed}

`dataloader_num_workers` 从 4 降到 0，用于移除 Day 8 checkpoint 保存后出现的 DataLoader 子进程 `Killed` 风险。该项不改变模型计算公式，但可能降低数据加载吞吐。正式全量数据恢复 seeded shuffle；Day 8 的 64 条审计子集因自身已有冻结顺序而关闭 shuffle。

## 不变量 Gate

{invariants}

## 可执行命令

仅 CPU preflight：

```bash
{payload['commands']['preflight_only']}
```

以下训练命令已经冻结，但只有 Day 9 全部 Gate 通过后才允许执行：

```bash
{payload['commands']['formal_training_after_all_gates_pass']}
```
"""


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--generated-at")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    config_path = (args.config or project_root / "configs/vopd_1024.yaml").resolve()
    baseline_path = (args.baseline or project_root / "artifacts/runs/E-D8-001/config.yaml").resolve()
    output_dir = (args.output_dir or project_root / "artifacts/runs/E-D10-001/preflight").resolve()
    payload = build_freeze(project_root, config_path, baseline_path, args.generated_at)
    write_json(output_dir / "task3_config_freeze.json", payload)
    write_text(output_dir / "task3_config_freeze.md", build_markdown(payload))
    print(f"ARTIFACT_STATUS={payload['artifact_status']}")
    print(f"CONFIG_GATE_STATUS={payload['config_gate_status']}")
    print(f"TASK3_COMPLETED={str(payload['task3_completed']).lower()}")
    return 0 if payload["task3_completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
