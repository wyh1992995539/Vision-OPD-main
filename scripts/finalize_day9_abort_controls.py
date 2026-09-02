#!/usr/bin/env python3
"""Generate the auditable Day 9 Task 5 abort-control report."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.monitor_vopd_training import load_policy, replay, utc_now, write_json
from scripts.run_vopd_guarded import EXPECTED_CONFIG_SHA256, static_preflight


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(project_root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=project_root, check=True, capture_output=True, text=True
    ).stdout.strip()


def run_tests(project_root: Path) -> dict[str, Any]:
    command = [sys.executable, "-m", "unittest", "tests.test_vopd_abort_guard", "-v"]
    result = subprocess.run(command, cwd=project_root, capture_output=True, text=True, check=False)
    output = result.stdout + result.stderr
    count = 0
    for line in output.splitlines():
        if line.startswith("Ran ") and " test" in line:
            count = int(line.split()[1])
    return {
        "status": "PASS" if result.returncode == 0 else "FAIL",
        "return_code": result.returncode,
        "test_count": count,
        "command": " ".join(command),
        "output": output,
    }


def direct_run_guard(project_root: Path) -> dict[str, Any]:
    env = {key: value for key, value in __import__("os").environ.items() if key != "VOPD_GUARD_ACTIVE"}
    result = subprocess.run(
        ["bash", "scripts/run_vopd_2gpu.sh", "--config", "configs/vopd_1024.yaml", "--run"],
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "status": "PASS" if result.returncode == 2 and "Direct --run is blocked" in result.stderr else "FAIL",
        "return_code": result.returncode,
        "stderr": result.stderr.strip(),
        "gpu_training_started": False,
    }


def build_report(
    project_root: Path, generated_at: str | None = None
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config_path = project_root / "configs/vopd_1024.yaml"
    policy_path = project_root / "configs/vopd_abort_policy.yaml"
    timestamp = generated_at or utc_now()
    policy = load_policy(policy_path)
    static = static_preflight(project_root, config_path, policy_path)
    tests = run_tests(project_root)
    replay_result = replay(
        project_root / "artifacts/runs/E-D8-001/metrics.jsonl",
        project_root / "artifacts/runs/E-D8-001/logs/train.log",
        policy,
    )
    guard = direct_run_guard(project_root)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    clean = not git(project_root, "status", "--porcelain=v1")
    replay_rules = {item["rule"] for item in replay_result["fatal_log_issues"]}
    gates = {
        "task4_and_static_contract": static["status"] == "PASS",
        "formal_config_hash_unchanged": static["config_sha256"] == EXPECTED_CONFIG_SHA256,
        "dataloader_workers_zero": config["data"]["dataloader_num_workers"] == 0,
        "policy_schema_and_formulas": policy["schema_version"] == 1,
        "unit_and_replay_tests": tests["status"] == "PASS" and tests["test_count"] >= 14,
        "day8_metric_contract_clean": replay_result["metric_rows"] == 8 and not replay_result["metric_issues"],
        "day8_killed_event_detected": "dataloader_worker_killed" in replay_rules,
        "direct_run_blocked": guard["status"] == "PASS",
        "cpu_only_task5": static["gpu_used"] is False and guard["gpu_training_started"] is False,
        "source_worktree_clean_before_report": clean,
    }
    failed = [name for name, passed in gates.items() if not passed]
    if failed:
        raise ValueError(f"Task 5 gates failed: {', '.join(failed)}")
    policy_artifact = {
        "schema_version": 1,
        "generated_at_utc": timestamp,
        "experiment_id": "E-D10-001",
        "status": "FROZEN",
        "policy_path": "configs/vopd_abort_policy.yaml",
        "policy_sha256": sha256_file(policy_path),
        "formal_config_sha256": static["config_sha256"],
        "policy": policy,
    }
    test_artifact = {
        "schema_version": 1,
        "generated_at_utc": timestamp,
        "status": "PASS",
        "gpu_used": False,
        "unit_tests": tests,
        "day8_replay": replay_result,
        "direct_run_guard": guard,
        "static_preflight": static,
    }
    payload = {
        "schema_version": 1,
        "generated_at_utc": timestamp,
        "experiment_id": "E-D10-001",
        "purpose": "day9_task5_executable_abort_controls",
        "artifact_status": "COMPLETE",
        "report_status": "PASS_TO_DAY10",
        "task5_completed": True,
        "advance_to_day10": True,
        "gpu_used": False,
        "gates": gates,
        "formal_config_sha256": static["config_sha256"],
        "abort_policy_sha256": policy_artifact["policy_sha256"],
        "audited_source_commit": git(project_root, "rev-parse", "HEAD"),
        "audited_source_worktree_clean": clean,
        "controls": {
            "sample_interval_seconds": policy["telemetry"]["sample_interval_seconds"],
            "max_wall_time_hours": policy["runtime"]["max_wall_time_hours"],
            "gpu_memory_ratio": policy["memory"]["gpu_used_ratio_abort"],
            "cgroup_memory_ratio": policy["memory"]["cgroup_used_ratio_abort"],
            "prelaunch_storage_bytes": policy["disk"]["prelaunch_required_bytes"],
            "runtime_storage_soft_floor_bytes": policy["disk"]["runtime_soft_floor_bytes"],
            "runtime_storage_hard_floor_bytes": policy["disk"]["runtime_hard_floor_bytes"],
            "expected_checkpoint_step": policy["checkpoint"]["expected_final_step"],
            "required_checkpoint_files": len(policy["checkpoint"]["required_relative_files"]),
        },
        "day8_replay": {
            "metric_rows": replay_result["metric_rows"],
            "metric_issues": replay_result["metric_issues"],
            "fatal_log_rules": sorted(replay_rules),
        },
        "launch_requirements": [
            "Refresh the AutoDL cumulative charge within 15 minutes of launch.",
            "Keep the Git worktree clean.",
            "Recheck output collision, storage, two GPUs, and cgroup v2 at launch.",
            "Use scripts/run_vopd_guarded.py; never call the raw --run entry directly.",
        ],
        "commands": {
            "cpu_only_preflight": "python scripts/run_vopd_guarded.py --preflight-only",
            "formal_training": (
                "python scripts/run_vopd_guarded.py --current-autodl-cost-cny <LATEST> "
                "--billing-observed-at-utc <ISO-8601> --run"
            ),
        },
        "sources": {
            "policy": {"path": "configs/vopd_abort_policy.yaml", "sha256": sha256_file(policy_path)},
            "monitor": {
                "path": "scripts/monitor_vopd_training.py",
                "sha256": sha256_file(project_root / "scripts/monitor_vopd_training.py"),
            },
            "guarded_launcher": {
                "path": "scripts/run_vopd_guarded.py",
                "sha256": sha256_file(project_root / "scripts/run_vopd_guarded.py"),
            },
            "runbook": {
                "path": "docs/day9_task5_abort_runbook.md",
                "sha256": sha256_file(project_root / "docs/day9_task5_abort_runbook.md"),
            },
        },
    }
    return payload, policy_artifact, test_artifact


def build_markdown(payload: dict[str, Any]) -> str:
    gates = "\n".join(f"| {name} | {'PASS' if passed else 'FAIL'} |" for name, passed in payload["gates"].items())
    controls = payload["controls"]
    return f"""# E-D10-001 Task 5 已完成：允许进入 Day 10 动态启动检查

> 生成时间：{payload['generated_at_utc']}
> 决策：**{payload['report_status']}**
> GPU 使用：**false**

## 结论

Task 5 的可执行中止策略、逐卡 GPU/进程树 RSS/cgroup/磁盘观测、训练指标解析、进程组终止、
最终 checkpoint 后置校验和防绕过入口均已落地。任务五完成，可以进入 Day10；这不等于可以跳过
启动时动态 Gate，AutoDL 累计费用必须在启动前 15 分钟内刷新。

正式训练配置 SHA256 仍为 `{payload['formal_config_sha256']}`，没有因 Task 5 改动训练数学合同。
中止策略 SHA256 为 `{payload['abort_policy_sha256']}`。

## Gate

| Gate | 状态 |
|---|---|
{gates}

## 冻结控制

| 控制 | 值 |
|---|---:|
| 观测周期 | {controls['sample_interval_seconds']} 秒 |
| 最长墙钟时间 | {controls['max_wall_time_hours']} 小时 |
| GPU 显存中止比例 | {controls['gpu_memory_ratio']:.0%}，连续 3 次 |
| cgroup 内存中止比例 | {controls['cgroup_memory_ratio']:.0%}，连续 3 次 |
| 启动磁盘要求 | {controls['prelaunch_storage_bytes']} bytes |
| 运行期磁盘软下限 | {controls['runtime_storage_soft_floor_bytes']} bytes |
| 运行期磁盘硬下限 | {controls['runtime_storage_hard_floor_bytes']} bytes |
| 最终 checkpoint | step {controls['expected_checkpoint_step']}，{controls['required_checkpoint_files']} 个必需文件 |

NaN/Inf、Teacher 直接梯度、Teacher optimizer 改变、cgroup OOM、checkpoint 保存错误和磁盘硬下限
属于立即中止条件；EMA/Student 不更新、连续生成错误、内存压力、磁盘软下限和日志心跳使用冻结的
连续阈值。中止先发送 `SIGTERM`，60 秒后必要时升级为 `SIGKILL`。

## Day8 回放

- 结构化指标：{payload['day8_replay']['metric_rows']} 步，合同异常 {len(payload['day8_replay']['metric_issues'])} 项。
- 日志命中：{', '.join(payload['day8_replay']['fatal_log_rules'])}。
- 这证明守护器能保留 Day8 的真实 caveat，而不是将其改写为已解决。

## Day10 启动前仍需执行

1. 在 AutoDL 控制台读取最新累计费用，并记录带时区的 ISO-8601 时间。
2. 保持 Git 工作区 clean；重新检查输出冲突、磁盘、两张 GPU 和 cgroup v2。
3. 仅使用以下入口：

```bash
{payload['commands']['formal_training']}
```

直接执行 `bash scripts/run_vopd_2gpu.sh --run` 已被拒绝。完整操作见
`docs/day9_task5_abort_runbook.md`。
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--generated-at")
    args = parser.parse_args()
    root = args.project_root.resolve()
    payload, policy, tests = build_report(root, args.generated_at)
    preflight = root / "artifacts/runs/E-D10-001/preflight"
    write_json(preflight / "task5_abort_policy.json", policy)
    write_json(preflight / "task5_test_report.json", tests)
    write_json(preflight / "task5_abort_controls.json", payload)
    (root / "artifacts/runs/E-D10-001/preflight.md").write_text(build_markdown(payload), encoding="utf-8")
    print(json.dumps({"status": payload["report_status"], "gpu_used": False}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

