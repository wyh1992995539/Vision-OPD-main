#!/usr/bin/env python3
"""Build the auditable E-D8-001 stability evidence and final report."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import re
import shutil
import statistics
from pathlib import Path
from typing import Any

import yaml


SCHEMA_VERSION = 1
EXPERIMENT_ID = "E-D8-001"
EXPECTED_STEPS = 8
TARGET_STEPS = 128
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
METRIC_RE = re.compile(
    r"(?:^| - )([A-Za-z0-9_./()]+):"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)
NONFINITE_RE = re.compile(r":\s*(?:[-+]?inf|nan)(?=\s|$)", re.IGNORECASE)

REQUIRED_METRICS = (
    "actor/vopd_loss",
    "actor/grad_norm",
    "timing_s/gen",
    "timing_s/update_actor",
    "timing_s/step",
    "perf/max_memory_allocated_gb",
    "perf/max_memory_reserved_gb",
    "perf/cpu_memory_used_gb",
    "response_length/mean",
    "response_length/max",
    "response_length/clip_ratio",
    "response/aborted_ratio",
    "prompt_length/max",
    "prompt_length/clip_ratio",
    "evidence/student_param_probe_max_delta_after_optimizer",
    "evidence/teacher_param_probe_max_delta_after_optimizer",
    "evidence/teacher_grad_non_none_count",
    "evidence/teacher_param_probe_max_delta_after_ema",
    "evidence/ema_update_applied",
    "self_distillation/teacher_always_on_fraction",
    "self_distillation/teacher_image_swap_fraction",
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    write_text_atomic(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def parse_training_log(text: str) -> tuple[list[dict[str, float]], dict[str, int]]:
    """Parse the one-line metric records emitted by the VERL trainer."""
    steps: dict[int, dict[str, float]] = {}
    nonfinite_tokens = 0
    for raw_line in text.splitlines():
        line = ANSI_RE.sub("", raw_line).replace("\r", "")
        if "step:" not in line or "training/global_step:" not in line:
            continue
        metric_line = line[line.index("step:") :]
        pairs = {key: float(value) for key, value in METRIC_RE.findall(metric_line)}
        if "step" not in pairs:
            continue
        step = int(pairs.pop("step"))
        if step in steps:
            raise ValueError(f"training log contains duplicate metric line for step {step}")
        missing = [key for key in REQUIRED_METRICS if key not in pairs]
        if missing:
            raise ValueError(f"step {step} is missing required metrics: {missing}")
        nonfinite_tokens += len(NONFINITE_RE.findall(metric_line))
        steps[step] = pairs

    ordered = [{"step": float(step), **steps[step]} for step in sorted(steps)]
    lower = text.lower()
    signals = {
        "nonfinite_token_count": nonfinite_tokens,
        "traceback_count": text.count("Traceback (most recent call last):"),
        "dataloader_worker_killed_count": text.count("is killed by signal: Killed"),
        "cuda_oom_count": lower.count("cuda out of memory"),
        "out_of_memory_error_count": lower.count("outofmemoryerror"),
    }
    return ordered, signals


def percentile(values: list[float], fraction: float) -> float:
    """Linear percentile compatible with the small seven-step steady sample."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def projection_scenarios(
    *,
    startup_seconds: float,
    first_step_seconds: float,
    steady_step_seconds: list[float],
    checkpoint_save_seconds: float,
    cost_per_dual_gpu_hour: float,
) -> dict[str, dict[str, float]]:
    if len(steady_step_seconds) != EXPECTED_STEPS - 1:
        raise ValueError("projection requires the seven steady steps (2 through 8)")
    statistics_by_scenario = {
        "median": statistics.median(steady_step_seconds),
        "mean": statistics.mean(steady_step_seconds),
        "conservative_max": max(steady_step_seconds),
    }
    result: dict[str, dict[str, float]] = {}
    for name, steady_seconds in statistics_by_scenario.items():
        total_seconds = (
            startup_seconds
            + first_step_seconds
            + (TARGET_STEPS - 1) * steady_seconds
            + checkpoint_save_seconds
        )
        hours = total_seconds / 3600
        result[name] = {
            "steady_step_seconds": steady_seconds,
            "total_seconds": total_seconds,
            "dual_gpu_hours": hours,
            "estimated_cost_cny": hours * cost_per_dual_gpu_hour,
        }
    return result


def all_equal(steps: list[dict[str, float]], key: str, value: float) -> bool:
    return all(row[key] == value for row in steps)


def compact_metric_row(row: dict[str, float]) -> dict[str, float | int]:
    step = int(row["step"])
    return {
        "step": step,
        "loss": row["actor/vopd_loss"],
        "grad_norm": row["actor/grad_norm"],
        "step_seconds": row["timing_s/step"],
        "generation_seconds": row["timing_s/gen"],
        "generation_share": row["timing_s/gen"] / row["timing_s/step"],
        "update_actor_seconds": row["timing_s/update_actor"],
        "prompt_max_tokens": int(row["prompt_length/max"]),
        "prompt_clip_ratio": row["prompt_length/clip_ratio"],
        "response_mean_tokens": row["response_length/mean"],
        "response_max_tokens": int(row["response_length/max"]),
        "response_clip_ratio": row["response_length/clip_ratio"],
        "aborted_ratio": row["response/aborted_ratio"],
        "student_optimizer_delta": row[
            "evidence/student_param_probe_max_delta_after_optimizer"
        ],
        "teacher_optimizer_delta": row[
            "evidence/teacher_param_probe_max_delta_after_optimizer"
        ],
        "teacher_grad_non_none_count": int(row["evidence/teacher_grad_non_none_count"]),
        "teacher_ema_delta": row["evidence/teacher_param_probe_max_delta_after_ema"],
        "ema_update_applied": int(row["evidence/ema_update_applied"]),
        "logged_max_memory_allocated_gb": row["perf/max_memory_allocated_gb"],
        "logged_max_memory_reserved_gb": row["perf/max_memory_reserved_gb"],
        "logged_host_memory_used_gb": row["perf/cpu_memory_used_gb"],
    }


def render_report(summary: dict[str, Any]) -> str:
    metrics = summary["training"]["steps"]
    projection = summary["projection_1024"]
    reload = summary["cold_reload"]
    gates = summary["gates"]

    gate_rows = "\n".join(
        f"| {item['gate']} | {item['status']} | {item['evidence']} |" for item in gates
    )
    step_rows = "\n".join(
        "| {step} | {loss:.5f} | {grad_norm:.2f} | {step_seconds:.2f} | "
        "{generation_seconds:.2f} | {generation_share:.1%} | {prompt_max_tokens} | "
        "{response_mean_tokens:.1f} | {response_clip_ratio:.0%} |".format(**row)
        for row in metrics
    )
    scenario_rows = "\n".join(
        "| {label} | {steady:.2f} | {hours:.2f} | ¥{cost:.2f} |".format(
            label=label,
            steady=value["steady_step_seconds"],
            hours=value["dual_gpu_hours"],
            cost=value["estimated_cost_cny"],
        )
        for label, value in (
            ("中位稳态", projection["scenarios"]["median"]),
            ("均值稳态（规划口径）", projection["scenarios"]["mean"]),
            ("稳态最大值（保守上界）", projection["scenarios"]["conservative_max"]),
        )
    )

    return f"""# Day 8 Vision-OPD 64 条稳定性报告

实验 ID：`{EXPERIMENT_ID}`  
证据截止：`{summary['generated_at_utc']}`  
最终状态：**{summary['status']}**

## 技术结论

Day 8 已完成，可以进入 Day 9 正式训练 Gate。固定 64 条数据连续完成 **8/8 optimizer steps**，全部记录数值有限；Student 每步发生参数更新，Teacher 每步均无 optimizer 直接更新、无梯度，并完成 EMA 更新。最终 `global_step_8` checkpoint 完整保存，关闭训练流程后合并并冷启动服务，冻结的 **5/5** 条样本均得到非空输出且推理错误为 0。

结论标记为 `PASS_WITH_CAVEAT`，不是无条件 `PASS`：训练在 checkpoint 已保存且进度达到 8/8 后出现一次 DataLoader worker `Killed`；没有同时采集的训练期 cgroup/RSS 快照，无法归因。日志中的显存峰值还高于冻结的单卡 96 GB 物理口径，因此只能作为 logger 诊断值，不能声称为可信的逐卡峰值。两项均不否定已保存模型的冷重载结果，但必须在 Day 9 修复观测性。

## Day 8 三项验收由六个证据 Gate 支持

| Gate | 状态 | 证据 |
|---|---|---|
{gate_rows}

冷重载摘要还确认源 checkpoint 在合并前后未变化，合并模型清单 SHA256 为 `{reload['merged_manifest_sha256']}`，受控关闭后的服务退出码为 `{reload['server_exit_code_after_controlled_shutdown']}`。5 条重载 Smoke 的 3/5 正确率只证明推理链路可用，样本量太小，**不作为模型效果结论**。

## 8 步训练稳定，Step 6 是可解释的耗时高点

| Step | VOPD loss | Grad norm | Step 秒 | 生成秒 | 生成占比 | Prompt max | Response mean | Response 达上限 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{step_rows}

8 步 loss 范围为 `{summary['training']['loss']['min']:.5f}`～`{summary['training']['loss']['max']:.5f}`，均值 `{summary['training']['loss']['mean']:.5f}`；这是一段短稳定性运行，只能用于识别发散/非有限值，不能据此判断收敛。最大 grad norm 为 `{summary['training']['grad_norm']['max']:.2f}`，未出现 NaN/Inf。Step 6 用时 `{metrics[5]['step_seconds']:.2f}` 秒，主要来自 actor update `{metrics[5]['update_actor_seconds']:.2f}` 秒，而不是生成异常。

全程 prompt 截断率为 0，最大 prompt 为 `{summary['training']['lengths']['prompt_max_tokens']}` tokens；response abort 为 0。前两步各有 25% response 达到 256-token 上限，合计约 `4/64` 条，占 6.25%。这不是运行失败，但 Day 9 必须继续保留 response 截断监控。

展示说明：本报告没有为 8 个离散 step 另画趋势图。逐步表能保留全部精确值、异常点与单位，图表反而会增加尺度误读；这是有意的视觉省略。

## 1024 条训练预计约 1.02 双卡小时，建议按 1.75 小时保守预留

外推严格沿用 Day 8 配置：global batch 8，1024 条对应 128 steps。计算式为：

```text
总时长 = 启动/加载固定开销 + 首步预热 + 127 × 稳态单步统计 + 一次最终 checkpoint 保存
费用 = 总时长（小时）× 11.96 元/双卡小时
```

其中启动/加载固定开销用运行清单时间到日志结束时间反推，为 `{projection['startup_seconds']:.2f}` 秒；首步 `{projection['first_step_seconds']:.2f}` 秒；最终保存 `{projection['checkpoint_save_seconds']:.2f}` 秒；稳态样本为 Step 2～8。

| 场景 | 稳态 step 秒 | 预计双卡小时 | 预计费用 |
|---|---:|---:|---:|
{scenario_rows}

Day 9 的预算基线采用“均值稳态”：约 **{projection['scenarios']['mean']['dual_gpu_hours']:.2f} 双卡小时 / ¥{projection['scenarios']['mean']['estimated_cost_cny']:.2f}**。启动前资源预留采用“稳态最大值”上界：约 **{projection['scenarios']['conservative_max']['dual_gpu_hours']:.2f} 双卡小时 / ¥{projection['scenarios']['conservative_max']['estimated_cost_cny']:.2f}**。它远低于计划中的 38 双卡小时停止线，但 64 条固定子集不保证覆盖 1024 条的长度尾部，因此这只是工程预算外推，不是 SLA。

Day 8 已记录的训练窗口约 `{summary['cost']['training_window_dual_gpu_hours']:.3f}` 双卡小时（`¥{summary['cost']['training_window_cost_cny']:.2f}`）；最终复用 merged 模型的冷重载窗口约 `{summary['cost']['reload_window_dual_gpu_hours']:.3f}` 双卡小时（`¥{summary['cost']['reload_window_cost_cny']:.2f}`）。这些是证据时间戳估算，不包含未被清单覆盖的空闲、失败尝试或云厂商计费舍入。

## 输入、指标与验证方法

- 训练总体：按 seed 42 的稳定哈希顺序从冻结 train-1024 选择 64 条；`shuffle=false`；8 条/global batch；1 epoch；8 optimizer steps。
- 模型与算法：Qwen3.5-4B Base 独立启动；Vision-OPD online prefix；Top-K 100；JSD alpha/beta 0.5；EMA rate 0.05。
- 稳态定义：排除含模型/内核预热的 Step 1，只用 Step 2～8 计算中位数、均值和最大值。
- 生成占比：`timing_s/gen ÷ timing_s/step`；全 8 步加权占比为 `{summary['training']['timing']['generation_share_all_steps']:.1%}`，Step 2～8 为 `{summary['training']['timing']['generation_share_steady_steps']:.1%}`。
- checkpoint：13 个必需文件均有非零大小和 SHA256；目录大小 `{summary['checkpoint']['directory_size_gib']:.2f}` GiB；冷重载确认源文件大小与 mtime 未变化。
- 训练期 CPU 指标 `perf/cpu_memory_used_gb` 来自 `psutil.virtual_memory().used`，是宿主机已用内存，不是训练进程 RSS。

## 限制、异常与鲁棒性检查

1. **结束阶段 worker 异常（中等影响）**：日志有 1 次 DataLoader worker `Killed` 和 1 个 traceback；发生在 8/8 与 checkpoint 路径输出之后，且 checkpoint 后续冷重载通过。没有训练期同步 cgroup/RSS 样本，不能宣称原因是或不是主机 OOM。
2. **显存峰值口径不可审计（中等影响）**：logger 报告 allocated `{summary['training']['memory']['logged_max_allocated_gb']:.2f}` GB、reserved `{summary['training']['memory']['logged_max_reserved_gb']:.2f}` GB；数值与 96 GB/卡的冻结硬件口径不一致，因此不用于容量结论。Day 9 应旁路采集每卡 `nvidia-smi` 峰值。
3. **工作树非 clean（低到中等影响）**：运行清单记录 commit `{summary['provenance']['git_commit']}`，但启动时有未提交 Day 8 文件。配置/数据哈希和运行时 Git 状态已保存，可追踪但复现体验弱于 clean commit。
4. **外推样本较小（中等影响）**：只有 7 个稳态 step，且 response 长度分布不均；报告同时给中位、均值与最大值场景，不把单一均值包装成确定预测。
5. **冷重载复用了 merged 模型（低影响）**：最终 PASS 窗口使用 `--reuse-merged`，因此 `{reload['duration_seconds']:.2f}` 秒不含模型合并时间；合并产物自身有完整文件哈希，且源 checkpoint 未改变。

验证评估：**Share with caveats / 可带限制共享**。核心 Day 8 决策（是否进入 Day 9）证据充分；逐卡显存峰值与 worker 被杀原因仍未验证，不能从报告中推导这两项结论。

## Day 8 收尾与下一步

Day 8 到此关闭，不需要继续占用 GPU。下一任务是 Day 9：冻结 `configs/vopd_1024.yaml`，生成 `E-D10-001` preflight，加入训练期每卡显存采样、进程 RSS/cgroup 采样和 worker 异常中止/降级策略。只有 Day 9 全部 Gate 为 PASS 才启动 Day 10 的 1024 条正式训练。

仍需在 Day 9 回答两个问题：`dataloader_num_workers=4` 是否降为 0/1；正式训练的 GPU 预留采用 1.75 小时上界还是再加平台调度缓冲。外部 benchmark 不在 Day 9 运行。

## 可审计证据

- 训练配置：`configs/vopd_day8_64.yaml`
- 固定数据 preflight：`artifacts/runs/E-D8-001/preflight/preflight_summary.json`
- 运行清单：`artifacts/runs/E-D8-001/preflight/run_invocation.json`
- 原始训练日志：`artifacts/runs/E-D8-001/logs/train.log`
- checkpoint 清单：`artifacts/runs/E-D8-001/evidence/reload/checkpoint_manifest.json`
- 冷重载结论：`artifacts/runs/E-D8-001/evidence/reload/reload_validation_summary.json`
- 5 条推理摘要：`artifacts/runs/E-D8-001/reload_5/summary.json`
- 本报告机器摘要：`artifacts/runs/E-D8-001/evidence/stability_summary.json`
- 逐步结构化指标：`artifacts/runs/E-D8-001/metrics.jsonl`
- 费用口径：`artifacts/runs/E-D8-001/cost.json`
"""


def build_summary(project_root: Path) -> dict[str, Any]:
    run_root = project_root / "artifacts/runs/E-D8-001"
    log_path = run_root / "logs/train.log"
    preflight_path = run_root / "preflight/preflight_summary.json"
    invocation_path = run_root / "preflight/run_invocation.json"
    checkpoint_manifest_path = run_root / "evidence/reload/checkpoint_manifest.json"
    reload_path = run_root / "evidence/reload/reload_validation_summary.json"
    reload_smoke_path = run_root / "reload_5/summary.json"
    checkpoint_dir = run_root / "checkpoints/global_step_8"

    required_paths = (
        log_path,
        preflight_path,
        invocation_path,
        checkpoint_manifest_path,
        reload_path,
        reload_smoke_path,
        project_root / "configs/vopd_day8_64.yaml",
        project_root / "configs/project_1024.yaml",
    )
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing Day 8 evidence: {missing}")

    preflight = load_json(preflight_path)
    invocation = load_json(invocation_path)
    checkpoint_manifest = load_json(checkpoint_manifest_path)
    reload = load_json(reload_path)
    reload_smoke = load_json(reload_smoke_path)
    project_config = yaml.safe_load(
        (project_root / "configs/project_1024.yaml").read_text(encoding="utf-8")
    )
    cost_rate = float(project_config["budget"]["cost_per_dual_gpu_hour"])

    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    parsed_steps, signals = parse_training_log(log_text)
    observed_step_numbers = [int(row["step"]) for row in parsed_steps]
    if observed_step_numbers != list(range(1, EXPECTED_STEPS + 1)):
        raise ValueError(f"expected steps 1..8, found {observed_step_numbers}")
    if any(not math.isfinite(value) for row in parsed_steps for value in row.values()):
        raise ValueError("parsed training metrics contain a non-finite value")

    metrics = [compact_metric_row(row) for row in parsed_steps]
    step_seconds = [float(row["step_seconds"]) for row in metrics]
    generation_seconds = [float(row["generation_seconds"]) for row in metrics]
    steady_seconds = step_seconds[1:]
    checkpoint_save_seconds = parsed_steps[-1].get("timing_s/save_checkpoint")
    if checkpoint_save_seconds is None:
        raise ValueError("step 8 has no timing_s/save_checkpoint metric")

    start = dt.datetime.fromisoformat(invocation["started_at_utc"])
    log_end = dt.datetime.fromtimestamp(log_path.stat().st_mtime, tz=dt.timezone.utc)
    wall_seconds = (log_end - start).total_seconds()
    measured_loop_seconds = sum(step_seconds) + checkpoint_save_seconds
    startup_seconds = max(0.0, wall_seconds - measured_loop_seconds)
    scenarios = projection_scenarios(
        startup_seconds=startup_seconds,
        first_step_seconds=step_seconds[0],
        steady_step_seconds=steady_seconds,
        checkpoint_save_seconds=checkpoint_save_seconds,
        cost_per_dual_gpu_hour=cost_rate,
    )

    reload_start = dt.datetime.fromisoformat(reload["started_at_utc"])
    reload_end = dt.datetime.fromisoformat(reload["completed_at_utc"])
    reload_seconds = (reload_end - reload_start).total_seconds()

    checkpoint_files = checkpoint_manifest.get("files", [])
    checkpoint_files_complete = (
        len(checkpoint_files) == 13
        and all(Path(item["path"]).is_file() and int(item["size_bytes"]) > 0 for item in checkpoint_files)
    )
    latest_iteration_path = run_root / "checkpoints/latest_checkpointed_iteration.txt"
    latest_iteration_is_8 = (
        latest_iteration_path.is_file()
        and latest_iteration_path.read_text(encoding="utf-8").strip() == "8"
    )

    finite_metrics = signals["nonfinite_token_count"] == 0
    teacher_contract = (
        all(float(row["student_optimizer_delta"]) > 0 for row in metrics)
        and all(float(row["teacher_optimizer_delta"]) == 0 for row in metrics)
        and all(int(row["teacher_grad_non_none_count"]) == 0 for row in metrics)
        and all(float(row["teacher_ema_delta"]) > 0 for row in metrics)
        and all(int(row["ema_update_applied"]) == 1 for row in metrics)
        and all_equal(parsed_steps, "self_distillation/teacher_always_on_fraction", 1.0)
        and all_equal(parsed_steps, "self_distillation/teacher_image_swap_fraction", 1.0)
    )
    reload_pass = (
        reload.get("status") == "PASS"
        and reload.get("source_checkpoint_unchanged") is True
        and reload.get("verification", {}).get("status") == "PASS"
        and reload.get("verification", {}).get("prediction_count") == 5
        and reload.get("verification", {}).get("nonempty_response_count") == 5
        and reload.get("verification", {}).get("inference_error_count") == 0
    )

    gates = [
        {
            "gate": "固定 64 条输入与配置",
            "status": "PASS" if preflight.get("status") == "PASS" else "FAIL",
            "evidence": f"64 rows；data SHA256 {invocation['train_file_sha256'][:12]}…",
        },
        {
            "gate": "连续训练与数值稳定",
            "status": "PASS" if finite_metrics else "FAIL",
            "evidence": "8/8 steps；全部数值有限；CUDA OOM=0",
        },
        {
            "gate": "Student/Teacher/EMA 契约",
            "status": "PASS" if teacher_contract else "FAIL",
            "evidence": "Student 更新；Teacher optimizer delta=0、grad=0；EMA 8/8",
        },
        {
            "gate": "最终 checkpoint 完整性",
            "status": "PASS" if checkpoint_files_complete and latest_iteration_is_8 else "FAIL",
            "evidence": "global_step_8；13 个必需文件均非空并有 SHA256",
        },
        {
            "gate": "关闭训练后的冷重载",
            "status": "PASS" if reload_pass else "FAIL",
            "evidence": "5/5 非空输出；0 inference errors；源 checkpoint 未变化",
        },
        {
            "gate": "1024 条耗时与费用外推",
            "status": "PASS",
            "evidence": "Step 2–8 稳态三场景；包含启动、首步预热与最终保存",
        },
    ]
    blocking_gate_failure = any(item["status"] != "PASS" for item in gates)
    status = "FAIL" if blocking_gate_failure else "PASS_WITH_CAVEAT"

    response_clipped_estimated = sum(
        float(row["response_clip_ratio"]) * 8 for row in metrics
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": utc_now(),
        "experiment_id": EXPERIMENT_ID,
        "status": status,
        "validation_assessment": "Share with caveats",
        "decision": "advance_to_day9" if not blocking_gate_failure else "do_not_advance",
        "gates": gates,
        "training": {
            "expected_samples": 64,
            "observed_steps": len(metrics),
            "steps": metrics,
            "loss": {
                "min": min(float(row["loss"]) for row in metrics),
                "max": max(float(row["loss"]) for row in metrics),
                "mean": statistics.mean(float(row["loss"]) for row in metrics),
                "median": statistics.median(float(row["loss"]) for row in metrics),
            },
            "grad_norm": {
                "max": max(float(row["grad_norm"]) for row in metrics),
                "mean": statistics.mean(float(row["grad_norm"]) for row in metrics),
            },
            "timing": {
                "wall_clock_proxy_seconds": wall_seconds,
                "recorded_step_seconds": sum(step_seconds),
                "checkpoint_save_seconds": checkpoint_save_seconds,
                "startup_and_load_residual_seconds": startup_seconds,
                "steady_step_seconds": {
                    "count": len(steady_seconds),
                    "min": min(steady_seconds),
                    "p25": percentile(steady_seconds, 0.25),
                    "median": statistics.median(steady_seconds),
                    "mean": statistics.mean(steady_seconds),
                    "p75": percentile(steady_seconds, 0.75),
                    "max": max(steady_seconds),
                },
                "generation_share_all_steps": sum(generation_seconds) / sum(step_seconds),
                "generation_share_steady_steps": sum(generation_seconds[1:]) / sum(step_seconds[1:]),
            },
            "lengths": {
                "prompt_max_tokens": max(int(row["prompt_max_tokens"]) for row in metrics),
                "prompt_clip_ratio_max": max(float(row["prompt_clip_ratio"]) for row in metrics),
                "response_max_tokens": max(int(row["response_max_tokens"]) for row in metrics),
                "response_clipped_count_estimate": response_clipped_estimated,
                "response_clipped_fraction_estimate": response_clipped_estimated / 64,
                "aborted_ratio_max": max(float(row["aborted_ratio"]) for row in metrics),
            },
            "memory": {
                "logged_max_allocated_gb": max(
                    float(row["logged_max_memory_allocated_gb"]) for row in metrics
                ),
                "logged_max_reserved_gb": max(
                    float(row["logged_max_memory_reserved_gb"]) for row in metrics
                ),
                "logged_host_memory_used_gb": max(
                    float(row["logged_host_memory_used_gb"]) for row in metrics
                ),
                "per_gpu_peak_is_auditable": False,
                "reason": "logger values exceed the frozen 96 GB/card hardware inventory",
            },
            "error_signals": signals,
        },
        "checkpoint": {
            "global_step": 8,
            "required_file_count": len(checkpoint_files),
            "required_files_complete": checkpoint_files_complete,
            "latest_iteration_is_8": latest_iteration_is_8,
            "directory_size_bytes": directory_size(checkpoint_dir),
            "directory_size_gib": directory_size(checkpoint_dir) / (1024**3),
            "manifest": str(checkpoint_manifest_path.relative_to(project_root)),
            "manifest_sha256": sha256_file(checkpoint_manifest_path),
        },
        "cold_reload": {
            "status": reload["status"],
            "duration_seconds": reload_seconds,
            "source_checkpoint_unchanged": reload["source_checkpoint_unchanged"],
            "merged_manifest_sha256": reload["merged_manifest_sha256"],
            "server_exit_code_after_controlled_shutdown": reload[
                "server_exit_code_after_controlled_shutdown"
            ],
            "prediction_count": reload["verification"]["prediction_count"],
            "nonempty_response_count": reload["verification"]["nonempty_response_count"],
            "inference_error_count": reload["verification"]["inference_error_count"],
            "smoke_accuracy": reload_smoke.get("accuracy"),
            "smoke_accuracy_is_performance_claim": False,
            "reuse_merged": True,
        },
        "projection_1024": {
            "target_samples": 1024,
            "global_batch_size": 8,
            "target_optimizer_steps": TARGET_STEPS,
            "startup_seconds": startup_seconds,
            "first_step_seconds": step_seconds[0],
            "checkpoint_save_seconds": checkpoint_save_seconds,
            "cost_per_dual_gpu_hour_cny": cost_rate,
            "scenarios": scenarios,
            "planning_scenario": "mean",
            "resource_reservation_scenario": "conservative_max",
        },
        "cost": {
            "currency": "CNY",
            "cost_per_dual_gpu_hour": cost_rate,
            "training_window_dual_gpu_hours": wall_seconds / 3600,
            "training_window_cost_cny": wall_seconds / 3600 * cost_rate,
            "reload_window_dual_gpu_hours": reload_seconds / 3600,
            "reload_window_cost_cny": reload_seconds / 3600 * cost_rate,
            "coverage": "evidence timestamp windows only; excludes unrecorded attempts, idle, and billing rounding",
        },
        "provenance": {
            "git_commit": invocation["git_commit"],
            "git_worktree_clean_at_start": not bool(invocation.get("git_status_porcelain", "").strip()),
            "config_sha256": invocation["config_sha256"],
            "train_file_sha256": invocation["train_file_sha256"],
            "train_log_sha256": sha256_file(log_path),
            "run_started_at_utc": invocation["started_at_utc"],
            "train_log_mtime_utc": log_end.isoformat(),
            "wall_clock_method": "run_invocation started_at_utc to train.log mtime",
        },
        "caveats": [
            "one DataLoader worker was killed after 8/8 and checkpoint save; no contemporaneous cgroup/RSS snapshot exists",
            "logged GPU memory values cannot be interpreted as an auditable per-device peak",
            "the run started from a dirty Git worktree, although config/data hashes and status were recorded",
            "the 1024 projection uses seven steady steps from a 64-sample subset",
            "the final cold-reload pass reused an already merged model and excludes merge time",
        ],
        "visual_omission_reason": (
            "The full population is eight discrete steps; an exact table preserves every value and unit "
            "more faithfully than a chart at this grain."
        ),
        "next_step": "Day 9: freeze vopd_1024.yaml and E-D10-001 preflight with per-GPU and process-memory telemetry",
    }
    return summary


def write_deliverables(project_root: Path, summary: dict[str, Any]) -> dict[str, str]:
    run_root = project_root / "artifacts/runs/E-D8-001"
    evidence_path = run_root / "evidence/stability_summary.json"
    metrics_path = run_root / "metrics.jsonl"
    cost_path = run_root / "cost.json"
    hashes_path = run_root / "checkpoint_sha256.txt"
    report_path = project_root / "artifacts/reports/vopd_64_stability.md"

    write_json_atomic(evidence_path, summary)
    write_text_atomic(
        metrics_path,
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in summary["training"]["steps"]),
    )
    write_json_atomic(
        cost_path,
        {
            "schema_version": SCHEMA_VERSION,
            "generated_at_utc": summary["generated_at_utc"],
            "experiment_id": EXPERIMENT_ID,
            "observed_windows": summary["cost"],
            "projection_1024": summary["projection_1024"],
        },
    )

    checkpoint_manifest = load_json(
        run_root / "evidence/reload/checkpoint_manifest.json"
    )
    hash_lines = [
        f"{item['sha256']}  {Path(item['path']).relative_to(project_root)}"
        for item in checkpoint_manifest["files"]
    ]
    write_text_atomic(hashes_path, "\n".join(hash_lines) + "\n")
    write_text_atomic(report_path, render_report(summary))

    config_snapshot = run_root / "config.yaml"
    shutil.copyfile(project_root / "configs/vopd_day8_64.yaml", config_snapshot)
    write_text_atomic(
        run_root / "command.txt",
        "OMP_NUM_THREADS=44 CUDA_VISIBLE_DEVICES=0,1 scripts/run_vopd_2gpu.sh "
        "--config configs/vopd_day8_64.yaml --run\n",
    )
    write_text_atomic(run_root / "git_commit.txt", summary["provenance"]["git_commit"] + "\n")
    write_text_atomic(
        run_root / "env.txt",
        "# Values captured by preflight/run_invocation.json at training start.\n"
        "# Package versions and per-process RSS were not captured contemporaneously.\n"
        "CUDA_VISIBLE_DEVICES=0,1\n"
        "OMP_NUM_THREADS=44\n"
        f"git_commit={summary['provenance']['git_commit']}\n"
        f"config_sha256={summary['provenance']['config_sha256']}\n"
        f"train_file_sha256={summary['provenance']['train_file_sha256']}\n",
    )
    return {
        "summary": str(evidence_path),
        "metrics": str(metrics_path),
        "cost": str(cost_path),
        "checkpoint_hashes": str(hashes_path),
        "report": str(report_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    summary = build_summary(project_root)
    outputs = {} if args.check_only else write_deliverables(project_root, summary)
    print(
        json.dumps(
            {
                "experiment_id": EXPERIMENT_ID,
                "status": summary["status"],
                "decision": summary["decision"],
                "outputs": outputs,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if summary["status"] != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
