#!/usr/bin/env python3
"""Audit Day 9 Task 2 inputs and paths without launching GPU work."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import yaml


EXPERIMENT_ID = "E-D10-001"
TRAIN_PATH = Path("/root/autodl-tmp/data/vision_opd_1024/train_1024.parquet")
TRAIN_SHA256 = "84062ec65baea0e18d5c712e7ee89c814105c0f1a51cd50df2e369b6036258fa"
BASE_PATH = Path("/root/autodl-tmp/models/Qwen3.5-4B")
BASE_SHARDS = {
    "model.safetensors-00001-of-00002.safetensors": "26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61",
    "model.safetensors-00002-of-00002.safetensors": "cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188",
}
REQUIRED_COLUMNS = {
    "data_source", "prompt", "images", "bbox_images", "ability",
    "reward_model", "extra_info",
}
REQUIRED_MODEL_FILES = {
    "config.json", "model.safetensors.index.json", "preprocessor_config.json",
    "tokenizer.json", "tokenizer_config.json", *BASE_SHARDS,
}
PREFLIGHT_WHITELIST = {
    "autodl_billing_input.json", "base_model_manifest.json", "budget_evidence.md",
    "budget_projection.json", "budget_report_artifact.json", "data_manifest.json",
    "git_state.json", "output_path_gate.json", "preflight_summary.json",
    "storage_gate.json", "task2_readiness.json", "task2_readiness.md",
    "task3_config_freeze.json", "task3_config_freeze.md",
}
GIB = 1024 ** 3

FORMAL_CONFIG_EXPECTED = {
    "experiment.id": EXPERIMENT_ID,
    "experiment.name": "vision-opd-qwen35-4b-day10-formal-1024",
    "experiment.group_name": EXPERIMENT_ID,
    "experiment.prefix_source": "online",
    "experiment.seed": 42,
    "paths.model": str(BASE_PATH),
    "paths.train_file": str(TRAIN_PATH),
    "paths.output_dir": "artifacts/runs/E-D10-001",
    "data.expected_train_rows": 1024,
    "data.train_batch_size": 8,
    "data.max_prompt_length": 8192,
    "data.max_response_length": 256,
    "data.truncation": "error",
    "data.shuffle": True,
    "data.dataloader_num_workers": 0,
    "actor.learning_rate": 2.0e-6,
    "actor.ppo_mini_batch_size": 8,
    "actor.use_dynamic_batch_size": True,
    "actor.gradient_checkpointing": True,
    "actor.max_token_length_per_gpu": 8448,
    "rollout.n": 1,
    "rollout.tensor_model_parallel_size": 1,
    "rollout.gpu_memory_utilization": 0.45,
    "rollout.log_prob_micro_batch_size_per_gpu": 1,
    "rollout.agent_num_workers": 2,
    "self_distillation.loss_mode": "vopd",
    "self_distillation.top_k": 100,
    "self_distillation.alpha": 0.5,
    "self_distillation.teacher_always_on": True,
    "self_distillation.teacher_model_source": "legacy",
    "self_distillation.teacher_regularization": "ema",
    "self_distillation.teacher_update_rate": 0.05,
    "self_distillation.dont_reprompt_on_self_success": True,
    "self_distillation.include_environment_feedback": False,
    "self_distillation.importance_sampling_clip": 2.0,
    "resources.nodes": 1,
    "resources.gpus_per_node": 2,
    "training.expected_samples": 1024,
    "training.total_optimizer_steps": 128,
    "training.total_epochs": 1,
    "training.require_full_epoch": True,
    "training.save_frequency": -1,
    "training.test_frequency": -1,
    "training.max_actor_ckpt_to_keep": 1,
    "training.resume_mode": "disable",
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(content, encoding="utf-8")
    temp.replace(path)


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def audit_data() -> dict[str, Any]:
    if not TRAIN_PATH.is_file():
        return {"status": "FAIL", "path": str(TRAIN_PATH), "finding": "train parquet missing"}
    table = pq.read_table(TRAIN_PATH)
    rows = table.to_pylist()
    columns = set(table.column_names)
    sample_ids: list[str] = []
    prompt_empty = 0
    missing_student_paths: list[str] = []
    missing_teacher_paths: list[str] = []
    empty_student = 0
    empty_teacher = 0
    for row in rows:
        prompt = row.get("prompt") or []
        if not prompt or not any(str(item.get("content", "")).strip() for item in prompt):
            prompt_empty += 1
        provenance = ((row.get("extra_info") or {}).get("provenance") or {})
        sample_ids.append(str(provenance.get("sample_id") or ""))
        for key, missing, empty_name in (
            ("images", missing_student_paths, "student"),
            ("bbox_images", missing_teacher_paths, "teacher"),
        ):
            values = row.get(key) or []
            if not values:
                if empty_name == "student":
                    empty_student += 1
                else:
                    empty_teacher += 1
            for value in values:
                path = Path(str((value or {}).get("path") or ""))
                if not path.is_file():
                    missing.append(str(path))
    cached_columns = sorted(columns & {
        "cached_prefix", "cached_response", "response", "rollout_response", "teacher_response"
    })
    digest = sha256_file(TRAIN_PATH)
    checks = {
        "sha256_frozen": digest == TRAIN_SHA256,
        "row_count_1024": table.num_rows == 1024,
        "required_columns": REQUIRED_COLUMNS <= columns,
        "sample_id_nonempty": all(sample_ids),
        "sample_id_unique": len(set(sample_ids)) == len(sample_ids),
        "prompt_nonempty": prompt_empty == 0,
        "student_images_present": empty_student == 0 and not missing_student_paths,
        "teacher_images_present": empty_teacher == 0 and not missing_teacher_paths,
        "no_cached_response_columns": not cached_columns,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "dataset": "frozen Vision-OPD train split",
        "grain": "one multimodal training example per sample_id",
        "path": str(TRAIN_PATH),
        "sha256": digest,
        "expected_sha256": TRAIN_SHA256,
        "row_count": table.num_rows,
        "columns": table.column_names,
        "null_count_by_column": {name: table[name].null_count for name in table.column_names},
        "unique_sample_ids": len(set(sample_ids)),
        "prompt_empty_count": prompt_empty,
        "empty_student_image_rows": empty_student,
        "empty_teacher_image_rows": empty_teacher,
        "missing_student_image_count": len(missing_student_paths),
        "missing_teacher_image_count": len(missing_teacher_paths),
        "cached_response_columns": cached_columns,
        "checks": checks,
    }


def audit_base() -> dict[str, Any]:
    files = sorted(item for item in BASE_PATH.iterdir() if item.is_file()) if BASE_PATH.is_dir() else []
    missing = sorted(REQUIRED_MODEL_FILES - {item.name for item in files})
    manifest = []
    frozen_ok = True
    for item in files:
        if item.name in REQUIRED_MODEL_FILES:
            digest = sha256_file(item)
            expected = BASE_SHARDS.get(item.name)
            if expected and digest != expected:
                frozen_ok = False
            manifest.append({"name": item.name, "size_bytes": item.stat().st_size, "sha256": digest, "expected_sha256": expected})
    status = "PASS" if BASE_PATH.is_dir() and not missing and frozen_ok else "FAIL"
    return {
        "status": status, "path": str(BASE_PATH), "missing_required_files": missing,
        "frozen_weight_hashes_match": frozen_ok, "total_size_bytes": directory_size(BASE_PATH) if BASE_PATH.is_dir() else 0,
        "files": manifest,
    }


def nested_get(value: dict[str, Any], dotted: str) -> Any:
    current: Any = value
    for key in dotted.split("."):
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def audit_config(project_root: Path, config_path: Path | None = None) -> dict[str, Any]:
    path = config_path or project_root / "configs/vopd_1024.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    comparisons = {
        key: {"expected": wanted, "actual": nested_get(config, key), "match": nested_get(config, key) == wanted}
        for key, wanted in FORMAL_CONFIG_EXPECTED.items()
    }
    comparisons["legacy_smoke_absent"] = {
        "expected": True,
        "actual": "smoke" not in config,
        "match": "smoke" not in config,
    }
    return {
        "status": "PASS" if all(item["match"] for item in comparisons.values()) else "FAIL",
        "path": str(path.relative_to(project_root)) if path.is_relative_to(project_root) else str(path),
        "comparisons": comparisons,
        "handoff": "Task 3 formal parameter contract is frozen; rerun this gate after any config edit.",
    }


def run_git(project_root: Path, *args: str) -> tuple[int, str]:
    result = subprocess.run(["git", *args], cwd=project_root, text=True, capture_output=True, check=False)
    return result.returncode, (result.stdout + result.stderr).strip()


def audit_git(project_root: Path) -> dict[str, Any]:
    _, commit = run_git(project_root, "rev-parse", "HEAD")
    _, branch = run_git(project_root, "branch", "--show-current")
    _, porcelain = run_git(project_root, "status", "--porcelain=v1")
    diff_code, diff_output = run_git(project_root, "diff", "--check")
    clean = not porcelain
    return {
        "status": "PASS" if clean and diff_code == 0 else "PENDING_COMMIT",
        "commit": commit, "branch": branch, "clean": clean,
        "porcelain": porcelain.splitlines() if porcelain else [],
        "diff_check_status": "PASS" if diff_code == 0 else "FAIL",
        "diff_check_output": diff_output,
    }


def audit_output(project_root: Path, output_dir: Path) -> dict[str, Any]:
    runs_root = (project_root / "artifacts/runs").resolve()
    resolved = output_dir.resolve()
    inside_runs = resolved == runs_root / EXPERIMENT_ID and str(resolved).startswith(str(runs_root) + os.sep)
    preflight = output_dir / "preflight"
    unexpected: list[str] = []
    if output_dir.exists():
        for item in output_dir.rglob("*"):
            if item.is_file():
                rel = item.relative_to(output_dir)
                if len(rel.parts) != 2 or rel.parts[0] != "preflight" or rel.name not in PREFLIGHT_WHITELIST:
                    unexpected.append(str(rel))
    symlink_free = not output_dir.is_symlink() and not preflight.is_symlink()
    writable = os.access(preflight if preflight.exists() else output_dir, os.W_OK)
    status = "PASS" if inside_runs and not unexpected and symlink_free and writable else "FAIL"
    log_path = output_dir / "logs/train.log"
    log_ok = str(log_path.resolve()).startswith(str(resolved) + os.sep) and not log_path.exists() and not log_path.is_symlink()
    return {
        "status": status, "path": str(output_dir), "resolved_path": str(resolved),
        "inside_expected_runs_root": inside_runs, "symlink_free": symlink_free,
        "writable": writable, "unexpected_existing_files": unexpected,
        "allowed_preflight_files": sorted(PREFLIGHT_WHITELIST),
        "log_path": {"status": "PASS" if log_ok else "FAIL", "path": str(log_path), "collision": log_path.exists()},
    }


def audit_storage(output_dir: Path, checkpoint_path: Path) -> dict[str, Any]:
    usage = shutil.disk_usage(output_dir)
    checkpoint_size = directory_size(checkpoint_path)
    required = 2 * checkpoint_size + 5 * GIB
    shortage = max(0, required - usage.free)
    return {
        "status": "PASS" if shortage == 0 else "FAIL",
        "filesystem_probe": str(output_dir), "total_bytes": usage.total, "used_bytes": usage.used,
        "available_bytes": usage.free, "day8_checkpoint_path": str(checkpoint_path),
        "day8_checkpoint_size_bytes": checkpoint_size,
        "formula": "2 * final_checkpoint_estimate + 5 GiB",
        "required_bytes": required, "shortage_bytes": shortage,
        "remediation": "Free or add sufficient storage, then rerun Task 2. No files were deleted by this audit." if shortage else None,
    }


def build_markdown(report: dict[str, Any]) -> str:
    gates = report["gates"]
    rows = "\n".join(f"| {name} | {value['status']} | {value['evidence']} |" for name, value in gates.items())
    findings = "\n".join(
        f"- **{item['severity']} / {item['confidence']}**：{item['finding']} 影响：{item['impact']} 处理：{item['remediation']}"
        for item in report["findings"]
    )
    storage = report["storage"]
    if report["config"]["status"] == "PASS":
        conclusion = (
            "正式 E-D10-001 配置已经冻结并通过身份检查。任务 2/3 的审计产物完整，"
            "但 Git 尚未提交且磁盘容量不足，因此 readiness 仍为 `BLOCKED`，不能进入 Day 10。"
        )
    else:
        conclusion = (
            "任务 2 的审计已完成，但正式 config 尚未冻结；同时 Git 或磁盘 Gate 仍可能阻塞。"
            "下一步必须完成任务 3，且所有 Gate 通过前不能进入 Day 10。"
        )
    return f"""# Day 9 Task 2：E-D10-001 正式训练输入与路径审计

> 产物状态：**{report['artifact_status']}**  
> Readiness：**{report['readiness_status']}**  
> 审计时间：{report['generated_at_utc']}

## 结论

{conclusion}

## Gate

| 检查项 | 状态 | 证据 |
|---|---|---|
{rows}

## 数据质量发现

{findings}

## 磁盘计算

- Day 8 checkpoint：{storage['day8_checkpoint_size_bytes']} bytes。
- 要求：`2 × checkpoint + 5 GiB` = {storage['required_bytes']} bytes。
- 当前可用：{storage['available_bytes']} bytes；缺口：{storage['shortage_bytes']} bytes。
- 审计未删除、移动任何文件，也未启动 GPU。

## 范围与限制

数据粒度为每个 `sample_id` 一条多模态训练样本。费用控制值来自用户在当前会话报告的 AutoDL 控制台累计费用，不是仓库账单导出。离散 readiness 快照不适合做时间趋势分析。任务 3 才负责修改并冻结 config；本任务只读核验。
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--day8-checkpoint", type=Path, default=Path("artifacts/runs/E-D8-001/checkpoints/global_step_8"))
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    output_dir = (args.output_dir or project_root / "artifacts/runs/E-D10-001").resolve()
    preflight = output_dir / "preflight"
    preflight.mkdir(parents=True, exist_ok=True)

    data = audit_data()
    base = audit_base()
    config = audit_config(project_root)
    git = audit_git(project_root)
    output = audit_output(project_root, output_dir)
    storage = audit_storage(output_dir, (project_root / args.day8_checkpoint).resolve())
    budget_path = preflight / "budget_projection.json"
    budget = json.loads(budget_path.read_text(encoding="utf-8")) if budget_path.is_file() else {}
    budget_status = budget.get("gate_status", "MISSING")
    gates = {
        "data": {"status": data["status"], "evidence": f"{data.get('row_count', 0)} rows; frozen SHA256 match={data.get('checks', {}).get('sha256_frozen', False)}"},
        "base": {"status": base["status"], "evidence": f"frozen shard hashes match={base.get('frozen_weight_hashes_match', False)}"},
        "config_identity": {"status": config["status"], "evidence": "E-D10 identity and 1024/128 formal settings"},
        "git_state": {"status": git["status"], "evidence": f"commit={git['commit'][:12]}; clean={git['clean']}; diff-check={git['diff_check_status']}"},
        "output_directory": {"status": output["status"], "evidence": f"unexpected files={len(output['unexpected_existing_files'])}"},
        "log_path": {"status": output["log_path"]["status"], "evidence": output["log_path"]["path"]},
        "storage": {"status": storage["status"], "evidence": f"required={storage['required_bytes']}; available={storage['available_bytes']}; shortage={storage['shortage_bytes']}"},
        "budget": {"status": budget_status, "evidence": f"current={budget.get('project_cap', {}).get('current_platform_cumulative_charge_cny')} CNY; projected={budget.get('project_cap', {}).get('projected_total_after_e_d10_reservation_cny')} CNY"},
    }
    blocking = [name for name, gate in gates.items() if gate["status"] != "PASS"]
    findings = []
    if config["status"] != "PASS":
        findings.append({"severity": "HIGH", "confidence": "HIGH", "finding": "configs/vopd_1024.yaml still identifies the Day 7 smoke run.", "impact": "The current launcher would write to E-D7-001 and run 16 samples / 2 steps.", "remediation": "Complete Task 3 parameter freeze before any preflight launcher invocation."})
    if git["status"] != "PASS":
        findings.append({"severity": "MEDIUM", "confidence": "HIGH", "finding": "The working tree contains uncommitted Day 9 artifacts and scripts.", "impact": "The exact launch state is not yet represented by a commit.", "remediation": "After Task 3-5 review, commit the frozen config, scripts, tests, and evidence together."})
    if storage["status"] != "PASS":
        findings.append({"severity": "CRITICAL", "confidence": "HIGH", "finding": f"Storage is short by {storage['shortage_bytes']} bytes against the frozen formula.", "impact": "Checkpoint creation or retention could exhaust the filesystem.", "remediation": "Free or add capacity without deleting evidence blindly, then rerun this audit."})
    if not findings:
        findings.append({"severity": "INFO", "confidence": "HIGH", "finding": "All Task 2 gates pass.", "impact": "Task 3 may proceed.", "remediation": "Continue with parameter freeze."})
    report = {
        "schema_version": 1, "generated_at_utc": utc_now(), "experiment_id": EXPERIMENT_ID,
        "purpose": "day9_task2_formal_training_inputs_and_paths",
        "artifact_status": "COMPLETE", "task2_completed": True,
        "readiness_status": "PASS" if not blocking else "BLOCKED",
        "advance_to_task3": True, "advance_to_day10": not blocking,
        "blocking_gates": blocking, "gates": gates, "data": data, "base": base,
        "config": config, "git": git, "output": output, "storage": storage,
        "budget_source": str(budget_path), "findings": findings,
        "assumptions": ["User-reported AutoDL cumulative cost is 200 CNY at audit time.", "Day 8 global_step_8 size is the final-checkpoint estimate."],
        "temporal_analysis": {"applicable": False, "reason": "This is a point-in-time readiness audit, not a continuous series."},
    }
    write_json(preflight / "data_manifest.json", data)
    write_json(preflight / "base_model_manifest.json", base)
    write_json(preflight / "git_state.json", git)
    write_json(preflight / "output_path_gate.json", output)
    write_json(preflight / "storage_gate.json", storage)
    write_json(preflight / "task2_readiness.json", report)
    write_text(preflight / "task2_readiness.md", build_markdown(report))
    print(f"ARTIFACT_STATUS={report['artifact_status']}")
    print(f"READINESS_STATUS={report['readiness_status']}")
    print(f"BLOCKING_GATES={','.join(blocking)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
