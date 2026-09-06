#!/usr/bin/env python3
"""Build the fail-closed Day 11 aggregate preflight for formal 6K training."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.day11_validation_evidence import collect_validation_evidence
from scripts.freeze_formal_cpu import verify as verify_cpu_freeze

RUN_ROOT = ROOT / "artifacts/runs/E-D11-6K-GATE-001"
DEFAULT_OUTPUT = RUN_ROOT / "preflight.json"
GIB = 1024**3

PATHS = {
    "static_gate": RUN_ROOT / "static_gate.json",
    "freeze": RUN_ROOT / "training_config_freeze.json",
    "budget": RUN_ROOT / "budget_freeze.json",
    "pilot_16": RUN_ROOT / "pilot/16/evidence/postflight.json",
    "pilot_64": RUN_ROOT / "pilot/64/evidence/postflight.json",
    "cold_reload": RUN_ROOT / "pilot/64/cold_reload_attempt_002/reload_validation_summary.json",
    "pilot_64_resource": RUN_ROOT / "pilot/64/evidence/resource_summary.json",
    "formal_config": ROOT / "configs/vopd_6241.yaml",
    "formal_candidate_config": ROOT / "configs/vopd_6241_candidate.yaml",
    "formal_policy": ROOT / "configs/vopd_6241_abort_policy.yaml",
    "project_config": ROOT / "configs/project_6241.yaml",
    "candidate_gate_freeze": RUN_ROOT / "formal_candidate_validation_v1/formal_gate_freeze.json",
    "promotion_receipt": RUN_ROOT / "formal_promotion_v1/promotion_receipt.json",
}
VALIDATION_ROOT = RUN_ROOT / 'memory_optimization/fixed_validation_v1'
PATHS.update({
    'fixed_baseline': VALIDATION_ROOT / 'fixed_baseline/run/evidence/postflight.json',
    'fixed_deferred': VALIDATION_ROOT / 'fixed_deferred/run/evidence/postflight.json',
    'fixed_comparison': VALIDATION_ROOT / 'fixed_comparison.json',
    'pressure': VALIDATION_ROOT / 'pressure_v2/run/evidence/postflight.json',
})


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


def memory_capacity_bytes() -> int:
    candidates: list[int] = []
    path = Path("/sys/fs/cgroup/memory.max")
    if path.is_file():
        value = path.read_text(encoding="utf-8").strip()
        if value != "max":
            candidates.append(int(value))
    try:
        import psutil
        candidates.append(int(psutil.virtual_memory().total))
    except ImportError:
        pass
    return min(candidates) if candidates else 0


def source_entry_matches(entry: dict[str, Any]) -> bool:
    path = Path(str(entry.get("path", "")))
    return path.is_file() and entry.get("sha256") == sha256_file(path)


def decision_status(
    evidence_checks: dict[str, bool],
    runtime_checks: dict[str, bool],
    safety_checks: dict[str, bool],
    release_check: bool,
) -> str:
    if not all(evidence_checks.values()):
        return "FAIL_EVIDENCE_INTEGRITY"
    if not all(runtime_checks.values()):
        return "BLOCKED_RUNTIME_RESOURCES"
    if not all(safety_checks.values()):
        return "BLOCKED_ADDITIONAL_RESOURCE_VALIDATION"
    if not release_check:
        return "READY_TO_UNBLOCK_FORMAL_CONFIG"
    return "PASS"


def build_preflight(
    paths: dict[str, Path],
    *,
    disk_free_bytes: int,
    cpu_capacity_bytes: int,
    generated_at: str | None = None,
) -> dict[str, Any]:
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        return {
            "schema_version": 1,
            "generated_at_utc": generated_at or utc_now(),
            "status": "FAIL_MISSING_INPUTS",
            "artifact_status": "COMPLETE",
            "formal_training_authorized": False,
            "missing_inputs": missing,
        }
    static, freeze, budget, p16, p64, reload, pilot_resource = (
        load_json(paths[name])
        for name in (
            "static_gate", "freeze", "budget", "pilot_16", "pilot_64",
            "cold_reload", "pilot_64_resource",
        )
    )
    formal = yaml.safe_load(paths["formal_config"].read_text(encoding="utf-8"))
    policy = yaml.safe_load(paths["formal_policy"].read_text(encoding="utf-8"))
    project = yaml.safe_load(paths["project_config"].read_text(encoding="utf-8"))
    validation = collect_validation_evidence(paths)
    diagnostics_valid = validation['status'] == 'PASS_DIAGNOSTIC_EVIDENCE'
    pressure = validation.get('pressure', {})

    freeze_p16 = freeze.get("verification", {}).get("pilot_16_postflight", {})
    freeze_p64 = freeze.get("verification", {}).get("pilot_64_postflight", {})
    freeze_reload = freeze.get("verification", {}).get("pilot_64_cold_reload", {})
    budget_sources = budget.get("sources", {})
    static_inputs = static.get("inputs", {})
    evidence_checks = {
        'latest_diagnostic_evidence_integrity': diagnostics_valid,
        "static_gate_pass_and_inputs_current": (
            static.get("status") == "PASS_PENDING_GPU_PILOT"
            and all(static.get("checks", {}).values())
            and all(source_entry_matches(entry) for entry in static_inputs.values())
        ),
        "training_freeze_matches_formal_config_and_policy": (
            freeze.get("hashes", {}).get("vopd_config") == sha256_file(paths["formal_config"])
            and freeze.get("hashes", {}).get("abort_policy") == sha256_file(paths["formal_policy"])
        ),
        "pilot_16_pass_and_frozen": (
            p16.get("status") == "PASS" and p16.get("stage_gate_pass") is True
            and freeze_p16.get("status") == "PASS"
            and freeze_p16.get("sha256") == sha256_file(paths["pilot_16"])
        ),
        "pilot_64_pass_and_frozen": (
            p64.get("status") == "PASS" and p64.get("stage_gate_pass") is True
            and freeze_p64.get("status") == "PASS"
            and freeze_p64.get("sha256") == sha256_file(paths["pilot_64"])
        ),
        "cold_reload_pass_bound_to_pilot_64": (
            reload.get("status") == "PASS"
            and reload.get("experiment_id") == p64.get("experiment_id")
            and reload.get("source_checkpoint_unchanged") is True
            and reload.get("verification", {}).get("status") == "PASS"
            and p64.get("reload_report") == str(paths["cold_reload"].resolve())
            and freeze_reload.get("sha256") == sha256_file(paths["cold_reload"])
        ),
        "budget_frozen_and_sources_current": (
            budget.get("status") == "PASS_BUDGET_FROZEN_WITH_RESOURCE_CAVEATS"
            and budget.get("project_cap", {}).get("budget_pass") is True
            and source_entry_matches(budget_sources.get("pilot_64_postflight", {}))
            and source_entry_matches(budget_sources.get("formal_abort_policy", {}))
            and source_entry_matches(budget_sources.get("historical_billing_observation", {}))
        ),
        "formal_data_and_drop_last_contract": (
            int(formal["data"]["expected_train_rows"]) == 6241
            and int(formal["training"]["expected_samples"]) == 6240
            and int(formal["training"]["dropped_rows"]) == 1
            and formal["data"]["tail_policy"] == "native_drop_last"
            and project["training_contract"]["tail_policy"] == "native_drop_last"
        ),
        "pilot_resource_summary_bound_to_training_run": (
            pilot_resource.get("training_gate_pass") is True
            and int(pilot_resource.get("observed_steps", 0)) == 8
            and int(pilot_resource.get("cpu_capacity_bytes", 0)) > 0
            and len(pilot_resource.get("gpu_peaks", [])) == 2
        ),
    }

    required_disk = int(policy["disk"]["prelaunch_required_bytes"])
    formal_cpu_floor = int(policy["memory"]["prelaunch_cgroup_minimum_bytes"])
    reviewed_cpu_floor = 240 * GIB
    pilot_cpu_capacity = int(pilot_resource["cpu_capacity_bytes"])
    runtime_checks = {
        "pilot_runtime_capacity_met_reviewed_240_gib": pilot_cpu_capacity >= reviewed_cpu_floor,
        "disk_free_meets_formal_120_gib": disk_free_bytes >= required_disk,
    }

    coverage = budget["coverage"]
    response_stress_floor = int(0.75 * int(coverage["configured_response_limit_tokens"]))
    safety_checks = {
        "formal_cpu_floor_matches_reviewed_240_gib": (
            verify_cpu_freeze() and formal_cpu_floor == reviewed_cpu_floor
            and formal['resources']['prelaunch_cgroup_minimum_bytes'] == formal_cpu_floor
        ),
        'diagnostic_gpu_peaks_below_formal_abort_line': (
            diagnostics_valid and pressure['gpu_peak_ratio'] < float(policy['memory']['gpu_used_ratio_abort'])
            and pressure['marker_peak_ratio'] < float(policy['memory']['gpu_used_ratio_abort'])
        ),
        'diagnostic_cpu_peak_below_formal_abort_line': (
            diagnostics_valid and pressure['cpu_peak_ratio'] < float(policy['memory']['cgroup_used_ratio_abort'])
        ),
        'at_least_two_post_warmup_steps_observed': (
            diagnostics_valid and pressure['coverage']['passed'] is True
        ),
        'long_response_training_pressure_observed': (
            diagnostics_valid and pressure['coverage']['passed'] is True
            and pressure['maximum_response_tokens'] >= response_stress_floor
        ),
        'diagnostic_length_and_warmup_match_formal': (
            diagnostics_valid and formal['data']['max_response_length'] == 1024
            and formal['actor']['lr_warmup_steps'] == 10
        ),
        'formal_uses_natural_eos': formal['rollout']['ignore_eos'] is False,
        # Diagnostic evidence is not a final-candidate receipt. The later candidate
        # promotion workflow must replace this pending gate with bound validation.
        # Neither a YAML status nor diagnostic PASS can authorize that promotion.
        'formal_candidate_validation_bound': False,
    }
    formal_config_released = formal.get("status") == "ready_for_formal_training"
    status = decision_status(evidence_checks, runtime_checks, safety_checks, formal_config_released)
    all_checks = {**evidence_checks, **runtime_checks, **safety_checks, "formal_config_released": formal_config_released}
    blockers = sorted(name for name, passed in all_checks.items() if not passed)

    return {
        "schema_version": 1,
        "gate_id": "E-D11-6K-FINAL-PREFLIGHT-001",
        "audit_revision": "latest_diagnostic_evidence_v2",
        "generated_at_utc": generated_at or utc_now(),
        "experiment_id": "E-D12-6K-VOPD-001",
        "artifact_status": "COMPLETE",
        "status": status,
        "gpu_used_to_build_report": False,
        "formal_training_authorized": status == "PASS",
        "ready_to_unblock_formal_config": (
            all(evidence_checks.values()) and all(runtime_checks.values()) and all(safety_checks.values())
        ),
        "checks": all_checks,
        "blocking_gates": blockers,
        "evidence_checks": evidence_checks,
        "runtime_checks": runtime_checks,
        "safety_checks": safety_checks,
        'latest_validation': validation,
        'gate_implementation': {
            str(path.resolve()): sha256_file(path)
            for path in (Path(__file__), ROOT / 'scripts/day11_validation_evidence.py', ROOT / 'scripts/freeze_formal_cpu.py')
        },
        'formal_candidate': {
            'status': 'PENDING_FORMAL_CANDIDATE_VALIDATION',
            'reason': 'Historical replay / forced-EOS pressure runs do not validate the final natural-generation candidate.',
            'automatic_promotion_supported': False,
            'training_resume_validated': False,
        },
        "runtime_snapshot": {
            "builder_process_cpu_capacity_bytes": cpu_capacity_bytes,
            "builder_process_cpu_capacity_gib": cpu_capacity_bytes / GIB,
            "builder_process_capacity_is_launch_evidence": False,
            "pilot_runtime_cpu_capacity_bytes": pilot_cpu_capacity,
            "pilot_runtime_cpu_capacity_gib": pilot_cpu_capacity / GIB,
            "pilot_runtime_cpu_peak_bytes": int(pilot_resource["cpu_peak_bytes"]),
            "pilot_runtime_cpu_peak_gib": int(pilot_resource["cpu_peak_bytes"]) / GIB,
            "reviewed_cpu_floor_bytes": reviewed_cpu_floor,
            "formal_policy_cpu_floor_bytes": formal_cpu_floor,
            "disk_free_bytes": disk_free_bytes,
            "disk_free_gib": disk_free_bytes / GIB,
            "disk_required_bytes": required_disk,
            "disk_required_gib": required_disk / GIB,
            "disk_deficit_bytes": max(0, required_disk - disk_free_bytes),
            "disk_deficit_gib": max(0, required_disk - disk_free_bytes) / GIB,
        },
        "pilot_coverage": {
            **coverage,
            "required_post_warmup_steps": 2,
            "long_response_pressure_floor_tokens": response_stress_floor,
            "note": "Historical Pilot-64 / budget coverage retained unchanged, not used to reject newer valid pressure coverage. The 75% floor is a project criterion, not a paper hyperparameter.",
        },
        "budget": {
            "status": budget["status"],
            "planning": budget["selected_budget"],
            "project_cap": budget["project_cap"],
        },
        "risks": [
            *([{
                "severity": "BLOCKING",
                "risk": "Formal disk space is below the frozen 120 GiB prelaunch floor.",
                "mitigation": "Preserve current successful checkpoints; migrate or explicitly remove superseded archived artifacts, then rerun this script.",
            }] if disk_free_bytes < required_disk else []),
            {
                "severity": "HISTORICAL" if diagnostics_valid else "BLOCKING",
                "risk": "Old Pilot-64 GPU peaks exceeded 98%; that run did not cover long responses or post-warmup steps.",
                "mitigation": "Keep the old evidence; evaluate the separately bound pressure_v2 diagnostics above, without claiming they validate the formal candidate.",
            },
            {
                "severity": "BLOCKING",
                "risk": "The final natural-generation candidate has not been promoted and validated against the diagnostic sources/configuration.",
                "mitigation": "Prepare the candidate, retain normal EOS, validate its actual launch chain and bind candidate evidence before implementing final promotion.",
            },
            {
                "severity": "DISCLOSURE",
                "risk": "Fixed Actor payloads match, but the whole online rollout/ref/logprob workload is not fixed; whole-run causal memory claims remain unsupported.",
                "mitigation": "Retain optimization_validated=false and whole_run_causal_claim_allowed=false; report pressure feasibility separately.",
            },
            {
                "severity": "REVIEW",
                "risk": "Budget still uses historical short-response Pilot-64; cold reload proves historical Student inference, not training resume.",
                "mitigation": "Refresh budget using the final natural-generation candidate and review candidate checkpoint/reload coverage. Do not extrapolate forced-length pressure as natural average throughput.",
            },
            {
                "severity": "CONTROL",
                "risk": "The report-builder process may run in a smaller cgroup than the training launcher.",
                "mitigation": "Use the Pilot runtime capacity evidence here; the guarded launcher must freshly recheck cgroup capacity before formal --run.",
            },
            {
                "severity": "CONTROL",
                "risk": "The cumulative billing observation is historical.",
                "mitigation": "Refresh cumulative charge and UTC timestamp immediately before formal launch.",
            },
        ],
        "next_actions": [
            *(["Free enough disk to meet the 120 GiB formal floor."]
              if disk_free_bytes < required_disk else []),
            "Use the evidence-bound 240 GiB CPU floor; freshly recheck the launcher cgroup. Refresh final static/config/budget source bindings after resource and candidate changes.",
            *(["Repair missing/changed diagnostic evidence before using its coverage."] if not diagnostics_valid else []),
            "Prepare and validate the final natural-generation candidate with deferred loading; preserve historical diagnostic runtimes and do not reuse forced-EOS/replay checkpoints as formal training starts.",
            "Bind candidate source/configuration, checkpoint validation and refreshed budget in the promotion workflow; this diagnostic-only revision intentionally cannot authorize training.",
            "Set configs/vopd_6241.yaml status to ready_for_formal_training only after every prior check passes, then rerun this script.",
        ],
        "sources": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
    }


# Supersede the historical diagnostic-only composer with the source-bound
# natural-candidate Gate. The old implementation remains readable as history.
from scripts.finalize_day11_candidate_gate import build_preflight


def render_markdown(value: dict[str, Any]) -> str:
    checks = "\n".join(
        f"| `{name}` | {'PASS' if passed else 'FAIL'} |"
        for name, passed in value.get("checks", {}).items()
    )
    blockers = "\n".join(f"- `{item}`" for item in value.get("blocking_gates", [])) or "- 无"
    actions = "\n".join(f"{index}. {item}" for index, item in enumerate(value.get("next_actions", []), 1))
    runtime = value.get("runtime_snapshot", {})
    validation = value.get('latest_validation', {})
    pressure = validation.get('pressure', {})
    candidate = value.get('formal_candidate', {})
    budget = value.get('budget', {})
    selected_budget = budget.get('selected', {})
    diagnostic_summary = (
        f"- 证据状态：`{validation.get('status', 'NOT_AVAILABLE')}`\n"
        f"- 固定输入对照：`{validation.get('fixed_comparison_status', 'NOT_AVAILABLE')}`；不允许整段显存因果归因。\n"
        f"- 压力更新：{pressure.get('observed_steps', 0)} 步；有效回复 {pressure.get('response_count', 0)} 条；"
        f"长度 {pressure.get('minimum_response_tokens', 0)}～{pressure.get('maximum_response_tokens', 0)} tokens。\n"
        f"- NVML / 同步物理显存峰值：{pressure.get('gpu_peak_ratio', 0):.2%} / {pressure.get('marker_peak_ratio', 0):.2%}。\n"
        f"- CPU 峰值：{pressure.get('cpu_peak_bytes', 0) / GIB:.2f} GiB。\n"
        f"- 证据错误：`{validation.get('errors', [])}`\n"
        f"- 正式候选：`{candidate.get('status', 'NOT_AVAILABLE')}`；已绑定自然生成与最终 checkpoint。\n"
        f"- 新预算：计划 {selected_budget.get('planning_hours', 0):.2f} 双卡小时 / "
        f"{selected_budget.get('planning_incremental_cost_cny', 0):.2f} 元；"
        "候选 PASS 仍不等于正式配置已放行。"
    )
    return f"""# Vision-OPD 6241 Day11 最终汇总 Gate

- 状态：**{value['status']}**
- 报告生成使用 GPU：`false`
- 正式训练授权：`{str(value.get('formal_training_authorized', False)).lower()}`
- Pilot 运行期 CPU 容量：{runtime.get('pilot_runtime_cpu_capacity_gib', 0):.2f} GiB
- 本报告进程 cgroup（不可作启动证据）：{runtime.get('builder_process_cpu_capacity_gib', 0):.2f} GiB
- 当前磁盘可用：{runtime.get('disk_free_gib', 0):.2f} GiB
- 正式磁盘门槛：{runtime.get('disk_required_gib', 0):.2f} GiB

## 最新诊断证据

{diagnostic_summary}

## 检查

| 检查 | 结果 |
| --- | --- |
{checks}

## 阻塞项

{blockers}

## 下一步

{actions}

此文件是决策快照；资源变化、配置变化或证据文件变化后必须重新生成。
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--disk-path", type=Path, default=ROOT.parent)
    args = parser.parse_args()
    paths = {name: path.resolve() for name, path in PATHS.items()}
    value = build_preflight(
        paths,
        disk_free_bytes=shutil.disk_usage(args.disk_path.resolve()).free,
        cpu_capacity_bytes=memory_capacity_bytes(),
    )
    output = args.output.resolve()
    if output.exists():
        archive = output.parent / 'gate_history' / (
            dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '_' + sha256_file(output)[:12]
        )
        archive.mkdir(parents=True, exist_ok=False)
        for previous in (output, output.with_suffix('.md'), output.with_suffix('.sha256')):
            if previous.is_file():
                shutil.copy2(previous, archive / previous.name)
        value['previous_report_archive'] = str(archive)
    write_atomic(output, json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    write_atomic(output.with_suffix(".md"), render_markdown(value))
    write_atomic(output.with_suffix(".sha256"), f"{sha256_file(output)}  {output}\n{sha256_file(output.with_suffix('.md'))}  {output.with_suffix('.md')}\n")
    print(f"DAY11_FINAL_PREFLIGHT={value['status']}")
    print(f"FORMAL_TRAINING_AUTHORIZED={value.get('formal_training_authorized', False)}")
    print(f"BLOCKING_GATES={json.dumps(value.get('blocking_gates', []))}")
    print(f"OUTPUT={output}")
    return 0 if value.get("artifact_status") == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
