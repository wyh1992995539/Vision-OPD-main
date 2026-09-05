#!/usr/bin/env python3
"""Assemble the CPU-only Day 11 gate without claiming that the GPU Pilot passed."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys

import yaml
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.vopd_training_preflight import validate_config
from scripts.run_vopd_6241_pilot_guarded import load_pilot_policy


RUN_ROOT = ROOT / "artifacts/runs/E-D11-6K-GATE-001"
FORMAL_CONFIG = ROOT / "configs/vopd_6241.yaml"
ABORT_POLICY = ROOT / "configs/vopd_6241_abort_policy.yaml"
PROJECT_CONFIG = ROOT / "configs/project_6241.yaml"
PROMPT = RUN_ROOT / "prompt_length/prompt_length_summary.json"
PROMPT_RESOURCE_AMENDMENT = RUN_ROOT / "prompt_length/resource_only_config_amendment.json"
OVERLAP = ROOT / "artifacts/runs/E-D10-6K-DATA-001/overlap/overlap_validation.json"
DROP_LAST = RUN_ROOT / "drop_last/drop_last_audit.json"
CACHED = RUN_ROOT / "cached_prefix/report.json"
FREEZE = RUN_ROOT / "training_config_freeze.json"
PILOTS = {
    "16": ROOT / "configs/vopd_6241_pilot_16.yaml",
    "64": ROOT / "configs/vopd_6241_pilot_64.yaml",
}
PILOT_POLICY = ROOT / "configs/vopd_6241_pilot_abort_policy.yaml"
PILOT_LAUNCHER = ROOT / "scripts/run_vopd_6241_pilot_guarded.py"
PILOT_POSTFLIGHT = ROOT / "scripts/audit_vopd_6241_pilot.py"
RUNTIME_GUARD = ROOT / "scripts/monitor_vopd_training.py"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def referenced_file_matches(entry: dict[str, Any]) -> bool:
    path = Path(str(entry.get("path", "")))
    return path.is_file() and entry.get("sha256") == sha256_file(path)


RESOURCE_ONLY_PATHS = {
    "actor.parameter_offload", "actor.optimizer_offload",
    "actor.reference_parameter_offload", "rollout.gpu_memory_utilization",
    "rollout.engine_kwargs.vllm.compilation_config.cudagraph_capture_sizes",
    "resources.memory_profile", "resources.actor_parameter_offload_required",
    "resources.optimizer_offload_required", "resources.reference_parameter_offload_required",
    "resources.prelaunch_cgroup_minimum_bytes",
}


def semantic_changes(before: dict, after: dict, prefix: str = "") -> list[dict]:
    changes = []
    for key in sorted(before.keys() | after.keys()):
        path = f"{prefix}.{key}" if prefix else key
        old, new = before.get(key), after.get(key)
        if isinstance(old, dict) and isinstance(new, dict):
            changes.extend(semantic_changes(old, new, path))
        elif old != new:
            changes.append({"path": path, "before": old, "after": new, "prompt_affecting": path not in RESOURCE_ONLY_PATHS})
    return changes


def resource_changes_match(amendment: dict, current: dict, audited_sha: str) -> bool:
    entry = amendment.get("audited_config", {})
    path = Path(str(entry.get("path", "")))
    if not path.is_file() or entry.get("sha256") != audited_sha or sha256_file(path) != audited_sha:
        return False
    actual = semantic_changes(yaml.safe_load(path.read_text(encoding="utf-8")), current)
    return (bool(actual) and actual == amendment.get("semantic_changes")
            and all(not item["prompt_affecting"] for item in actual))


def compact_preflight(summary: dict[str, Any]) -> dict[str, Any]:
    compact = dict(summary)
    sample_ids = [str(item) for item in compact.pop("sample_ids", [])]
    compact["sample_id_count"] = len(sample_ids)
    compact["sample_ids_sha256"] = hashlib.sha256(
        ("\n".join(sample_ids) + "\n").encode("utf-8")
    ).hexdigest()
    return compact


def build_gate() -> dict[str, Any]:
    required = [
        FORMAL_CONFIG, ABORT_POLICY, PROJECT_CONFIG, PROMPT, PROMPT_RESOURCE_AMENDMENT, OVERLAP,
        DROP_LAST, CACHED, FREEZE, PILOT_POLICY, PILOT_LAUNCHER,
        PILOT_POSTFLIGHT, RUNTIME_GUARD, *PILOTS.values(),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        return {
            "schema_version": 1,
            "generated_at_utc": utc_now(),
            "status": "FAIL",
            "gpu_used": False,
            "missing_inputs": missing,
            "checks": {},
        }

    prompt = read_json(PROMPT)
    prompt_resource_amendment = read_json(PROMPT_RESOURCE_AMENDMENT)
    overlap = read_json(OVERLAP)
    drop_last = read_json(DROP_LAST)
    cached = read_json(CACHED)
    freeze = read_json(FREEZE)
    pilot_preflights = {
        stage: validate_config(path, ROOT) for stage, path in PILOTS.items()
    }
    formal_preflight = validate_config(FORMAL_CONFIG, ROOT)
    pilot_policies = {
        stage: load_pilot_policy(PILOT_POLICY, stage) for stage in ("16", "64")
    }

    pilot_16_ids = pilot_preflights["16"].get("sample_ids", [])
    pilot_64_ids = pilot_preflights["64"].get("sample_ids", [])
    cached_output = Path(str(cached.get("output_parquet", "")))
    formal_expected_block = formal_preflight.get("errors") == [
        "frozen config checks failed: ['config_not_explicitly_blocked']"
    ]
    formal_config_data = yaml.safe_load(FORMAL_CONFIG.read_text(encoding="utf-8"))
    abort_policy_data = yaml.safe_load(ABORT_POLICY.read_text(encoding="utf-8"))
    pilot_policy_data = yaml.safe_load(PILOT_POLICY.read_text(encoding="utf-8"))
    pilot_config_data = {
        stage: yaml.safe_load(path.read_text(encoding="utf-8"))
        for stage, path in PILOTS.items()
    }
    amendment_prior = prompt_resource_amendment.get("prior_audit", {})
    amendment_revised = prompt_resource_amendment.get("revised_config", {})
    amendment_contract = prompt_resource_amendment.get("prompt_contract", {})
    amendment_changes = prompt_resource_amendment.get("semantic_changes", [])
    amendment_summary_path = Path(str(amendment_prior.get("summary_path", "")))
    amendment_records_path = Path(str(amendment_prior.get("records_path", "")))
    current_train = Path(formal_config_data["paths"]["train_file"])
    if not current_train.is_absolute():
        current_train = ROOT / current_train
    current_template = Path(formal_config_data["paths"]["chat_template"])
    if not current_template.is_absolute():
        current_template = ROOT / current_template
    prompt_resource_amendment_valid = (
        prompt_resource_amendment.get("status") == "PASS_RESOURCE_ONLY_CHANGE"
        and amendment_prior.get("audited_config_sha256")
        == prompt.get("input", {}).get("config_sha256")
        and amendment_revised.get("sha256") == sha256_file(FORMAL_CONFIG)
        and amendment_summary_path.resolve() == PROMPT.resolve()
        and amendment_summary_path.is_file()
        and amendment_prior.get("summary_sha256") == sha256_file(amendment_summary_path)
        and amendment_records_path.is_file()
        and amendment_prior.get("records_sha256") == sha256_file(amendment_records_path)
        and resource_changes_match(
            prompt_resource_amendment, formal_config_data,
            prompt.get("input", {}).get("config_sha256"),
        )
        and all(item.get("prompt_affecting") is False for item in amendment_changes)
        and amendment_contract.get("model_path") == formal_config_data["paths"]["model"]
        and amendment_contract.get("train_file") == str(current_train.resolve())
        and amendment_contract.get("train_file_sha256") == sha256_file(current_train)
        and amendment_contract.get("chat_template") == formal_config_data["paths"]["chat_template"]
        and amendment_contract.get("chat_template_sha256") == sha256_file(current_template)
        and amendment_contract.get("image_key") == formal_config_data["data"]["image_key"]
        and amendment_contract.get("teacher_image_key")
        == formal_config_data["data"]["teacher_image_key"]
        and amendment_contract.get("max_prompt_length")
        == formal_config_data["data"]["max_prompt_length"]
        and amendment_contract.get("truncation") == formal_config_data["data"]["truncation"]
        and prompt_resource_amendment.get("verification", {}).get(
            "reconstructed_prior_config_sha256_matches_audit"
        ) is True
        and prompt_resource_amendment.get("verification", {}).get(
            "prompt_affecting_change_count"
        ) == 0
    )
    prompt_config_binding_valid = (
        prompt.get("input", {}).get("config_sha256") == sha256_file(FORMAL_CONFIG)
        or prompt_resource_amendment_valid
    )
    checks = {
        "prompt_length_pass": (
            prompt.get("status") == "PASS"
            and prompt.get("row_count") == 6241
            and prompt.get("error_count") == 0
            and all(view.get("overlength_count") == 0 for view in prompt.get("views", {}).values())
            and prompt_config_binding_valid
        ),
        "prompt_resource_only_amendment_valid": prompt_config_binding_valid,
        "overlap_audit_complete_with_disclosure": (
            overlap.get("status") == "PASS_WITH_CONFIRMED_OVERLAP"
            and all(overlap.get("checks", {}).values())
            and overlap.get("result", {}).get("unresolved_pairs") == 0
            and overlap.get("interpretation", {}).get("training_action") == "retain_all_train_rows"
        ),
        "drop_last_pass": (
            drop_last.get("status") == "PASS"
            and all(drop_last.get("checks", {}).values())
            and referenced_file_matches(drop_last.get("inputs", {}).get("project_config", {}))
            and referenced_file_matches(drop_last.get("inputs", {}).get("training_config", {}))
            and referenced_file_matches(drop_last.get("inputs", {}).get("abort_policy", {}))
            and referenced_file_matches(drop_last.get("inputs", {}).get("train_parquet", {}))
        ),
        "cached_prefix_pass": (
            cached.get("status") == "PASS"
            and cached.get("actual_records") == cached.get("expected_records") == 6241
            and cached.get("unique_sample_ids") == 6241
            and cached.get("duplicate_sample_ids") == 0
            and cached.get("inference_errors") == 0
            and cached.get("empty_responses") == 0
            and cached.get("empty_token_ids") == 0
            and cached.get("overlength_token_ids") == 0
            and cached_output.is_file()
            and cached.get("output_sha256") == sha256_file(cached_output)
        ),
        "training_config_static_freeze_current": (
            freeze.get("status") == "STATIC_FROZEN_PENDING_DAY11_PILOT"
            and freeze.get("hashes", {}).get("vopd_config") == sha256_file(FORMAL_CONFIG)
            and freeze.get("hashes", {}).get("abort_policy") == sha256_file(ABORT_POLICY)
            and freeze.get("hashes", {}).get("cached_prefix_input_parquet")
            == prompt.get("input", {}).get("train_file_sha256")
        ),
        "formal_config_fail_closed_only_on_day11_status": formal_expected_block,
        "pilot_16_static_preflight_pass": pilot_preflights["16"].get("status") == "PASS",
        "pilot_64_static_preflight_pass": pilot_preflights["64"].get("status") == "PASS",
        "pilot_16_has_two_steps": (
            pilot_preflights["16"].get("train_rows") == 16
            and pilot_preflights["16"].get("training_contract", {}).get("total_optimizer_steps") == 2
        ),
        "pilot_64_has_eight_steps": (
            pilot_preflights["64"].get("train_rows") == 64
            and pilot_preflights["64"].get("training_contract", {}).get("total_optimizer_steps") == 8
        ),
        "pilot_16_is_prefix_of_pilot_64": pilot_16_ids == pilot_64_ids[:16],
        "pilot_ids_unique": (
            len(set(pilot_16_ids)) == 16 and len(set(pilot_64_ids)) == 64
        ),
        "warmup_aware_student_update_contract": (
            int(formal_config_data["actor"]["lr_warmup_steps"]) == 10
            and all(
                int(config["actor"]["lr_warmup_steps"]) == 10
                for config in pilot_config_data.values()
            )
            and abort_policy_data["metrics"].get(
                "student_update_required_only_when_lr_positive"
            ) is True
            and pilot_policy_data["metrics"].get(
                "student_update_required_only_when_lr_positive"
            ) is True
            and freeze.get("warmup_aware_audit_contract", {}).get(
                "pilot_requires_positive_lr_step"
            ) is True
            and freeze.get("warmup_aware_audit_contract", {}).get(
                "teacher_contract_relaxed_during_warmup"
            ) is False
            and freeze.get("hashes", {}).get("pilot_abort_policy")
            == sha256_file(PILOT_POLICY)
            and freeze.get("hashes", {}).get("runtime_guard_script")
            == sha256_file(RUNTIME_GUARD)
            and freeze.get("hashes", {}).get("pilot_postflight_script")
            == sha256_file(PILOT_POSTFLIGHT)
        ),
        "three_way_offload_resource_contract": (
            formal_config_data["resources"].get("memory_profile") == "offload_3way_graph4_v1"
            and formal_preflight["checks"].get("three_way_offload_memory_profile") is True
            and all(
                config["actor"] == formal_config_data["actor"]
                and config["rollout"] == formal_config_data["rollout"]
                and config["resources"] == formal_config_data["resources"]
                for config in pilot_config_data.values()
            )
            and abort_policy_data["memory"]["prelaunch_cgroup_minimum_bytes"]
            == pilot_policy_data["memory"]["prelaunch_cgroup_minimum_bytes"]
            == formal_config_data["resources"]["prelaunch_cgroup_minimum_bytes"]
            == 192 * 1024**3
        ),
        "pilot_guard_stage_contracts_valid": (
            pilot_policies["16"][1]["expected_optimizer_steps"] == 2
            and pilot_policies["64"][1]["expected_optimizer_steps"] == 8
            and pilot_policies["16"][1]["prerequisite_postflight"] is None
            and bool(pilot_policies["64"][1]["prerequisite_postflight"])
            and pilot_policies["64"][1]["require_cold_reload"] is True
        ),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    status = "PASS_PENDING_GPU_PILOT" if not failed else "FAIL"
    inputs = {
        "formal_config": FORMAL_CONFIG,
        "abort_policy": ABORT_POLICY,
        "project_config": PROJECT_CONFIG,
        "prompt_length": PROMPT,
        "prompt_resource_amendment": PROMPT_RESOURCE_AMENDMENT,
        "prompt_audited_config": RUN_ROOT / "prompt_length/audited_training_config.yaml",
        "overlap": OVERLAP,
        "drop_last": DROP_LAST,
        "cached_prefix": CACHED,
        "training_config_freeze": FREEZE,
        "pilot_16_config": PILOTS["16"],
        "pilot_64_config": PILOTS["64"],
        "pilot_abort_policy": PILOT_POLICY,
        "pilot_guarded_launcher": PILOT_LAUNCHER,
        "pilot_postflight": PILOT_POSTFLIGHT,
        "runtime_guard": RUNTIME_GUARD,
        "training_launcher": ROOT / "scripts/run_vopd_2gpu.sh",
        "training_preflight": ROOT / "scripts/vopd_training_preflight.py",
        "vllm_adapter": ROOT / "verl/workers/rollout/vllm_rollout/vllm_async_server.py",
        "static_gate_builder": Path(__file__).resolve(),
    }
    return {
        "schema_version": 1,
        "gate_id": "E-D11-6K-STATIC-GATE-001",
        "generated_at_utc": utc_now(),
        "status": status,
        "gpu_used": False,
        "ready_for_gpu_pilot": not failed,
        "formal_training_authorized": False,
        "final_gate_written": False,
        "final_gate_path": str(RUN_ROOT / "preflight.json"),
        "checks": checks,
        "failed_checks": failed,
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in inputs.items()
        },
        "pilot_preflights": {
            stage: compact_preflight(summary)
            for stage, summary in pilot_preflights.items()
        },
        "formal_preflight": compact_preflight(formal_preflight),
        "overlap_disclosure": {
            "confirmed_pairs": overlap.get("result", {}).get("confirmed_overlap_pairs"),
            "unresolved_pairs": overlap.get("result", {}).get("unresolved_pairs"),
            "human_project_owner_signoff": overlap.get("interpretation", {}).get(
                "human_project_owner_signoff"
            ),
            "required_evaluation_action": overlap.get("interpretation", {}).get(
                "evaluation_action"
            ),
        },
        "runtime_gates": {
            "pilot_16": "PENDING_GPU",
            "pilot_64": "BLOCKED_UNTIL_PILOT_16_PASS",
            "pilot_budget_reestimate": "PENDING_GPU_MEASUREMENTS",
            "day11_final_preflight": "NOT_WRITTEN",
        },
        "next_command": (
            "conda run --no-capture-output -n vision-opd python "
            "scripts/run_vopd_6241_pilot_guarded.py --stage 16 --preflight-only"
        ),
    }


def markdown(gate: dict[str, Any]) -> str:
    rows = "\n".join(
        f"| `{name}` | {'PASS' if passed else 'FAIL'} |"
        for name, passed in gate.get("checks", {}).items()
    )
    return f"""# Day 11 6K 静态 Gate

- 状态：**{gate['status']}**
- GPU 使用：`false`
- 可进入 GPU Pilot：`{str(gate.get('ready_for_gpu_pilot', False)).lower()}`
- 正式训练授权：`false`
- 最终 `preflight.json`：未生成；只有运行时 Pilot 与预算复算通过后才能生成。

## 静态检查

| 检查 | 结果 |
| --- | --- |
{rows}

## 尚未完成

- 16 条、2 steps、1024-token 双卡真实训练 Pilot。
- 16 条通过后执行 64 条、8 steps 稳定性 Pilot。
- 使用实测吞吐重算 780 steps 墙钟和费用。
- 冻结最终配置哈希并生成 `artifacts/runs/E-D11-6K-GATE-001/preflight.json`。

## 下一条安全命令

```bash
{gate.get('next_command', '')}
```
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RUN_ROOT / "static_gate.json")
    parser.add_argument("--markdown", type=Path, default=RUN_ROOT / "static_gate.md")
    parser.add_argument("--sha256", type=Path, default=RUN_ROOT / "static_gate_sha256.txt")
    args = parser.parse_args()
    gate = build_gate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(gate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.markdown.write_text(markdown(gate), encoding="utf-8")
    args.sha256.write_text(
        f"{sha256_file(args.output)}  {args.output.resolve()}\n"
        f"{sha256_file(args.markdown)}  {args.markdown.resolve()}\n",
        encoding="utf-8",
    )
    print(f"DAY11_STATIC_GATE={gate['status']}")
    print(f"READY_FOR_GPU_PILOT={gate.get('ready_for_gpu_pilot', False)}")
    print(f"FORMAL_TRAINING_AUTHORIZED={gate.get('formal_training_authorized', False)}")
    print(f"OUTPUT={args.output.resolve()}")
    return 0 if gate["status"] == "PASS_PENDING_GPU_PILOT" else 1


if __name__ == "__main__":
    raise SystemExit(main())
