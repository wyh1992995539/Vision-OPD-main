#!/usr/bin/env python3
"""Candidate-aware Day 11 formal Gate composition.

The aggregate remains fail-closed: a validated candidate can make the project
ready for a separate promotion step, but this module never promotes or launches
the formal configuration itself.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from scripts.day11_validation_evidence import collect_validation_evidence
from scripts.freeze_formal_candidate_resources import verify as verify_candidate_freeze
from scripts.freeze_formal_cpu import verify as verify_cpu_freeze
from scripts.promote_vopd_6241_candidate import verify_receipt as verify_promotion_receipt

GIB = 1024**3


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


def source_entry_matches(entry: dict[str, Any]) -> bool:
    path = Path(str(entry.get("path", "")))
    return path.is_file() and entry.get("sha256") == sha256_file(path)


def decision_status(evidence: dict[str, bool], runtime: dict[str, bool],
                    safety: dict[str, bool], released: bool) -> str:
    if not all(evidence.values()):
        return "FAIL_EVIDENCE_INTEGRITY"
    if not all(runtime.values()):
        return "BLOCKED_RUNTIME_RESOURCES"
    if not all(safety.values()):
        return "BLOCKED_ADDITIONAL_RESOURCE_VALIDATION"
    if not released:
        return "READY_TO_UNBLOCK_FORMAL_CONFIG"
    return "PASS"


def build_preflight(
    paths: dict[str, Path], *, disk_free_bytes: int, cpu_capacity_bytes: int,
    generated_at: str | None = None,
) -> dict[str, Any]:
    missing = [
        str(path) for name, path in paths.items()
        if name != "promotion_receipt" and not path.is_file()
    ]
    if missing:
        return {
            "schema_version": 1,
            "generated_at_utc": generated_at or utc_now(),
            "status": "FAIL_MISSING_INPUTS",
            "artifact_status": "COMPLETE",
            "formal_training_authorized": False,
            "missing_inputs": missing,
        }

    static = load_json(paths["static_gate"])
    historical_freeze = load_json(paths["freeze"])
    historical_budget = load_json(paths["budget"])
    p16 = load_json(paths["pilot_16"])
    p64 = load_json(paths["pilot_64"])
    reload = load_json(paths["cold_reload"])
    pilot_resource = load_json(paths["pilot_64_resource"])
    candidate_freeze = load_json(paths["candidate_gate_freeze"])
    promotion_path = paths.get("promotion_receipt")
    promotion = (
        load_json(promotion_path)
        if promotion_path is not None and promotion_path.is_file()
        else None
    )
    candidate = yaml.safe_load(paths["formal_candidate_config"].read_text(encoding="utf-8"))
    formal = yaml.safe_load(paths["formal_config"].read_text(encoding="utf-8"))
    policy = yaml.safe_load(paths["formal_policy"].read_text(encoding="utf-8"))
    project = yaml.safe_load(paths["project_config"].read_text(encoding="utf-8"))

    validation = collect_validation_evidence(paths)
    diagnostics_valid = validation.get("status") == "PASS_DIAGNOSTIC_EVIDENCE"
    pressure = validation.get("pressure", {})
    candidate_freeze_valid = verify_candidate_freeze(paths["candidate_gate_freeze"])
    promotion_valid = bool(
        promotion_path is not None
        and promotion_path.is_file()
        and verify_promotion_receipt(
            promotion_path,
            formal_path=paths["formal_config"],
            candidate_path=paths["formal_candidate_config"],
        )
    )
    cpu_floor_evidence_valid = (
        verify_cpu_freeze() if promotion is None else promotion_valid
        and promotion.get("checks", {}).get("cpu_freeze_verified_before_promotion") is True
    )

    freeze_p16 = historical_freeze.get("verification", {}).get("pilot_16_postflight", {})
    freeze_p64 = historical_freeze.get("verification", {}).get("pilot_64_postflight", {})
    freeze_reload = historical_freeze.get("verification", {}).get("pilot_64_cold_reload", {})

    evidence_checks = {
        "latest_diagnostic_evidence_integrity": diagnostics_valid,
        "historical_static_gate_preserved": (
            static.get("status") == "PASS_PENDING_GPU_PILOT" and bool(static.get("checks"))
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
        "candidate_gate_freeze_current": candidate_freeze_valid,
        "candidate_validation_pass_and_bound": (
            candidate_freeze_valid
            and candidate_freeze.get("status") == "PASS_CANDIDATE_GATE_FREEZE"
            and candidate_freeze.get("candidate_validation", {}).get("status")
            == "PASS_CANDIDATE_VALIDATION"
            and candidate_freeze.get("candidate_validation", {}).get("training_gate_pass") is True
            and candidate_freeze.get("candidate_validation", {}).get("validation_gate_pass") is True
            and candidate_freeze.get("candidate_validation", {}).get("checkpoint_pass") is True
        ),
        "validated_candidate_source_current": (
            candidate_freeze_valid
            and candidate_freeze.get("candidate_validation", {}).get("validated_candidate_sha256")
            == sha256_file(paths["formal_candidate_config"])
            and source_entry_matches(candidate_freeze.get("sources", {}).get("formal_candidate_config", {}))
        ),
        "candidate_budget_refrozen_and_below_project_cap": (
            candidate_freeze_valid
            and candidate_freeze.get("budget", {}).get("status")
            == "PASS_BUDGET_REFROZEN_FROM_NATURAL_CANDIDATE"
            and candidate_freeze.get("budget", {}).get("project_cap", {}).get("budget_pass") is True
        ),
        "formal_data_and_drop_last_contract": (
            int(candidate["data"]["expected_train_rows"]) == 6241
            and int(candidate["training"]["expected_samples"]) == 6240
            and int(candidate["training"]["dropped_rows"]) == 1
            and candidate["data"]["tail_policy"] == "native_drop_last"
            and project["training_contract"]["tail_policy"] == "native_drop_last"
        ),
        "pilot_resource_summary_bound_to_training_run": (
            pilot_resource.get("training_gate_pass") is True
            and int(pilot_resource.get("observed_steps", 0)) == 8
            and int(pilot_resource.get("cpu_capacity_bytes", 0)) == 240 * GIB
            and len(pilot_resource.get("gpu_peaks", [])) == 2
        ),
    }

    required_disk = int(candidate_freeze["disk"]["refrozen_prelaunch_required_bytes"])
    reviewed_cpu_floor = 240 * GIB
    formal_cpu_floor = int(policy["memory"]["prelaunch_cgroup_minimum_bytes"])
    pilot_cpu_capacity = int(pilot_resource["cpu_capacity_bytes"])
    runtime_checks = {
        "pilot_runtime_capacity_met_reviewed_240_gib": pilot_cpu_capacity >= reviewed_cpu_floor,
        "disk_free_meets_refrozen_formal_floor": int(disk_free_bytes) >= required_disk,
    }

    historical_coverage = historical_budget["coverage"]
    response_stress_floor = int(0.75 * int(historical_coverage["configured_response_limit_tokens"]))
    safety_checks = {
        "formal_cpu_floor_matches_reviewed_240_gib": (
            cpu_floor_evidence_valid and formal_cpu_floor == reviewed_cpu_floor
            and candidate["resources"]["prelaunch_cgroup_minimum_bytes"] == formal_cpu_floor
            and formal["resources"]["prelaunch_cgroup_minimum_bytes"] == formal_cpu_floor
        ),
        "diagnostic_gpu_peaks_below_formal_abort_line": (
            diagnostics_valid and pressure["gpu_peak_ratio"] < float(policy["memory"]["gpu_used_ratio_abort"])
            and pressure["marker_peak_ratio"] < float(policy["memory"]["gpu_used_ratio_abort"])
        ),
        "diagnostic_cpu_peak_below_formal_abort_line": (
            diagnostics_valid and pressure["cpu_peak_ratio"] < float(policy["memory"]["cgroup_used_ratio_abort"])
        ),
        "at_least_two_post_warmup_steps_observed": (
            diagnostics_valid and pressure["coverage"]["passed"] is True
        ),
        "long_response_training_pressure_observed": (
            diagnostics_valid and pressure["coverage"]["passed"] is True
            and pressure["maximum_response_tokens"] >= response_stress_floor
        ),
        "candidate_length_and_warmup_match_formal": (
            candidate["data"]["max_response_length"] == 1024
            and candidate["actor"]["lr_warmup_steps"] == 10
        ),
        "formal_candidate_uses_natural_eos": candidate["rollout"]["ignore_eos"] is False,
        "candidate_gpu_peak_below_formal_abort_line": (
            candidate_freeze_valid
            and candidate_freeze["candidate_validation"]["gpu_peak_ratio"]
            < float(policy["memory"]["gpu_used_ratio_abort"])
        ),
        "candidate_cpu_peak_below_formal_abort_line": (
            candidate_freeze_valid
            and candidate_freeze["candidate_validation"]["cpu_peak_ratio"]
            < float(policy["memory"]["cgroup_used_ratio_abort"])
        ),
        "candidate_checkpoint_validation_bound": (
            candidate_freeze_valid
            and candidate_freeze["candidate_validation"]["checkpoint_pass"] is True
        ),
        "formal_candidate_validation_bound": candidate_freeze_valid,
    }

    # Only a source-bound receipt can release; changing a YAML status cannot.
    formal_config_released = promotion_valid
    status = decision_status(evidence_checks, runtime_checks, safety_checks, formal_config_released)
    all_checks = {**evidence_checks, **runtime_checks, **safety_checks,
                  "formal_config_released": formal_config_released}
    blockers = sorted(name for name, passed in all_checks.items() if not passed)

    historical_sources = historical_budget.get("sources", {})
    return {
        "schema_version": 1,
        "gate_id": "E-D11-6K-FINAL-PREFLIGHT-001",
        "audit_revision": "candidate_validation_budget_disk_v3",
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
        "latest_validation": validation,
        "gate_implementation": {
            str(path.resolve()): sha256_file(path)
            for path in (
                Path(__file__),
                Path(__file__).with_name("finalize_day11_preflight.py"),
                Path(__file__).with_name("day11_validation_evidence.py"),
                Path(__file__).with_name("freeze_formal_candidate_resources.py"),
                Path(__file__).with_name("freeze_formal_cpu.py"),
                Path(__file__).with_name("promote_vopd_6241_candidate.py"),
                Path(__file__).with_name("run_vopd_6241_guarded.py"),
            )
        },

        "formal_candidate": {
            "status": candidate_freeze["candidate_validation"]["status"],
            "reason": "The 128x16 natural-generation candidate and checkpoint are source-bound.",
            "promotion_receipt": str(promotion_path) if promotion_path is not None else None,
            "promotion_receipt_valid": promotion_valid,
            "candidate_gate_freeze": str(paths["candidate_gate_freeze"]),
            "candidate_gate_freeze_sha256": sha256_file(paths["candidate_gate_freeze"]),
            "automatic_promotion_supported": False,
            "training_resume_validated": False,
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
            "disk_headroom_bytes": disk_free_bytes - required_disk,
            "disk_headroom_gib": (disk_free_bytes - required_disk) / GIB,
        },
        "pilot_coverage": {
            **historical_coverage,
            "required_post_warmup_steps": 2,
            "long_response_pressure_floor_tokens": response_stress_floor,
            "note": "Historical Pilot coverage is retained; pressure and natural-candidate evidence independently close its gaps.",
        },
        "budget": candidate_freeze["budget"],
        "disk": candidate_freeze["disk"],
        "historical_freezes": {
            "static_gate_sources_current": all(source_entry_matches(entry)
                                                for entry in static.get("inputs", {}).values()),
            "training_freeze_matches_old_formal_config_and_policy": (
                historical_freeze.get("hashes", {}).get("vopd_config") == sha256_file(paths["formal_config"])
                and historical_freeze.get("hashes", {}).get("abort_policy") == sha256_file(paths["formal_policy"])
            ),
            "pilot64_budget_status": historical_budget.get("status"),
            "pilot64_budget_sources_current": all(source_entry_matches(entry)
                                                      for entry in historical_sources.values()),
            "note": "Historical freezes remain auditable but the natural-candidate freeze supersedes them as the release budget/disk input.",
        },
        "risks": [
            *([{
                "severity": "BLOCKING",
                "risk": "Current free disk is below the refrozen formal floor.",
                "mitigation": "Preserve the candidate receipt; migrate or explicitly retire superseded artifacts, then rebuild this Gate.",
            }] if disk_free_bytes < required_disk else []),
            {
                "severity": "HISTORICAL",
                "risk": "Old Pilot-64 exceeded 98% in isolated samples and lacked post-warmup coverage.",
                "mitigation": "Retain it as history; use the separately bound pressure and natural-candidate results for the current decision.",
            },
            *([{
                "severity": "CONTROL",
                "risk": "The validated candidate has not yet been copied into and hash-bound as the formal launch config.",
                "mitigation": "Use a dedicated promotion receipt; never release training by changing only a status string.",
            }] if not promotion_valid else []),
            {
                "severity": "REVIEW",
                "risk": "The 340 CNY value predates the candidate run; its checkpoint was validated but not cold-reloaded.",
                "mitigation": "Refresh billing immediately before launch. Formal training starts from base weights, so candidate reload is non-blocking unless resume is requested.",
            },
            {
                "severity": "CONTROL",
                "risk": "The report-builder cgroup and disk snapshot are not future launch evidence.",
                "mitigation": "The guarded launcher must freshly recheck 240 GiB CPU, 120 GiB disk, two free GPUs and billing.",
            },
        ],
        "next_actions": [
            *(["Free enough disk to meet the refrozen 120 GiB floor."]
              if disk_free_bytes < required_disk else []),
            *(["Promote the source-bound candidate into configs/vopd_6241.yaml with a dedicated immutable receipt."]
              if not promotion_valid else []),
            *(["Bind the promoted config/policy hashes and this candidate Gate receipt in the formal launcher."]
              if not promotion_valid else []),
            "Refresh cumulative AutoDL billing and live resources immediately before formal --run.",
        ],
        "sources": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in paths.items() if path.is_file()
        },
        "formal_config_observed_status": formal.get("status"),
    }
