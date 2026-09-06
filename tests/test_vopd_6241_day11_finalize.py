import copy
import json
from pathlib import Path
import tempfile

import pytest
import yaml

from scripts.finalize_day11_preflight import GIB, PATHS, build_preflight, decision_status
from scripts.freeze_vopd_6241_budget import build_budget


ROOT = Path(__file__).resolve().parents[1]


def test_budget_freeze_recomputes_and_preserves_resource_caveats():
    postflight_path = ROOT / "artifacts/runs/E-D11-6K-GATE-001/pilot/64/evidence/postflight.json"
    policy_path = ROOT / "configs/vopd_6241_abort_policy.yaml"
    billing_path = ROOT / "artifacts/runs/E-D11-6K-GATE-001/pilot/billing_observation.json"
    postflight = json.loads(postflight_path.read_text())
    policy = yaml.safe_load(policy_path.read_text())
    billing = json.loads(billing_path.read_text())
    result = build_budget(
        postflight, policy, billing,
        postflight_path=postflight_path,
        policy_path=policy_path,
        billing_path=billing_path,
        generated_at="2026-09-05T00:00:00+00:00",
    )
    assert result["status"] == "PASS_BUDGET_FROZEN_WITH_RESOURCE_CAVEATS"
    assert result["selected_budget"]["reservation_scenario"] == "conservative_max"
    assert result["project_cap"]["launch_value_fresh"] is False
    assert result["coverage"]["post_warmup_steps_observed"] == 0
    assert result["coverage"]["maximum_response_tokens_observed"] == 392
    assert result["coverage"]["gpu_peak_below_abort_ratio"] is False

    corrupt = copy.deepcopy(postflight)
    corrupt["projection_780"]["scenarios"]["mean"]["projected_cost_cny"] += 1
    with pytest.raises(ValueError, match="cost does not match"):
        build_budget(
            corrupt, policy, billing,
            postflight_path=postflight_path,
            policy_path=policy_path,
            billing_path=billing_path,
        )


@pytest.mark.parametrize(
    "evidence,runtime,safety,released,expected",
    [
        ({"e": False}, {"r": True}, {"s": True}, False, "FAIL_EVIDENCE_INTEGRITY"),
        ({"e": True}, {"r": False}, {"s": True}, False, "BLOCKED_RUNTIME_RESOURCES"),
        ({"e": True}, {"r": True}, {"s": False}, False, "BLOCKED_ADDITIONAL_RESOURCE_VALIDATION"),
        ({"e": True}, {"r": True}, {"s": True}, False, "READY_TO_UNBLOCK_FORMAL_CONFIG"),
        ({"e": True}, {"r": True}, {"s": True}, True, "PASS"),
    ],
)
def test_day11_decision_is_fail_closed(evidence, runtime, safety, released, expected):
    assert decision_status(evidence, runtime, safety, released) == expected


def test_repository_aggregate_binds_evidence_and_reports_real_blockers():
    result = build_preflight(
        {name: path.resolve() for name, path in PATHS.items()},
        disk_free_bytes=69 * GIB,
        cpu_capacity_bytes=2 * GIB,
        generated_at="2026-09-05T00:00:00+00:00",
    )
    assert result['evidence_checks']['latest_diagnostic_evidence_integrity']
    assert result['evidence_checks']['candidate_gate_freeze_current']
    assert result['evidence_checks']['candidate_validation_pass_and_bound']
    assert result['evidence_checks']['validated_candidate_source_current']
    assert result['evidence_checks']['candidate_budget_refrozen_and_below_project_cap']
    assert len(result['gate_implementation']) == 7
    assert all(len(digest) == 64 for digest in result['gate_implementation'].values())
    assert result['checks']['formal_cpu_floor_matches_reviewed_240_gib']
    assert result["runtime_checks"]["pilot_runtime_capacity_met_reviewed_240_gib"]
    assert not result["runtime_checks"]["disk_free_meets_refrozen_formal_floor"]
    assert result["runtime_snapshot"]["pilot_runtime_cpu_capacity_gib"] == 240
    assert result["runtime_snapshot"]["builder_process_cpu_capacity_gib"] == 2
    assert result["runtime_snapshot"]["builder_process_capacity_is_launch_evidence"] is False
    assert result['status'] == 'BLOCKED_RUNTIME_RESOURCES'
    assert result["formal_training_authorized"] is False
    assert any('free disk is below' in item['risk'] for item in result['risks'])


def test_expanded_disk_removes_stale_disk_blocker_but_does_not_release_training():
    result = build_preflight(
        {name: path.resolve() for name, path in PATHS.items()},
        disk_free_bytes=422 * GIB,
        cpu_capacity_bytes=2 * GIB,
    )
    assert result['runtime_checks']['disk_free_meets_refrozen_formal_floor']
    assert not any('free disk is below' in item['risk'] for item in result['risks'])
    assert not any('Free enough disk' in item for item in result['next_actions'])
    assert result['status'] == 'PASS'
    assert result['ready_to_unblock_formal_config'] is True
    assert result['blocking_gates'] == []
    assert result['formal_training_authorized'] is True
