import json

import pytest

from scripts import freeze_formal_candidate_resources as freeze


def test_real_candidate_freeze_is_current_and_uses_natural_workload():
    assert freeze.verify()
    value = json.loads(freeze.DEFAULT_OUTPUT.read_text())
    assert value["status"] == "PASS_CANDIDATE_GATE_FREEZE"
    assert value["formal_training_authorized"] is False
    assert value["candidate_validation"]["status"] == "PASS_CANDIDATE_VALIDATION"
    assert value["candidate_validation"]["normal_eos_observed"] is True
    assert value["candidate_validation"]["observed_steps"] == 16
    assert value["candidate_validation"]["post_warmup_steps"] == list(range(11, 17))
    assert value["candidate_validation"]["maximum_response_tokens"] == 471


def test_disk_and_budget_are_recomputed_from_candidate_measurements():
    value = json.loads(freeze.DEFAULT_OUTPUT.read_text())
    disk = value["disk"]
    budget = value["budget"]
    assert disk["checkpoint_payload_bytes"] == 57034957461
    assert disk["checkpoint_apparent_bytes"] == 57034961759
    assert disk["refrozen_prelaunch_required_bytes"] == 120 * freeze.GIB
    assert all(disk["checks"].values())
    assert budget["steady_step_sample_count"] == 15
    assert budget["selected"]["planning_hours"] == pytest.approx(7.043446635950425)
    assert budget["selected"]["planning_incremental_cost_cny"] == pytest.approx(84.23962176596709)
    assert budget["selected"]["reservation_hours"] == pytest.approx(8.074135051401229)
    assert budget["selected"]["reservation_incremental_cost_cny"] == pytest.approx(96.5666552147587)
    assert budget["project_cap"]["launch_value_fresh"] is False


def test_low_disk_blocks_freeze_without_changing_evidence_contract():
    value = freeze.build_freeze(disk_free_bytes=119 * freeze.GIB)
    assert value["status"] == "BLOCKED_CANDIDATE_RESOURCE_GATE"
    assert value["disk"]["status"] == "BLOCKED"
    assert not value["disk"]["checks"]["current_disk_meets_refrozen_floor"]
    assert value["budget"]["status"] == "PASS_BUDGET_REFROZEN_FROM_NATURAL_CANDIDATE"
    assert value["formal_training_authorized"] is False


@pytest.mark.parametrize("field", ["status", "authorization"])
def test_candidate_postflight_self_claims_are_rejected(tmp_path, field):
    report = json.loads(freeze.DEFAULT_POSTFLIGHT.read_text())
    if field == "status":
        report["status"] = "FAIL"
    else:
        report["formal_training_authorized"] = True
    path = tmp_path / "postflight.json"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        freeze.build_freeze(postflight_path=path, disk_free_bytes=200 * freeze.GIB)
