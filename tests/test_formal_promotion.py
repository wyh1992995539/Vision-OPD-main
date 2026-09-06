import copy
import json

import yaml

from scripts import promote_vopd_6241_candidate as promotion
from scripts.run_vopd_6241_guarded import static_preflight
from scripts.vopd_training_preflight import validate_config


def test_promoted_form_is_exact_candidate_plus_release_metadata(tmp_path):
    candidate = yaml.safe_load(promotion.CANDIDATE.read_text())
    value = promotion.promoted_config(candidate, receipt_reference=promotion.relative_or_absolute(promotion.RECEIPT))
    assert value["status"] == "ready_after_day11_gate"
    assert value["rollout"]["ignore_eos"] is False
    assert value["actor"]["defer_optimizer_state_load"] is True
    assert value["resources"]["memory_profile"] == "offload_3way_graph4_deferred_formal_v1"
    assert "candidate" not in value
    assert value["promotion"]["validated_workload"] == "natural_eos_128x16"
    path = tmp_path / "formal.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False))
    checked = validate_config(path, promotion.ROOT)
    assert checked["status"] == "PASS", checked["errors"]
    assert checked["checks"]["formal_candidate_promoted"] is True


def test_real_promotion_receipt_and_final_gate_are_bound():
    assert promotion.verify_receipt()
    receipt = json.loads(promotion.RECEIPT.read_text())
    gate = json.loads(promotion.PRE_PROMOTION_GATE.read_text())
    assert receipt["status"] == "PASS_FORMAL_CONFIG_PROMOTED"
    assert receipt["training_started"] is False
    assert gate["status"] == "PASS"
    assert gate["formal_training_authorized"] is True
    assert gate["blocking_gates"] == []
    assert gate["formal_candidate"]["promotion_receipt_valid"] is True


def test_receipt_rejects_semantic_config_tamper_even_if_target_hash_is_updated(tmp_path):
    receipt_value = copy.deepcopy(json.loads(promotion.RECEIPT.read_text()))
    candidate = yaml.safe_load(promotion.CANDIDATE.read_text())
    receipt_path = tmp_path / "promotion_receipt.json"
    formal_path = tmp_path / "formal.yaml"
    formal = promotion.promoted_config(candidate, receipt_reference=str(receipt_path))
    formal_path.write_text(yaml.safe_dump(formal, sort_keys=False))
    receipt_value["promoted_formal_config"].update(
        path=str(formal_path), sha256=promotion.sha256_file(formal_path)
    )
    receipt_path.write_text(json.dumps(receipt_value))
    assert promotion.verify_receipt(receipt_path, formal_path=formal_path)

    formal["rollout"]["ignore_eos"] = True
    formal_path.write_text(yaml.safe_dump(formal, sort_keys=False))
    receipt_value["promoted_formal_config"]["sha256"] = promotion.sha256_file(formal_path)
    receipt_path.write_text(json.dumps(receipt_value))
    assert not promotion.verify_receipt(receipt_path, formal_path=formal_path)


def test_formal_guarded_launcher_static_preflight_passes_without_gpu():
    value = static_preflight(promotion.FORMAL, promotion.POLICY)
    assert value["status"] == "PASS", {
        name: passed for name, passed in value["checks"].items() if not passed
    }
    assert value["budget_reservation_cny"] == promotion.load_json(
        promotion.CANDIDATE_FREEZE
    )["budget"]["selected"]["reservation_incremental_cost_cny"]
    assert value["disk_required_bytes"] == 120 * 1024**3
