import copy
import json
from pathlib import Path

import pytest
import yaml

from scripts.grpo_training_preflight import ROOT, build_hydra_overrides, validate_config

CONFIG = ROOT / "configs/grpo_6241.yaml"


def load_config():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_grpo_static_preflight_passes_and_is_fail_closed(tmp_path):
    report = validate_config(CONFIG, tmp_path / "preflight.json")
    assert report["status"] == "PASS"
    assert report["static_preflight_passed"] is True
    assert report["training_started"] is False
    assert report["gpu_used"] is False
    assert report["pilot_training_authorized"] is False
    assert report["formal_training_authorized"] is False
    assert report["rows"] == report["unique_sample_ids"] == 6241
    assert report["training_contract"] == {
        "prompt_batch": 8,
        "rollout_n": 4,
        "responses_per_outer_iteration": 32,
        "outer_iterations": 780,
        "effective_prompts": 6240,
        "dropped_prompts": 1,
        "effective_trajectories": 24960,
        "worker_global_mini_batch_after_rollout_expansion": 32,
        "worker_local_mini_batch_two_way_dp": 16,
    }
    saved = json.loads((tmp_path / "preflight.json").read_text())
    assert saved["config_sha256"] == report["config_sha256"]
    resolved = (tmp_path / "resolved_hydra_config.yaml").read_text()
    assert "vision_opd_mcq_grpo.py" in resolved
    assert "adv_estimator: grpo" in resolved


def test_hydra_overrides_select_rule_reward_and_plain_grpo():
    joined = "\n".join(build_hydra_overrides(load_config())).lower()
    assert "custom_reward_function.path=" in joined
    assert "custom_reward_function.name=compute_score" in joined
    assert "algorithm.adv_estimator=grpo" in joined
    assert "algorithm.norm_adv_by_std_in_grpo=true" in joined
    assert "actor_rollout_ref.rollout.n=4" in joined
    assert "actor_rollout_ref.actor.policy_loss.loss_mode=vanilla" in joined
    assert "actor_rollout_ref.actor.use_kl_loss=true" in joined
    assert "algorithm.use_kl_in_reward=false" in joined
    assert "critic.enable=false" in joined
    assert "reward_model.enable=false" in joined
    assert not any(token in joined for token in ("bbox_images", "cached_prefix", "self_distillation"))


def test_invalid_rollout_group_contract_fails(tmp_path):
    config = copy.deepcopy(load_config())
    config["rollout"]["n"] = 1
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="grpo_group_contract"):
        validate_config(path)
