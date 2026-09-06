"""CPU-only contract tests for the 128-row formal-candidate validation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from scripts.vopd_training_preflight import validate_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/vopd_6241_candidate_128.yaml"
FORMAL_CANDIDATE = ROOT / "configs/vopd_6241_candidate.yaml"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mapped_argv(config: Path) -> dict[str, str]:
    launcher = (ROOT / "scripts/run_vopd_2gpu.sh").read_text()
    mapping = launcher.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    assignments = subprocess.check_output(
        [sys.executable, "-c", mapping, str(config), str(ROOT)], text=True
    )
    command = (
        "python -m verl.trainer.main_ppo"
        + launcher.split("python -m verl.trainer.main_ppo", 1)[1]
    ).split(" 2>&1 | tee", 1)[0]
    script = (
        "set -eu\n"
        + assignments
        + "\nMAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))\n"
        + 'EXTRA_ARGS=()\npython() { printf "%s\\n" "$@"; }\n'
        + command
    )
    args = subprocess.check_output(["bash", "-c", script], text=True).splitlines()
    return dict(arg.lstrip("+").split("=", 1) for arg in args if "=" in arg)


def test_candidate_validation_static_preflight_passes():
    result = validate_config(CONFIG, ROOT)
    assert result["status"] == "PASS", result["errors"]
    assert result["train_rows"] == 128
    assert result["training_contract"]["total_optimizer_steps"] == 16
    assert result["training_contract"]["require_full_epoch"] is True
    assert result["training_contract"]["dropped_rows"] == 0
    assert result["checks"]["candidate_validation_execution_contract"]
    assert result["checks"]["paper_rollout_sampling_is_frozen"]


def test_candidate_validation_selection_is_hash_bound_and_tail_aware():
    config = yaml.safe_load(CONFIG.read_text())
    manifest_path = ROOT / config["paths"]["selection_manifest"]
    train_path = ROOT / config["paths"]["train_file"]
    manifest = json.loads(manifest_path.read_text())
    ids = [item["sample_id"] for item in manifest["samples"]]

    assert manifest["experiment_id"] == config["experiment"]["id"]
    assert manifest["selection"]["selected_rows"] == 128
    assert len(ids) == len(set(ids)) == 128
    assert manifest["output"]["sha256"] == sha256_file(train_path)
    assert manifest["output"]["sha256"] == (
        "087aecf9fd0d83883dc8b7818733542d0c319e6bad0aaa2fbb8defdb573f644d"
    )
    assert max(item["student_total_tokens"] for item in manifest["samples"]) >= 7880
    assert max(item["teacher_total_tokens"] for item in manifest["samples"]) >= 2809


def test_candidate_validation_keeps_formal_algorithm_and_normal_eos():
    validation = yaml.safe_load(CONFIG.read_text())
    candidate = yaml.safe_load(FORMAL_CANDIDATE.read_text())

    assert validation["actor"] | {"memory_profile_dir": None} == (
        candidate["actor"] | {"memory_profile_dir": None}
    )
    assert validation["rollout"] == candidate["rollout"]
    assert validation["self_distillation"] == candidate["self_distillation"]
    assert validation["rollout"]["ignore_eos"] is False
    assert "diagnostic_generation" not in validation
    assert validation["validation"]["forced_min_response_tokens"] is None
    assert validation["validation"]["required_post_warmup_steps"] == 6
    assert validation["validation"]["formal_training_authorized"] is False


def test_candidate_validation_actual_launcher_mapping():
    args = mapped_argv(CONFIG)
    assert args["trainer.total_training_steps"] == "16"
    assert args["data.shuffle"] == "false"
    assert args["actor_rollout_ref.rollout.ignore_eos"] == "false"
    assert args["actor_rollout_ref.actor.defer_optimizer_state_load"] == "true"
    assert args["actor_rollout_ref.actor.fsdp_config.optimizer_offload"] == "true"
    assert args["actor_rollout_ref.rollout.response_length"] == "1024"


@pytest.mark.parametrize("field", ["ignore_eos", "deferred", "authorization", "steps"])
def test_candidate_validation_preflight_rejects_contract_drift(tmp_path: Path, field: str):
    config = yaml.safe_load(CONFIG.read_text())
    if field == "ignore_eos":
        config["rollout"]["ignore_eos"] = True
    elif field == "deferred":
        config["actor"]["defer_optimizer_state_load"] = False
    elif field == "authorization":
        config["validation"]["formal_training_authorized"] = True
    else:
        config["training"]["total_optimizer_steps"] = 15
    path = tmp_path / "candidate_128.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    result = validate_config(path, ROOT)
    assert result["status"] == "FAIL"
    assert result["checks"]["candidate_validation_execution_contract"] is False
