"""CPU tests for the dedicated 128x16 candidate-validation controls."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

import scripts.audit_vopd_6241_candidate_128 as postflight
from scripts.run_vopd_6241_candidate_128_guarded import (
    DEFAULT_POLICY,
    load_candidate_policy,
    output_collisions,
    sha256_file,
    static_preflight,
)

ROOT = Path(__file__).resolve().parents[1]
GIB = 1024**3


def healthy_step(step: int) -> dict:
    return {
        "step": step,
        "loss": 0.1,
        "grad_norm": 1.0,
        "learning_rate": 0.0 if step == 1 else 2e-7,
        "student_optimizer_delta": 0.0 if step == 1 else 1e-6,
        "teacher_optimizer_delta": 0.0,
        "teacher_grad_non_none_count": 0.0,
        "teacher_ema_delta": 1e-7,
        "ema_update_applied": 1.0,
        "aborted_ratio": 0.0,
        "step_seconds": 20.0,
        "generation_seconds": 5.0,
        "checkpoint_save_seconds": 30.0 if step == 16 else None,
        "prompt_max_tokens": 7880.0,
        "prompt_clip_ratio": 0.0,
        "response_mean_tokens": 100.0,
        "response_max_tokens": 400.0,
        "response_clip_ratio": 0.0,
        "teacher_always_on_fraction": 1.0,
        "teacher_image_swap_fraction": 1.0,
    }


def test_dedicated_static_preflight_passes_without_gpu():
    result = static_preflight()
    assert result["status"] == "PASS", result["failed_checks"]
    assert result["gpu_used"] is False
    assert all(result["checks"].values())
    assert result["experiment_id"] == "E-D11-6K-VOPD-CANDIDATE-128"


def test_policy_freezes_resource_and_checkpoint_boundaries():
    policy, contract = load_candidate_policy(DEFAULT_POLICY)
    assert policy["memory"]["prelaunch_cgroup_minimum_bytes"] == 240 * GIB
    assert policy["memory"]["gpu_used_ratio_abort"] == 0.98
    assert policy["memory"]["cgroup_used_ratio_abort"] == 0.95
    assert policy["runtime"]["max_wall_time_hours"] == 4
    assert policy["checkpoint"]["allowed_save_steps"] == [16]
    assert policy["checkpoint"]["expected_final_step"] == 16
    assert contract["required_post_warmup_steps"] == 6
    assert contract["formal_training_authorized"] is False


@pytest.mark.parametrize("field", ["config_hash", "source_hash", "selection_hash", "authorization"])
def test_policy_drift_fails_closed(tmp_path: Path, field: str):
    value = yaml.safe_load(DEFAULT_POLICY.read_text())
    contract = value["candidate_validation"]
    if field == "config_hash":
        contract["expected_config_sha256"] = "0" * 64
    elif field == "source_hash":
        contract["expected_source_candidate_sha256"] = "0" * 64
    elif field == "selection_hash":
        contract["expected_selection_sha256"] = "0" * 64
    else:
        contract["formal_training_authorized"] = True
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(yaml.safe_dump(value))
    if field == "authorization":
        with pytest.raises(ValueError, match="must not authorize"):
            static_preflight(policy_path)
    else:
        result = static_preflight(policy_path)
        assert result["status"] == "FAIL"
        assert any(
            result["checks"][key] is False
            for key in ("config_hash_bound", "source_candidate_hash_bound", "selection_hash_bound")
        )


def test_output_collision_scope(tmp_path: Path):
    (tmp_path / "preflight").mkdir()
    (tmp_path / "preflight/candidate_guard_preflight.json").write_text("{}")
    assert output_collisions(tmp_path) == []
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs/train.log").write_text("started")
    assert output_collisions(tmp_path) == [str(tmp_path / "logs/train.log")]


def test_run_requires_fresh_billing_before_gpu_query():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_vopd_6241_candidate_128_guarded.py"),
            "--run",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "--run requires fresh billing amount and timestamp" in result.stderr


@pytest.fixture
def postflight_fixture(tmp_path: Path, monkeypatch):
    policy = yaml.safe_load(DEFAULT_POLICY.read_text())
    output = tmp_path / "run"
    policy["candidate_validation"]["output_dir"] = str(output)
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(yaml.safe_dump(policy))

    selection_path = ROOT / policy["candidate_validation"]["selection_manifest"]
    selection = json.loads(selection_path.read_text())
    sample_ids = [item["sample_id"] for item in selection["samples"]]
    config_path = ROOT / policy["candidate_validation"]["config"]

    for directory in (
        output / "logs",
        output / "preflight",
        output / "evidence/telemetry",
        output / "evidence/log_probs",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    (output / "logs/train.log").write_text("synthetic")
    (output / "evidence/guard_summary.json").write_text(json.dumps({"status": "PASS"}))
    live = {
        "status": "PASS",
        "live_checks": {"all": True},
        "source_bindings": {
            "config": {
                "path": str(config_path),
                "sha256": sha256_file(config_path),
            }
        },
    }
    (output / "preflight/live_launch_gate.json").write_text(json.dumps(live))
    invocation = {
        "experiment_id": policy["experiment_id"],
        "config_sha256": policy["candidate_validation"]["expected_config_sha256"],
        "train_file_sha256": selection["output"]["sha256"],
        "sample_ids": sample_ids,
        "training_contract": {"total_optimizer_steps": 16},
        "hydra_overrides": [],
    }
    (output / "preflight/run_invocation.json").write_text(json.dumps(invocation))
    (output / "preflight/preflight_summary.json").write_text(json.dumps({"status": "PASS"}))
    cgroup = {
        "supported": True,
        "memory_current_bytes": 180 * GIB,
        "memory_max_bytes": 240 * GIB,
        "memory_events": {"oom": 0, "oom_kill": 0},
    }
    (output / "evidence/telemetry/cgroup_memory.jsonl").write_text(
        json.dumps(cgroup) + "\n" + json.dumps(cgroup) + "\n"
    )
    for index in range(32):
        (output / f"evidence/log_probs/{index:02d}.pt").write_bytes(b"x")

    rows = [healthy_step(step) for step in range(1, 17)]
    signals = {
        "duplicate_metric_steps": 0,
        "traceback_count": 0,
        "cuda_oom_count": 0,
        "out_of_memory_error_count": 0,
        "dataloader_worker_killed_count": 0,
        "checkpoint_save_failure_count": 0,
    }
    monkeypatch.setattr(postflight, "parse_steps", lambda _: (copy.deepcopy(rows), copy.deepcopy(signals)))
    monkeypatch.setattr(
        postflight,
        "telemetry_summary",
        lambda _: {
            "status": "PASS",
            "peak_by_gpu": {
                "0": {"used_ratio": 0.90},
                "1": {"used_ratio": 0.91},
            },
        },
    )
    monkeypatch.setattr(postflight, "validate_checkpoint", lambda *_: {"status": "PASS"})
    monkeypatch.setattr(postflight, "checkpoint_io_matches", lambda *_: True)
    return policy_path, output, rows, signals


def test_postflight_passes_and_never_authorizes_formal_training(postflight_fixture):
    policy_path, _, _, _ = postflight_fixture
    report = postflight.audit(policy_path)
    assert report["status"] == "PASS_CANDIDATE_VALIDATION", report["failed_checks"]
    assert report["training_gate_pass"] is True
    assert report["validation_gate_pass"] is True
    assert report["formal_training_authorized"] is False
    assert report["warmup_contract"]["observed_post_warmup_steps"] == list(range(11, 17))
    assert report["response_length"]["observed_max_tokens"] == 400.0


@pytest.mark.parametrize("failure", ["forced_length", "post_warmup", "oom", "checkpoint", "binding"])
def test_postflight_fail_closed(postflight_fixture, monkeypatch, failure: str):
    policy_path, output, rows, _ = postflight_fixture
    if failure == "forced_length":
        for row in rows:
            row["response_max_tokens"] = 1024.0
    elif failure == "post_warmup":
        rows[-1]["student_optimizer_delta"] = 0.0
    elif failure == "oom":
        path = output / "evidence/telemetry/cgroup_memory.jsonl"
        samples = [json.loads(line) for line in path.read_text().splitlines()]
        samples[-1]["memory_events"]["oom_kill"] = 1
        path.write_text("\n".join(json.dumps(row) for row in samples) + "\n")
    elif failure == "checkpoint":
        monkeypatch.setattr(postflight, "validate_checkpoint", lambda *_: {"status": "FAIL"})
    else:
        live_path = output / "preflight/live_launch_gate.json"
        live = json.loads(live_path.read_text())
        live["source_bindings"]["config"]["sha256"] = "0" * 64
        live_path.write_text(json.dumps(live))
    report = postflight.audit(policy_path)
    assert report["status"] == "FAIL"
    assert report["validation_gate_pass"] is False
