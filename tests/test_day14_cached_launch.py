import copy
import json
import os
from pathlib import Path
import subprocess

import yaml

from scripts.monitor_vopd_training import RuleEvaluator, load_policy, parse_training_metric_line
from scripts.promote_cached_prefix_6241 import promoted_config, verify_receipt
from scripts.run_day14_cached import CONFIG, POLICY, RECEIPT, static_preflight

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts/runs/E-D14-6K-CACHED-001"
LIVE_LAUNCH_GATE = RUN / "preflight/day14_live_launch_gate.json"
EXIT_RECEIPT = RUN / "evidence/exit_receipt.json"


def completed_run_credentials():
    if not LIVE_LAUNCH_GATE.is_file() or not EXIT_RECEIPT.is_file():
        return None
    launch = json.loads(LIVE_LAUNCH_GATE.read_text())
    exit_receipt = json.loads(EXIT_RECEIPT.read_text())
    if exit_receipt.get("guard_exit_code") != 0:
        return None
    return launch, exit_receipt


def test_formal_config_is_exact_promoted_candidate():
    receipt = json.loads(RECEIPT.read_text())
    candidate_path = Path(receipt["source_candidate"]["path"])
    candidate = yaml.safe_load(candidate_path.read_text())
    formal = yaml.safe_load(CONFIG.read_text())
    assert formal == promoted_config(candidate, RECEIPT)
    assert formal["status"] == "ready_after_day13_gate"
    assert formal["promotion"]["formal_training_authorized"] is True
    completed = completed_run_credentials()
    if completed is None:
        assert verify_receipt(RECEIPT, CONFIG)
    else:
        launch, exit_receipt = completed
        assert launch["status"] == "PASS"
        assert launch["checks"]["promotion_receipt_valid"] is True
        assert launch["failed_checks"] == []
        assert exit_receipt["training_summary"]["status"] == "PASS"


def test_day14_static_preflight_passes_without_gpu_use():
    result = static_preflight()
    assert result["training_started"] is False
    assert result["gpu_used"] is False
    completed = completed_run_credentials()
    if completed is None:
        assert result["status"] == "PASS", result["failed_checks"]
        assert all(result["checks"].values())
    else:
        launch, _ = completed
        # The promotion receipt freezes launch-critical source hashes. A source
        # repair after a completed run must not rewrite that historical receipt.
        assert result["status"] == "FAIL"
        assert result["failed_checks"] == ["promotion_receipt_valid"]
        assert launch["status"] == "PASS"
        assert launch["checks"]["promotion_receipt_valid"] is True


def test_policy_requires_cached_runtime_invariants():
    policy = load_policy(POLICY)
    assert policy["experiment_id"] == "E-D14-6K-CACHED-001"
    assert policy["checkpoint"]["allowed_save_steps"] == [390, 780]
    assert policy["checkpoint"]["expected_final_step"] == 780
    assert policy["disk"]["prelaunch_required_bytes"] == 130 * 1024**3
    assert policy["memory"]["prelaunch_cgroup_minimum_bytes"] == 240 * 1024**3
    assert policy["metrics"]["cached_prefix_required"] is True
    assert policy["metrics"]["cached_expected_records_per_step"] == 8
    assert policy["metrics"]["cached_online_generation_calls_max"] == 0


def test_cached_runtime_metrics_are_parsed_and_guarded():
    policy = load_policy(POLICY)
    line = (
        "step:1 - cached_prefix/resolved_records:8.0 - "
        "cached_prefix/online_generation_calls:0.0 - actor/vopd_loss:0.01 - "
        "actor/grad_norm:1.0 - actor/lr:2e-7 - "
        "evidence/student_param_probe_max_delta_after_optimizer:1e-7 - "
        "evidence/teacher_param_probe_max_delta_after_optimizer:0.0 - "
        "evidence/teacher_grad_non_none_count:0.0 - "
        "evidence/teacher_param_probe_max_delta_after_ema:1e-7 - "
        "response/aborted_ratio:0.0"
    )
    row = parse_training_metric_line(line)
    assert row["cached_resolved_records"] == 8.0
    assert row["cached_online_generation_calls"] == 0.0
    assert RuleEvaluator(policy).evaluate_metric(row) == []

    online = copy.deepcopy(row)
    online["cached_online_generation_calls"] = 1.0
    issues = RuleEvaluator(policy).evaluate_metric(online)
    assert [issue["rule"] for issue in issues] == ["cached_prefix_online_generation_detected"]

    missing = copy.deepcopy(row)
    missing["cached_resolved_records"] = 7.0
    issues = RuleEvaluator(policy).evaluate_metric(missing)
    assert [issue["rule"] for issue in issues] == ["cached_prefix_resolution_mismatch"]


def test_raw_training_entry_remains_blocked_without_guard():
    env = os.environ.copy()
    env.pop("VOPD_GUARD_ACTIVE", None)
    env["CONDA_DEFAULT_ENV"] = "vision-opd"
    result = subprocess.run(
        ["bash", "scripts/run_cached_prefix_2gpu.sh", "--run"],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 2
    assert "Direct --run is blocked" in result.stderr
