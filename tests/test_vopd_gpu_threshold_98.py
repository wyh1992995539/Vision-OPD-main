import pytest

from scripts.monitor_vopd_training import RuleEvaluator, load_policy
from scripts.run_vopd_6241_pilot_guarded import DEFAULT_POLICY, ROOT, load_pilot_policy


@pytest.mark.parametrize("stage", ["formal", "16", "64"])
def test_98_percent_boundary_streak_reset_and_unchanged_cpu(stage):
    policy = (load_policy(ROOT / "configs/vopd_6241_abort_policy.yaml") if stage == "formal"
              else load_pilot_policy(DEFAULT_POLICY, stage)[0])
    assert policy["memory"]["gpu_used_ratio_abort"] == 0.98
    assert policy["memory"]["cgroup_used_ratio_abort"] == 0.95
    assert policy["memory"]["consecutive_samples"] == 3

    def sample(gpu, cpu=500):
        return {"gpus": [{"memory_used_bytes": 500, "memory_total_bytes": 1000},
                         {"memory_used_bytes": gpu, "memory_total_bytes": 1000}],
                "cgroup": {"supported": True, "memory_current_bytes": cpu,
                           "memory_max_bytes": 1000, "memory_events": {}},
                "disk_free_bytes": policy["disk"]["prelaunch_required_bytes"]}

    guard = RuleEvaluator(policy)
    # Sustained 97.9% does not abort; a dip also resets the high-water streak.
    for used in [979, 979, 979, 980, 990, 979, 980, 980]:
        assert guard.evaluate_telemetry(sample(used)) == []
    assert "gpu_memory_pressure" in {r["rule"] for r in guard.evaluate_telemetry(sample(980))}

    guard = RuleEvaluator(policy)
    assert guard.evaluate_telemetry(sample(500, 960)) == []
    assert guard.evaluate_telemetry(sample(500, 960)) == []
    assert "cgroup_memory_pressure" in {r["rule"] for r in guard.evaluate_telemetry(sample(500, 960))}
