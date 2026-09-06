import tempfile
import unittest
from pathlib import Path

import yaml

from scripts.audit_vopd_6241_pilot import evaluate_steps, parse_steps, projection
from scripts.run_vopd_6241_pilot_guarded import (
    DEFAULT_POLICY,
    load_pilot_policy,
    output_collisions,
)


def healthy_step(step: int) -> dict:
    return {
        "step": step,
        "loss": 0.1,
        "grad_norm": 1.0,
        "learning_rate": 2e-7,
        "student_optimizer_delta": 1e-6,
        "teacher_optimizer_delta": 0.0,
        "teacher_grad_non_none_count": 0.0,
        "teacher_ema_delta": 1e-7,
        "ema_update_applied": 1.0,
        "aborted_ratio": 0.0,
        "step_seconds": 20.0 + step,
        "generation_seconds": 5.0,
        "checkpoint_save_seconds": 30.0 if step == 8 else None,
        "prompt_max_tokens": 7880.0,
        "prompt_clip_ratio": 0.0,
        "response_mean_tokens": 100.0,
        "response_max_tokens": 1024.0,
        "response_clip_ratio": 0.1,
        "teacher_always_on_fraction": 1.0,
        "teacher_image_swap_fraction": 1.0,
    }


class Vopd6241PilotGuardTest(unittest.TestCase):
    def test_stage_policy_materialization(self):
        policy_16, contract_16 = load_pilot_policy(DEFAULT_POLICY, "16")
        policy_64, contract_64 = load_pilot_policy(DEFAULT_POLICY, "64")
        self.assertEqual(policy_16["checkpoint"]["expected_final_step"], 2)
        self.assertEqual(policy_64["checkpoint"]["expected_final_step"], 8)
        self.assertEqual(policy_16["runtime"]["max_wall_time_hours"], 4)
        self.assertTrue(
            policy_16["metrics"]["student_update_required_only_when_lr_positive"]
        )
        self.assertEqual(policy_64["runtime"]["max_wall_time_hours"], 8)
        self.assertIsNone(contract_16["prerequisite_postflight"])
        self.assertTrue(contract_64["require_cold_reload"])
        self.assertEqual(policy_16["memory"]["prelaunch_cgroup_minimum_bytes"], 192 * 1024**3)
        self.assertEqual(policy_64["memory"]["prelaunch_cgroup_minimum_bytes"], 224 * 1024**3)
        self.assertEqual(policy_64["memory"]["cgroup_used_ratio_abort"], 0.95)
        self.assertEqual(policy_64["memory"]["gpu_used_ratio_abort"], 0.98)
        self.assertEqual(policy_64["memory"]["consecutive_samples"], 3)

    def test_stage_memory_override_cannot_weaken_baseline(self):
        source = yaml.safe_load(DEFAULT_POLICY.read_text())
        source["pilot"]["stage_contracts"]["64"]["prelaunch_cgroup_minimum_bytes"] = 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.yaml"
            path.write_text(yaml.safe_dump(source))
            policy, _ = load_pilot_policy(path, "64")
        self.assertEqual(policy["memory"]["prelaunch_cgroup_minimum_bytes"], 192 * 1024**3)

    def test_peak_review_capacity_boundaries(self):
        from scripts.monitor_vopd_training import cgroup_has_minimum_capacity
        policy, _ = load_pilot_policy(DEFAULT_POLICY, "64")
        required = policy["memory"]["prelaunch_cgroup_minimum_bytes"]
        for gib, expected in ((220, False), (224, True), (256, True)):
            with self.subTest(gib=gib):
                self.assertEqual(cgroup_has_minimum_capacity(
                    {"supported": True, "memory_max_bytes": gib * 1024**3}, required
                ), expected)
        self.assertFalse(cgroup_has_minimum_capacity({}, required))

    def test_healthy_steps_pass_all_mechanism_checks(self):
        checks = evaluate_steps([healthy_step(1), healthy_step(2)], 2)
        self.assertTrue(all(checks.values()), checks)

    def test_warmup_zero_delta_is_allowed_but_positive_lr_requires_update(self):
        rows = [healthy_step(1), healthy_step(2)]
        rows[0]["learning_rate"] = 0.0
        rows[0]["student_optimizer_delta"] = 0.0
        checks = evaluate_steps(rows, 2, warmup_steps=10)
        self.assertTrue(checks["zero_lr_only_within_warmup"])
        self.assertTrue(checks["positive_learning_rate_observed"])
        self.assertTrue(checks["student_update_matches_learning_rate"])

        rows[1]["student_optimizer_delta"] = 0.0
        checks = evaluate_steps(rows, 2, warmup_steps=10)
        self.assertFalse(checks["student_update_matches_learning_rate"])

    def test_pilot_requires_at_least_one_positive_lr_step(self):
        rows = [healthy_step(1), healthy_step(2)]
        for row in rows:
            row["learning_rate"] = 0.0
            row["student_optimizer_delta"] = 0.0
        checks = evaluate_steps(rows, 2, warmup_steps=10)
        self.assertFalse(checks["positive_learning_rate_observed"])

    def test_zero_lr_after_warmup_and_negative_lr_fail_closed(self):
        rows = [healthy_step(1), healthy_step(2)]
        rows[1]["learning_rate"] = 0.0
        checks = evaluate_steps(rows, 2, warmup_steps=1)
        self.assertFalse(checks["zero_lr_only_within_warmup"])
        rows[1]["learning_rate"] = -1e-7
        checks = evaluate_steps(rows, 2, warmup_steps=10)
        self.assertFalse(checks["learning_rate_nonnegative"])

    def test_missing_learning_rate_fails_postflight(self):
        rows = [healthy_step(1), healthy_step(2)]
        del rows[0]["learning_rate"]
        checks = evaluate_steps(rows, 2, warmup_steps=10)
        self.assertFalse(checks["required_metrics_present_and_finite"])
        self.assertFalse(checks["student_update_matches_learning_rate"])

    def test_teacher_gradient_and_overlength_fail_closed(self):
        rows = [healthy_step(1), healthy_step(2)]
        rows[1]["teacher_grad_non_none_count"] = 1
        rows[1]["response_max_tokens"] = 1025
        checks = evaluate_steps(rows, 2)
        self.assertFalse(checks["teacher_direct_gradient_absent"])
        self.assertFalse(checks["response_within_frozen_limit"])

    def test_metric_parser_reads_pilot_evidence(self):
        line = (
            "step:1 - actor/vopd_loss:0.1 - actor/grad_norm:2.0 - actor/lr:0.0 "
            "- evidence/student_param_probe_max_delta_after_optimizer:1e-6 "
            "- evidence/teacher_param_probe_max_delta_after_optimizer:0 "
            "- evidence/teacher_grad_non_none_count:0 "
            "- evidence/teacher_param_probe_max_delta_after_ema:2e-7 "
            "- evidence/ema_update_applied:1 - response/aborted_ratio:0 "
            "- timing_s/step:25 - timing_s/gen:8 "
            "- prompt_length/max:7880 - prompt_length/clip_ratio:0 "
            "- response_length/mean:100 - response_length/max:1024 "
            "- response_length/clip_ratio:0.25 "
            "- self_distillation/teacher_always_on_fraction:1 "
            "- self_distillation/teacher_image_swap_fraction:1"
        )
        rows, signals = parse_steps(line)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["learning_rate"], 0.0)
        self.assertEqual(rows[0]["response_max_tokens"], 1024.0)
        self.assertEqual(rows[0]["teacher_image_swap_fraction"], 1.0)
        self.assertEqual(signals["duplicate_metric_steps"], 0)

    def test_64_step_projection_has_three_scenarios(self):
        rows = [healthy_step(step) for step in range(1, 9)]
        result = projection(rows, {"max_observed_elapsed_seconds": 260}, 11.96)
        self.assertEqual(result["target_optimizer_steps"], 780)
        self.assertEqual(result["checkpoint_count"], 2)
        self.assertEqual(
            set(result["scenarios"]), {"median", "mean", "conservative_max"}
        )

    def test_output_collision_scope_ignores_static_preflight_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "preflight").mkdir()
            (root / "preflight/preflight_summary.json").write_text("{}")
            self.assertEqual(output_collisions(root), [])
            (root / "logs").mkdir()
            (root / "logs/train.log").write_text("started")
            self.assertEqual(output_collisions(root), [str(root / "logs/train.log")])


if __name__ == "__main__":
    unittest.main()
