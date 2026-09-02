import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from scripts.monitor_vopd_training import (
    RuleEvaluator,
    load_policy,
    parse_training_metric_line,
    replay,
    scan_fatal_log_line,
    terminate_process_group,
    validate_checkpoint,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = PROJECT_ROOT / "configs/vopd_abort_policy.yaml"


def healthy_metric(step=1, **overrides):
    row = {
        "step": step,
        "loss": 0.1,
        "grad_norm": 1.0,
        "student_optimizer_delta": 1e-6,
        "teacher_optimizer_delta": 0.0,
        "teacher_grad_non_none_count": 0,
        "teacher_ema_delta": 1e-7,
        "aborted_ratio": 0.0,
    }
    row.update(overrides)
    return row


def telemetry(policy, gpu_ratio=0.5, cgroup_ratio=0.5, free_bytes=None, events=None):
    total = 1000
    return {
        "gpus": [{"memory_used_bytes": int(total * gpu_ratio), "memory_total_bytes": total}],
        "cgroup": {
            "supported": True,
            "memory_current_bytes": int(total * cgroup_ratio),
            "memory_max_bytes": total,
            "memory_events": events or {"oom": 0, "oom_kill": 0},
        },
        "disk_free_bytes": free_bytes or policy["disk"]["prelaunch_required_bytes"],
    }


class VopdAbortGuardTest(unittest.TestCase):
    def setUp(self):
        self.policy = load_policy(POLICY_PATH)

    def test_policy_formulas_are_frozen(self):
        disk = self.policy["disk"]
        expected = 2 * disk["checkpoint_estimate_bytes"] + disk["reserve_bytes"]
        self.assertEqual(disk["prelaunch_required_bytes"], expected)
        self.assertEqual(disk["runtime_soft_floor_bytes"], disk["checkpoint_estimate_bytes"] + disk["reserve_bytes"])

    def test_parses_real_console_metric_shape(self):
        line = (
            "(TaskRunner pid=1) step:7 - actor/vopd_loss:0.2 - actor/grad_norm:3.0 "
            "- evidence/student_param_probe_max_delta_after_optimizer:1e-6 "
            "- evidence/teacher_param_probe_max_delta_after_optimizer:0.0 "
            "- evidence/teacher_grad_non_none_count:0.0 "
            "- evidence/teacher_param_probe_max_delta_after_ema:2e-7 - response/aborted_ratio:0.0"
        )
        row = parse_training_metric_line(line)
        self.assertEqual(row["step"], 7)
        self.assertEqual(row["loss"], 0.2)
        self.assertEqual(row["teacher_grad_non_none_count"], 0.0)

    def test_healthy_metrics_do_not_abort(self):
        evaluator = RuleEvaluator(self.policy)
        self.assertEqual(evaluator.evaluate_metric(healthy_metric()), [])

    def test_nonfinite_and_teacher_contract_are_immediate(self):
        cases = [
            (healthy_metric(loss=float("nan")), "nonfinite_metric"),
            (healthy_metric(teacher_grad_non_none_count=1), "teacher_direct_gradient"),
            (healthy_metric(teacher_optimizer_delta=1e-9), "teacher_optimizer_changed"),
        ]
        for row, expected in cases:
            with self.subTest(expected=expected):
                issues = RuleEvaluator(self.policy).evaluate_metric(row)
                issue = next(item for item in issues if item["rule"] == expected)
                self.assertTrue(issue["immediate"])

    def test_two_nonpositive_ema_or_student_steps_abort(self):
        evaluator = RuleEvaluator(self.policy)
        self.assertEqual(evaluator.evaluate_metric(healthy_metric(1, teacher_ema_delta=0.0)), [])
        issues = evaluator.evaluate_metric(healthy_metric(2, teacher_ema_delta=0.0))
        self.assertIn("teacher_ema_not_updating", {item["rule"] for item in issues})
        evaluator = RuleEvaluator(self.policy)
        self.assertEqual(evaluator.evaluate_metric(healthy_metric(1, student_optimizer_delta=0.0)), [])
        issues = evaluator.evaluate_metric(healthy_metric(2, student_optimizer_delta=0.0))
        self.assertIn("student_optimizer_not_updating", {item["rule"] for item in issues})

    def test_three_consecutive_generation_errors_abort_and_reset(self):
        evaluator = RuleEvaluator(self.policy)
        evaluator.evaluate_metric(healthy_metric(1, aborted_ratio=0.2))
        evaluator.evaluate_metric(healthy_metric(2, aborted_ratio=0.0))
        evaluator.evaluate_metric(healthy_metric(3, aborted_ratio=0.2))
        evaluator.evaluate_metric(healthy_metric(4, aborted_ratio=0.2))
        issues = evaluator.evaluate_metric(healthy_metric(5, aborted_ratio=0.2))
        self.assertIn("consecutive_generation_errors", {item["rule"] for item in issues})

    def test_gpu_and_cgroup_pressure_require_three_samples(self):
        for key, kwargs, expected in (
            ("gpu", {"gpu_ratio": 0.96}, "gpu_memory_pressure"),
            ("cgroup", {"cgroup_ratio": 0.96}, "cgroup_memory_pressure"),
        ):
            with self.subTest(key=key):
                evaluator = RuleEvaluator(self.policy)
                self.assertEqual(evaluator.evaluate_telemetry(telemetry(self.policy, **kwargs)), [])
                self.assertEqual(evaluator.evaluate_telemetry(telemetry(self.policy, **kwargs)), [])
                issues = evaluator.evaluate_telemetry(telemetry(self.policy, **kwargs))
                self.assertIn(expected, {item["rule"] for item in issues})

    def test_cgroup_oom_increment_is_immediate(self):
        evaluator = RuleEvaluator(self.policy)
        evaluator.evaluate_telemetry(telemetry(self.policy, events={"oom": 2, "oom_kill": 1}))
        issues = evaluator.evaluate_telemetry(telemetry(self.policy, events={"oom": 3, "oom_kill": 1}))
        issue = next(item for item in issues if item["rule"] == "cgroup_oom_event")
        self.assertTrue(issue["immediate"])

    def test_disk_hard_and_soft_floors(self):
        evaluator = RuleEvaluator(self.policy)
        hard = evaluator.evaluate_telemetry(telemetry(self.policy, free_bytes=1))
        self.assertIn("disk_hard_floor", {item["rule"] for item in hard})
        evaluator = RuleEvaluator(self.policy)
        free = self.policy["disk"]["runtime_soft_floor_bytes"] - 1
        self.assertEqual(evaluator.evaluate_telemetry(telemetry(self.policy, free_bytes=free)), [])
        soft = evaluator.evaluate_telemetry(telemetry(self.policy, free_bytes=free))
        self.assertIn("disk_checkpoint_reserve", {item["rule"] for item in soft})

    def test_wall_time_limit(self):
        evaluator = RuleEvaluator(self.policy)
        self.assertEqual(evaluator.evaluate_elapsed(38 * 3600 - 1), [])
        self.assertEqual(evaluator.evaluate_elapsed(38 * 3600)[0]["rule"], "wall_time_limit")

    def test_fatal_log_patterns_are_specific(self):
        self.assertEqual(scan_fatal_log_line("normal inference message"), [])
        self.assertIn(
            "dataloader_worker_killed",
            scan_fatal_log_line("RuntimeError: DataLoader worker (pid 7) is killed by signal: Killed."),
        )
        self.assertIn("checkpoint_save_failure", scan_fatal_log_line("Error saving checkpoint: no space"))

    def test_checkpoint_postcondition(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            checkpoint = self.policy["checkpoint"]
            step = checkpoint["expected_final_step"]
            root = output / "checkpoints"
            (root / f"global_step_{step}").mkdir(parents=True)
            (root / checkpoint["marker"]).write_text(str(step), encoding="utf-8")
            for relative in checkpoint["required_relative_files"]:
                path = root / f"global_step_{step}" / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"x")
            self.assertEqual(validate_checkpoint(output, self.policy)["status"], "PASS")
            (root / f"global_step_{step}" / checkpoint["required_relative_files"][0]).write_bytes(b"")
            self.assertEqual(validate_checkpoint(output, self.policy)["status"], "FAIL")

    @patch("scripts.monitor_vopd_training.os.killpg")
    def test_termination_escalates_from_term_to_kill(self, killpg):
        process = Mock()
        process.pid = 1234
        process.wait.side_effect = [__import__("subprocess").TimeoutExpired("x", 1), 0]
        receipt = terminate_process_group(process, 0.01)
        self.assertIsNotNone(receipt["kill_sent_at_utc"])
        self.assertEqual(killpg.call_count, 2)

    def test_day8_replay_has_clean_metrics_and_detects_worker_kill(self):
        result = replay(
            PROJECT_ROOT / "artifacts/runs/E-D8-001/metrics.jsonl",
            PROJECT_ROOT / "artifacts/runs/E-D8-001/logs/train.log",
            self.policy,
        )
        self.assertEqual(result["metric_rows"], 8)
        self.assertEqual(result["metric_issues"], [])
        self.assertIn("dataloader_worker_killed", {row["rule"] for row in result["fatal_log_issues"]})


if __name__ == "__main__":
    unittest.main()
