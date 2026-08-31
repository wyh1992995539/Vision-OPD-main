import unittest

from scripts.finalize_day8_stability import parse_training_log, projection_scenarios


class FinalizeDay8StabilityTest(unittest.TestCase):
    def metric_line(self, step: int, *, step_seconds: float = 20.0) -> str:
        values = {
            "step": step,
            "actor/vopd_loss": 0.05,
            "actor/grad_norm": 2.0,
            "timing_s/gen": 8.0,
            "timing_s/update_actor": 11.0,
            "timing_s/step": step_seconds,
            "perf/max_memory_allocated_gb": 50.0,
            "perf/max_memory_reserved_gb": 60.0,
            "perf/cpu_memory_used_gb": 100.0,
            "response_length/mean": 4.0,
            "response_length/max": 5.0,
            "response_length/clip_ratio": 0.0,
            "response/aborted_ratio": 0.0,
            "prompt_length/max": 3500.0,
            "prompt_length/clip_ratio": 0.0,
            "evidence/student_param_probe_max_delta_after_optimizer": 1e-6,
            "evidence/teacher_param_probe_max_delta_after_optimizer": 0.0,
            "evidence/teacher_grad_non_none_count": 0.0,
            "evidence/teacher_param_probe_max_delta_after_ema": 2e-7,
            "evidence/ema_update_applied": 1.0,
            "self_distillation/teacher_always_on_fraction": 1.0,
            "self_distillation/teacher_image_swap_fraction": 1.0,
            "training/global_step": step,
        }
        return "worker " + " - ".join(f"{key}:{value}" for key, value in values.items())

    def test_parser_extracts_required_metrics_and_error_signals(self):
        text = self.metric_line(1) + "\nRuntimeError: DataLoader worker is killed by signal: Killed\n"
        steps, signals = parse_training_log(text)
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0]["actor/vopd_loss"], 0.05)
        self.assertEqual(signals["dataloader_worker_killed_count"], 1)
        self.assertEqual(signals["nonfinite_token_count"], 0)

    def test_projection_includes_fixed_overheads_and_127_steady_steps(self):
        result = projection_scenarios(
            startup_seconds=100.0,
            first_step_seconds=80.0,
            steady_step_seconds=[20.0] * 7,
            checkpoint_save_seconds=120.0,
            cost_per_dual_gpu_hour=12.0,
        )
        expected_seconds = 100.0 + 80.0 + 127 * 20.0 + 120.0
        self.assertEqual(result["median"]["total_seconds"], expected_seconds)
        self.assertAlmostEqual(
            result["mean"]["estimated_cost_cny"], expected_seconds / 3600 * 12.0
        )

    def test_parser_rejects_duplicate_step_records(self):
        text = self.metric_line(1) + "\n" + self.metric_line(1)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse_training_log(text)


if __name__ == "__main__":
    unittest.main()
