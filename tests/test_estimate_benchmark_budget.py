import unittest

from scripts.estimate_benchmark_budget import make_scenario


class EstimateBenchmarkBudgetTest(unittest.TestCase):
    def test_scenario_uses_dual_gpu_instance_hour_price_once(self):
        scenario = make_scenario(
            name="test",
            effective_concurrency=5,
            overhead_fraction=0,
            individual_inference_seconds=18000,
            judge_instances=0,
            price_cny_per_hour=11.96,
        )
        self.assertEqual(scenario["estimated_wall_hours"], 1.0)
        self.assertEqual(scenario["estimated_cost_cny"], 11.96)

    def test_scenario_includes_judge_and_buffer(self):
        scenario = make_scenario(
            name="test",
            effective_concurrency=10,
            overhead_fraction=0.5,
            individual_inference_seconds=36000,
            judge_instances=10,
            price_cny_per_hour=10,
        )
        self.assertEqual(scenario["judge_wall_seconds_before_buffer"], 50)
        self.assertEqual(scenario["estimated_wall_hours"], 1.5 * (3600 + 50) / 3600)


if __name__ == "__main__":
    unittest.main()

