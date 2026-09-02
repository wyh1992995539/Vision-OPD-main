import math
import unittest
from pathlib import Path

from scripts.finalize_day9_budget import build_artifact, build_budget


class FinalizeDay9BudgetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[1]
        cls.payload = build_budget(cls.project_root, "2026-09-01T00:00:00+00:00")

    def test_projection_recomputes_128_steps(self):
        self.assertEqual(self.payload["training_contract"]["target_optimizer_steps"], 128)
        self.assertEqual(self.payload["projection_validation"]["status"], "PASS")
        planning = self.payload["selected_budget"]["planning"]
        reservation = self.payload["selected_budget"]["reservation"]
        self.assertTrue(math.isclose(planning["dual_gpu_hours"], 1.0245956605105246))
        self.assertTrue(math.isclose(reservation["estimated_cost_cny"], 20.842752746504694))

    def test_projection_records_are_not_counted_as_observed_spend(self):
        entries = self.payload["historical_cost_reconciliation"]["entries"]
        day5 = next(row for row in entries if row["record_id"] == "E-D5-001-projection")
        self.assertFalse(day5["included_in_documented_subtotal"])
        self.assertEqual(day5["record_type"], "projection")

    def test_project_cap_waits_for_platform_bill(self):
        self.assertEqual(self.payload["gate_status"], "PENDING_BUDGET_RECONCILIATION")
        self.assertEqual(self.payload["project_cap"]["status"], "PENDING_PLATFORM_BILLING_INPUT")
        self.assertGreater(
            self.payload["project_cap"]["maximum_current_platform_cumulative_charge_for_launch_cny"],
            1900.0,
        )

    def test_user_reported_200_cny_closes_budget_gate(self):
        payload = build_budget(
            self.project_root,
            "2026-09-01T00:00:00+00:00",
            current_autodl_cost_cny=200.0,
        )
        cap = payload["project_cap"]
        self.assertEqual(payload["gate_status"], "PASS")
        self.assertEqual(cap["status"], "PASS")
        self.assertIsNone(payload["required_next_input"])
        self.assertTrue(
            math.isclose(
                cap["projected_total_after_e_d10_reservation_cny"],
                220.8427527465047,
            )
        )

    def test_report_artifact_has_auditable_visuals_and_sources(self):
        artifact = build_artifact(self.payload)
        self.assertEqual(artifact["surface"], "report")
        self.assertEqual(artifact["manifest"]["blocks"][0]["body"], "# Day 9 E-D10-001 Budget Gate")
        self.assertEqual(len(artifact["manifest"]["charts"]), 1)
        self.assertEqual(len(artifact["manifest"]["tables"]), 1)
        self.assertEqual(len(artifact["snapshot"]["datasets"]["projection_scenarios"]), 3)


if __name__ == "__main__":
    unittest.main()
