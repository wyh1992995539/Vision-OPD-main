import copy
import unittest

from scripts.finalize_day9_preflight_report import build_markdown, build_payload


def valid_sources():
    return {
        "budget": {
            "gate_status": "PASS",
            "selected_budget": {
                "planning": {"dual_gpu_hours": 1.02, "estimated_cost_cny": 12.25},
                "reservation": {"dual_gpu_hours": 1.74, "estimated_cost_cny": 20.84},
            },
            "project_cap": {
                "current_platform_cumulative_charge_cny": 200.0,
                "projected_total_after_e_d10_reservation_cny": 220.84,
                "remaining_after_e_d10_reservation_cny": 1779.16,
            },
        },
        "data": {"status": "PASS"},
        "base": {"status": "PASS"},
        "launcher": {
            "status": "PASS",
            "checks": {"sample_budget_matches_steps": True},
            "gpu_used": False,
            "config_sha256": "abc123",
            "train_rows": 1024,
            "missing_image_paths": [],
        },
        "readiness": {
            "readiness_status": "PASS",
            "blocking_gates": [],
            "gates": {
                "data": {"status": "PASS", "evidence": "1024 rows"},
                "storage": {"status": "PASS", "evidence": "capacity formula passes"},
            },
            "storage": {
                "status": "PASS",
                "day8_checkpoint_size_bytes": 57_000_000_000,
                "required_bytes": 119_000_000_000,
                "available_bytes": 122_000_000_000,
            },
            "git": {"commit": "deadbeef", "clean": True},
        },
        "config_freeze": {
            "config_gate_status": "PASS",
            "task3_completed": True,
            "sources": {"formal_config": {"sha256": "abc123"}},
            "formal_contract": {
                "expected_samples": 1024,
                "global_batch_size": 8,
                "optimizer_steps": 128,
                "total_epochs": 1,
                "require_full_epoch": True,
            },
            "commands": {
                "preflight_only": "bash scripts/run_vopd_2gpu.sh --preflight-only",
                "formal_training_after_all_gates_pass": "bash scripts/run_vopd_2gpu.sh --run",
            },
        },
        "day8_stability": {
            "cold_reload": {
                "status": "PASS",
                "prediction_count": 5,
                "inference_error_count": 0,
            },
            "caveats": ["DataLoader worker Killed after checkpoint save"],
        },
    }


class FinalizeDay9PreflightReportTest(unittest.TestCase):
    def test_builds_pass_to_task5_report_from_passing_sources(self):
        sources = valid_sources()
        provenance = {
            name: {"path": f"evidence/{name}.json", "sha256": name * 2}
            for name in sources
        }
        payload = build_payload(
            sources,
            provenance,
            "2026-09-02T00:00:00+00:00",
        )
        markdown = build_markdown(payload)

        self.assertEqual(payload["report_status"], "PASS_TO_TASK5")
        self.assertTrue(payload["task4_completed"])
        self.assertFalse(payload["advance_to_day10"])
        self.assertEqual(payload["storage"]["report_classification"], "PASS_WITH_LOW_HEADROOM")
        self.assertIn("Day 10 仍等待中止条件冻结", markdown)

    def test_rejects_stale_blocked_readiness(self):
        sources = copy.deepcopy(valid_sources())
        sources["readiness"]["readiness_status"] = "BLOCKED"
        sources["readiness"]["blocking_gates"] = ["storage"]

        with self.assertRaisesRegex(ValueError, "readiness_pass"):
            build_payload(sources, {})

    def test_classifies_material_storage_margin_as_pass(self):
        sources = valid_sources()
        sources["readiness"]["storage"]["available_bytes"] = 130_000_000_000
        payload = build_payload(sources, {})

        self.assertEqual(payload["storage"]["report_classification"], "PASS")
        self.assertEqual(payload["risks"][0]["severity"], "INFO")


if __name__ == "__main__":
    unittest.main()
