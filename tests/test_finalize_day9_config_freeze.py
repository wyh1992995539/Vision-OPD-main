import tempfile
import unittest
from pathlib import Path

import yaml

from scripts.day9_formal_training_readiness import FORMAL_CONFIG_EXPECTED
from scripts.finalize_day9_config_freeze import build_freeze


def config_from_contract():
    config = {}
    for dotted, value in FORMAL_CONFIG_EXPECTED.items():
        current = config
        keys = dotted.split(".")
        for key in keys[:-1]:
            current = current.setdefault(key, {})
        current[keys[-1]] = value
    return config


class FinalizeDay9ConfigFreezeTest(unittest.TestCase):
    def test_builds_auditable_1024_sample_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "configs/vopd_1024.yaml"
            baseline_path = root / "artifacts/runs/E-D8-001/config.yaml"
            config_path.parent.mkdir(parents=True)
            baseline_path.parent.mkdir(parents=True)
            formal = config_from_contract()
            baseline = config_from_contract()
            baseline["experiment"]["id"] = "E-D8-001"
            baseline["data"]["dataloader_num_workers"] = 4
            baseline["data"]["shuffle"] = False
            baseline["training"]["expected_samples"] = 64
            baseline["training"]["total_optimizer_steps"] = 8
            config_path.write_text(yaml.safe_dump(formal), encoding="utf-8")
            baseline_path.write_text(yaml.safe_dump(baseline), encoding="utf-8")

            result = build_freeze(
                root,
                config_path,
                baseline_path,
                "2026-09-01T00:00:00+00:00",
            )

            self.assertEqual(result["config_gate_status"], "PASS")
            self.assertTrue(result["task3_completed"])
            self.assertEqual(result["formal_contract"]["expected_samples"], 1024)
            self.assertEqual(result["formal_contract"]["optimizer_steps"], 128)
            self.assertTrue(result["invariants"]["dataloader_child_processes_disabled"])
            self.assertFalse(result["advance_to_day10"])

    def test_rejects_worker_count_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "configs/vopd_1024.yaml"
            baseline_path = root / "artifacts/runs/E-D8-001/config.yaml"
            config_path.parent.mkdir(parents=True)
            baseline_path.parent.mkdir(parents=True)
            config = config_from_contract()
            config["data"]["dataloader_num_workers"] = 4
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            baseline_path.write_text(yaml.safe_dump(config), encoding="utf-8")

            result = build_freeze(root, config_path, baseline_path)

            self.assertEqual(result["config_gate_status"], "FAIL")
            self.assertFalse(result["task3_completed"])


if __name__ == "__main__":
    unittest.main()
