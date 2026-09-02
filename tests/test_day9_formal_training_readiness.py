import tempfile
import unittest
from collections import namedtuple
from pathlib import Path
from unittest.mock import patch

import yaml

from scripts.day9_formal_training_readiness import (
    BASE_PATH,
    EXPERIMENT_ID,
    FORMAL_CONFIG_EXPECTED,
    GIB,
    TRAIN_PATH,
    audit_config,
    audit_output,
    audit_storage,
)


def formal_config():
    config = {}
    for dotted, value in FORMAL_CONFIG_EXPECTED.items():
        current = config
        keys = dotted.split(".")
        for key in keys[:-1]:
            current = current.setdefault(key, {})
        current[keys[-1]] = value
    return config


class Day9FormalTrainingReadinessTest(unittest.TestCase):
    def test_formal_config_identity_passes_and_day7_identity_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "configs").mkdir()
            config = formal_config()
            config["experiment"]["id"] = "E-D7-001"
            config["paths"]["output_dir"] = "artifacts/runs/E-D7-001"
            config["smoke"] = {"expected_samples": 16, "total_optimizer_steps": 2}
            path = root / "configs/vopd_1024.yaml"
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            self.assertEqual(audit_config(root)["status"], "FAIL")
            config["experiment"]["id"] = EXPERIMENT_ID
            config["paths"]["output_dir"] = "artifacts/runs/E-D10-001"
            del config["smoke"]
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            self.assertEqual(audit_config(root)["status"], "PASS")

    def test_worker_count_is_part_of_frozen_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "configs").mkdir()
            config = formal_config()
            config["data"]["dataloader_num_workers"] = 4
            path = root / "configs/vopd_1024.yaml"
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            result = audit_config(root)
            self.assertEqual(result["status"], "FAIL")
            self.assertFalse(result["comparisons"]["data.dataloader_num_workers"]["match"])

    def test_output_gate_allows_only_named_preflight_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "artifacts/runs/E-D10-001"
            preflight = output / "preflight"
            preflight.mkdir(parents=True)
            (preflight / "budget_projection.json").write_text("{}", encoding="utf-8")
            self.assertEqual(audit_output(root, output)["status"], "PASS")
            (output / "checkpoint.bin").write_bytes(b"collision")
            self.assertEqual(audit_output(root, output)["status"], "FAIL")

    def test_storage_formula_is_two_checkpoints_plus_five_gib(self):
        DiskUsage = namedtuple("usage", "total used free")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "out"
            checkpoint = root / "checkpoint"
            output.mkdir()
            checkpoint.mkdir()
            (checkpoint / "weights").write_bytes(b"x" * 16)
            with patch(
                "scripts.day9_formal_training_readiness.shutil.disk_usage",
                return_value=DiskUsage(20 * GIB, 10 * GIB, 5 * GIB),
            ):
                result = audit_storage(output, checkpoint)
            self.assertEqual(result["required_bytes"], 5 * GIB + 32)
            self.assertEqual(result["shortage_bytes"], 32)
            self.assertEqual(result["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
