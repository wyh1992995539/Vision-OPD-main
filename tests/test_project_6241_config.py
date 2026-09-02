import unittest
from pathlib import Path

import yaml

from scripts.build_project_parquet import configured_splits as parquet_splits
from scripts.extract_project_images import configured_splits as extraction_splits
from scripts.validate_project_data import configured_splits as validation_splits


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/project_6241.yaml"


class Project6241ConfigTest(unittest.TestCase):
    def test_only_train_is_active(self):
        config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(config["data"]["active_split_count"], 1)
        self.assertEqual(config["data"]["splits"], {"train": config["data"]["splits"]["train"]})
        self.assertEqual(config["data"]["splits"]["train"]["size"], 6241)
        self.assertTrue(all(item["historical_only"] for item in config["data"]["historical_splits"].values()))

    def test_all_data_tools_resolve_the_same_active_manifest(self):
        self.assertEqual(extraction_splits(CONFIG), {"train_6241.jsonl": 6241})
        self.assertEqual(validation_splits(CONFIG), {"train_6241.jsonl": ("train", 6241)})
        self.assertEqual(
            parquet_splits(CONFIG),
            {"train": ("train_6241.jsonl", "train_6241.parquet", 6241)},
        )

    def test_training_arithmetic_is_explicit(self):
        config = yaml.safe_load((ROOT / "configs/vopd_6241.yaml").read_text(encoding="utf-8"))
        self.assertEqual(config["training"]["total_optimizer_steps"], 781)
        self.assertEqual(config["training"]["padded_samples"], 6248)
        self.assertEqual(config["training"]["padding_rows"], 7)
        self.assertTrue(config["data"]["full_coverage_padding"]["enabled"])


if __name__ == "__main__":
    unittest.main()
