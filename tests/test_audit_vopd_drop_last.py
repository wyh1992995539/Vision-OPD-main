import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from scripts.audit_vopd_drop_last import build_audit, write_artifacts


class DropLastAuditTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.parquet = self.root / "train.parquet"
        pq.write_table(pa.table({"sample_id": [f"id-{index}" for index in range(9)]}), self.parquet)
        self.project_path = self.root / "project.yaml"
        self.training_path = self.root / "training.yaml"
        self.policy_path = self.root / "policy.yaml"
        self.trainer_path = self.root / "ray_trainer.py"
        self.project = {
            "reproducibility": {"dataloader_seed": 42},
            "training_contract": {
                "active_train_rows": 9,
                "global_batch_size": 4,
                "optimizer_steps": 2,
                "effective_train_samples": 8,
                "padding_rows": 0,
                "dropped_rows": 1,
                "require_full_coverage_sampler": False,
                "tail_policy": "native_drop_last",
            },
        }
        self.training = {
            "experiment": {"seed": 42},
            "paths": {"train_file": str(self.parquet)},
            "data": {
                "train_batch_size": 4,
                "shuffle": True,
                "tail_policy": "native_drop_last",
            },
            "training": {
                "source_samples": 9,
                "expected_samples": 8,
                "padded_samples": 8,
                "padding_rows": 0,
                "dropped_rows": 1,
                "total_optimizer_steps": 2,
                "total_epochs": 1,
                "require_full_epoch": False,
            },
        }
        self.policy = {
            "coverage": {
                "mode": "native_drop_last",
                "source_rows": 9,
                "expected_unique_source_seen": 8,
                "expected_effective_train_samples": 8,
                "expected_padding_rows": 0,
                "expected_dropped_rows": 1,
            }
        }
        self.trainer_path.write_text(
            "class Trainer:\n"
            "    def build(self):\n"
            "        self.train_dataloader = StatefulDataLoader(dataset=[], drop_last=True)\n",
            encoding="utf-8",
        )
        self._write_configs()

    def tearDown(self):
        self.temporary.cleanup()

    def _write_configs(self):
        self.project_path.write_text(yaml.safe_dump(self.project), encoding="utf-8")
        self.training_path.write_text(yaml.safe_dump(self.training), encoding="utf-8")
        self.policy_path.write_text(yaml.safe_dump(self.policy), encoding="utf-8")

    def _audit(self):
        return build_audit(
            self.project_path,
            self.training_path,
            self.policy_path,
            self.trainer_path,
        )

    def test_matching_native_drop_last_contract_passes(self):
        audit = self._audit()
        self.assertEqual(audit["status"], "PASS")
        self.assertTrue(all(audit["checks"].values()))
        self.assertEqual(audit["observed_contract"]["effective_train_samples"], 8)
        self.assertEqual(audit["observed_contract"]["dropped_rows_per_epoch"], 1)
        self.assertFalse(
            audit["runtime_receipt_requirement"]["exact_dropped_sample_id_known_statically"]
        )

    def test_old_full_coverage_contract_fails_closed(self):
        self.project["training_contract"].update(
            {
                "optimizer_steps": 3,
                "effective_train_samples": 9,
                "padding_rows": 3,
                "dropped_rows": 0,
                "require_full_coverage_sampler": True,
            }
        )
        self._write_configs()
        audit = self._audit()
        self.assertEqual(audit["status"], "FAIL")
        self.assertFalse(audit["checks"]["project_full_coverage_sampler_disabled"])
        self.assertFalse(audit["checks"]["project_dropped_rows_match_remainder"])

    def test_artifacts_are_written_with_hash_manifest(self):
        audit = self._audit()
