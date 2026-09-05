import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/benchmark_overlap_6241.yaml"


class Overlap6241ConfigTest(unittest.TestCase):
    def setUp(self):
        self.config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    def test_only_full_train_6241_is_in_scope(self):
        audit = self.config["overlap_audit"]
        self.assertEqual(
            audit["project_splits"],
            ["/root/autodl-tmp/data/vision_opd_6241/train_6241.parquet"],
        )
        self.assertEqual(audit["expected_project_sample_count"], 6241)
        self.assertNotIn("vision_opd_1024", str(audit))

    def test_all_paper_aligned_r3_benchmarks_are_frozen(self):
        audit = self.config["overlap_audit"]
        self.assertTrue(audit["require_all_benchmarks"])
        self.assertEqual(audit["fingerprint_workers"], 16)
        self.assertEqual(
            audit["expected_benchmark_sample_counts"],
            {"zoombench": 845, "mmstar": 1500, "vstar": 191},
        )
        self.assertEqual(
            set(audit["expected_benchmark_sha256"]),
            {"zoombench", "mmstar", "vstar"},
        )
        self.assertEqual(
            self.config["paths"]["data_root"],
            "/root/autodl-tmp/benchmark_data/paper_aligned",
        )

    def test_all_three_overlap_methods_are_enabled(self):
        checks = self.config["overlap_audit"]["checks"]
        self.assertTrue(checks["file_sha256"]["enabled"])
        self.assertTrue(checks["normalized_question"]["enabled"])
        self.assertTrue(checks["perceptual_hash"]["enabled"])
        self.assertEqual(checks["perceptual_hash"]["hash_size"], 8)
        self.assertEqual(checks["perceptual_hash"]["suspected_max_hamming_distance"], 5)

    def test_official_samples_are_never_silently_removed(self):
        policy = self.config["overlap_audit"]["on_overlap"]
        self.assertTrue(policy["preserve_official_test_samples"])
        self.assertTrue(policy["report_official_full_score"])
        self.assertTrue(policy["report_deduplicated_diagnostic_separately"])
        self.assertTrue(policy["forbid_claiming_fully_independent_test"])


if __name__ == "__main__":
    unittest.main()
