import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from PIL import Image

from scripts.check_benchmark_overlap import (
    ImageRef,
    Sample,
    apply_manual_decisions,
    detect_candidates,
    hamming_distance,
    normalize_question,
    phash64,
    run_audit,
)


class BenchmarkOverlapTest(unittest.TestCase):
    def test_question_normalization_is_frozen(self):
        left = "  WHAT\u3000Color is\n the Car?  "
        right = "what color is the car?"
        self.assertEqual(normalize_question(left), normalize_question(right))

    def test_hamming_distance_and_phash_are_deterministic(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "image.png"
            Image.new("RGB", (16, 16), "red").save(path)
            first = phash64(path)
            second = phash64(path)
            self.assertEqual(first, second)
            self.assertEqual(hamming_distance(first, second), 0)
            self.assertEqual(hamming_distance(0, 0b1011), 3)

    def test_candidate_classification_separates_exact_and_suspected(self):
        project = Sample(
            owner="project",
            dataset="vision_opd_project",
            split="train",
            sample_uid="project:1",
            source_id="source:1",
            question="What color?",
            normalized_question="what color?",
            images=[ImageRef("project", "vision_opd_project", "train", "project:1", "full", "/p.png")],
        )
        exact = Sample(
            owner="benchmark",
            dataset="zoombench",
            split="test",
            sample_uid="zoom:1",
            source_id="z1",
            question="WHAT COLOR?",
            normalized_question="what color?",
            images=[ImageRef("benchmark", "zoombench", "test", "zoom:1", "full", "/z1.png")],
        )
        suspected = Sample(
            owner="benchmark",
            dataset="zoombench",
            split="test",
            sample_uid="zoom:2",
            source_id="z2",
            question="Different question",
            normalized_question="different question",
            images=[ImageRef("benchmark", "zoombench", "test", "zoom:2", "full", "/z2.png")],
        )
        fingerprints = {
            "/p.png": {"sha256": "a" * 64, "phash_hex": "0000000000000000"},
            "/z1.png": {"sha256": "a" * 64, "phash_hex": "0000000000000000"},
            "/z2.png": {"sha256": "b" * 64, "phash_hex": "0000000000000001"},
        }
        candidates = detect_candidates(
            [project],
            [exact, suspected],
            fingerprints,
            phash_threshold=5,
            enable_sha256=True,
            enable_question=True,
            enable_phash=True,
        )
        by_uid = {item["benchmark_sample_uid"]: item for item in candidates}
        self.assertEqual(by_uid["zoom:1"]["review_status"], "confirmed_overlap")
        self.assertIn("exact_image_match", by_uid["zoom:1"]["match_types"])
        self.assertIn("exact_question_match", by_uid["zoom:1"]["match_types"])
        self.assertEqual(by_uid["zoom:2"]["review_status"], "unresolved")
        self.assertEqual(by_uid["zoom:2"]["minimum_phash_distance"], 1)

    def test_manual_decisions_are_validated_and_applied(self):
        candidates = [{
            "candidate_id": "overlap-1",
            "review_status": "unresolved",
        }]
        with tempfile.TemporaryDirectory() as temp_dir:
            decision_path = Path(temp_dir) / "manual_review_decisions.json"
            decision_path.write_text(json.dumps({
                "decisions": [{
                    "candidate_id": "overlap-1",
                    "review_status": "dismissed",
                    "reviewer": "test-reviewer",
                    "review_note": "Images are unrelated.",
                    "reviewed_at_utc": "2026-08-24T00:00:00Z",
                }],
            }), encoding="utf-8")
            apply_manual_decisions(candidates, decision_path)

        self.assertEqual(candidates[0]["review_status"], "dismissed")
        self.assertEqual(
            candidates[0]["manual_review"]["reviewer"],
            "test-reviewer",
        )

    def test_end_to_end_audit_writes_reproducible_evidence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            repo = temp / "repo"
            config_dir = repo / "configs"
            config_dir.mkdir(parents=True)
            data_root = temp / "benchmark_data"
            run_root = repo / "artifacts" / "runs" / "E-D5-001"
            project_image = temp / "project.png"
            benchmark_image = temp / "benchmark.png"
            Image.new("RGB", (16, 16), "blue").save(project_image)
            benchmark_image.write_bytes(project_image.read_bytes())

            project_parquet = temp / "train.parquet"
            table = pa.Table.from_pylist([{
                "images": [{"path": str(project_image)}],
                "bbox_images": [],
                "extra_info": {
                    "question": "How many objects?",
                    "answer": "2",
                    "provenance": {
                        "sample_id": "project-sample-1",
                        "source_id": "project-source-1",
                        "split": "train",
                    },
                },
            }])
            pq.write_table(table, project_parquet)

            converted = data_root / "converted" / "zoombench" / "zoombench.json"
            converted.parent.mkdir(parents=True)
            converted.write_text(json.dumps([{
                "sample_uid": "zoombench:source_id:z1",
                "source_id": "z1",
                "source_split": "test",
                "query": " HOW many\nobjects? ",
                "response": "2",
                "images": [str(benchmark_image)],
                "crop_images": [],
            }]), encoding="utf-8")

            config = {
                "paths": {
                    "data_root": str(data_root),
                    "run_root": str(run_root),
                },
                "overlap_audit": {
                    "project_splits": [str(project_parquet)],
                    "include_project_full_images": True,
                    "include_project_crop_images": True,
                    "checks": {
                        "file_sha256": {"enabled": True},
                        "normalized_question": {
                            "enabled": True,
                            "unicode_normalization": "NFKC",
                            "casefold": True,
                            "collapse_whitespace": True,
                        },
                        "perceptual_hash": {
                            "enabled": True,
                            "suspected_max_hamming_distance": 5,
                        },
                    },
                },
            }
            config_path = config_dir / "benchmark_eval.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

            report = run_audit(config_path, benchmarks=["zoombench"])
            self.assertEqual(report["decision_status"], "confirmed_overlap")
            self.assertEqual(report["confirmed_overlap_pair_count"], 1)
            output_dir = run_root / "overlap"
            self.assertTrue((output_dir / "overlap_report.json").is_file())
            self.assertTrue((output_dir / "overlap_report.md").is_file())
            self.assertTrue((output_dir / "manual_review.csv").is_file())
            candidates = [
                json.loads(line)
                for line in (output_dir / "overlap_candidates.jsonl").read_text().splitlines()
            ]
            self.assertEqual(candidates[0]["review_status"], "confirmed_overlap")
            self.assertIn("exact_image_match", candidates[0]["match_types"])
            self.assertIn("exact_question_match", candidates[0]["match_types"])


if __name__ == "__main__":
    unittest.main()

