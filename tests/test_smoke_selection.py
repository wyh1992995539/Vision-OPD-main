import json
import tempfile
import unittest
from pathlib import Path

import yaml
from PIL import Image

from scripts.select_benchmark_smoke import (
    build_manifest,
    rank_sample,
    select_rows,
    write_manifest,
)


class SmokeSelectionTest(unittest.TestCase):
    def test_rank_is_seeded_and_deterministic(self):
        self.assertEqual(
            rank_sample(42, "mmstar", "mmstar:source_id:1"),
            rank_sample(42, "mmstar", "mmstar:source_id:1"),
        )
        self.assertNotEqual(
            rank_sample(42, "mmstar", "mmstar:source_id:1"),
            rank_sample(43, "mmstar", "mmstar:source_id:1"),
        )

    def test_select_rows_uses_explicit_stratum_quota(self):
        rows = [
            {"sample_uid": "a", "category": "one"},
            {"sample_uid": "b", "category": "one"},
            {"sample_uid": "c", "category": "two"},
        ]
        selected = select_rows(
            rows,
            benchmark="mmstar",
            seed=42,
            stratum_key="category",
            quotas={"one": 1, "two": 1},
        )
        self.assertEqual({row["category"] for row in selected}, {"one", "two"})
        self.assertEqual(len(selected), 2)

    def test_manifest_is_reproducible_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            repo = temp / "repo"
            config_dir = repo / "configs"
            config_dir.mkdir(parents=True)
            data_root = temp / "data"
            overlap_dir = repo / "artifacts" / "overlap"
            image = temp / "image.png"
            Image.new("RGB", (8, 8), "blue").save(image)

            benchmarks = {
                "zoombench": {
                    "expected_sample_count": 2,
                    "dataset_revision": "a" * 40,
                },
                "mmstar": {
                    "expected_sample_count": 2,
                    "dataset_revision": "b" * 40,
                },
                "vstar": {
                    "expected_sample_count": 2,
                    "dataset_revision": "c" * 40,
                },
            }
            rows_by_name = {
                "zoombench": [("multiple_choice", "z1"), ("open_question", "z2")],
                "mmstar": [("one", "m1"), ("two", "m2")],
                "vstar": [("one", "v1"), ("two", "v2")],
            }
            for name, rows in rows_by_name.items():
                output = data_root / "converted" / name / f"{name}.json"
                output.parent.mkdir(parents=True)
                key = "question_format" if name == "zoombench" else "category"
                payload = [{
                    "sample_uid": f"{name}:source_id:{source_id}",
                    "source_id": source_id,
                    "source_revision": benchmarks[name]["dataset_revision"],
                    "query": "Question?",
                    "response": "A",
                    key: stratum,
                    "images": [str(image)],
                    "crop_images": [],
                    "image_sha256": "image-hash",
                } for stratum, source_id in rows]
                output.write_text(json.dumps(payload), encoding="utf-8")

            config = {
                "protocol": {"experiment_id": "test", "protocol_revision": 3},
                "paths": {
                    "data_root": str(data_root),
                    "overlap_dir": str(overlap_dir),
                },
                "benchmarks": benchmarks,
                "smoke": {
                    "selection_seed": 42,
                    "samples_per_benchmark": 2,
                    "selection_method": "sha256_ranked_stratified",
                    "selection_rank_input": "{selection_seed}:{benchmark}:{sample_uid}",
                    "selection_rank_order": "ascending_hex_digest",
                    "manifest": "artifacts/smoke_selection.json",
                    "selection": {
                        "zoombench": {
                            "stratify_by": "question_format",
                            "quotas": {"multiple_choice": 1, "open_question": 1},
                        },
                        "mmstar": {
                            "stratify_by": "category",
                            "quotas": {"one": 1, "two": 1},
                        },
                        "vstar": {
                            "stratify_by": "category",
                            "quotas": {"one": 1, "two": 1},
                        },
                    },
                },
            }
            config_path = config_dir / "benchmark_eval.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

            first = build_manifest(config_path, benchmarks=["zoombench", "mmstar", "vstar"])
            second = build_manifest(config_path, benchmarks=["zoombench", "mmstar", "vstar"])
            self.assertEqual(
                first["benchmarks"]["zoombench"]["sample_uids"],
                second["benchmarks"]["zoombench"]["sample_uids"],
            )
            output = repo / "artifacts" / "smoke_selection.json"
            hash_path = write_manifest(first, output, force=False)
            self.assertTrue(hash_path.is_file())
            with self.assertRaises(FileExistsError):
                write_manifest(first, output, force=False)


if __name__ == "__main__":
    unittest.main()
