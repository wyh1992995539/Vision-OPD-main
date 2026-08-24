import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from PIL import Image

from eval.frozen_benchmark_data import (
    _append_suffix,
    _question_format,
    load_frozen_config,
    parse_benchmarks,
    run_frozen_preparation,
)


class FrozenBenchmarkProtocolTest(unittest.TestCase):
    def test_repository_config_is_frozen(self):
        _path, config = load_frozen_config("configs/benchmark_eval.yaml")
        self.assertEqual(config["protocol"]["status"], "frozen")
        self.assertEqual(config["protocol"]["protocol_revision"], 2)
        zoom = config["benchmarks"]["zoombench"]
        self.assertEqual(
            zoom["per_sample_dimension_labels"]["status"],
            "unavailable_in_frozen_official_snapshot",
        )
        self.assertNotIn("category_accuracy", zoom["metrics"])
        self.assertIn("category_accuracy", zoom["unsupported_metrics"])
        self.assertEqual(
            config["smoke"]["selection"]["zoombench"]["quotas"],
            {"multiple_choice": 12, "open_question": 4},
        )
        self.assertEqual(
            config["benchmarks"]["vstar"]["dataset_repo_id"],
            "craigwu/vstar_bench",
        )
        for item in config["benchmarks"].values():
            self.assertRegex(item["dataset_revision"], r"^[0-9a-f]{40}$")

    def test_benchmark_list_is_ordered_unique(self):
        self.assertEqual(
            parse_benchmarks("zoombench,mmstar,zoombench,vstar"),
            ["zoombench", "mmstar", "vstar"],
        )
        with self.assertRaisesRegex(ValueError, "unsupported"):
            parse_benchmarks("zoombench,not-a-benchmark")

    def test_vstar_suffix_is_idempotent(self):
        suffix = "Answer with the option's letter from the given choices directly."
        once = _append_suffix("Question?", suffix)
        twice = _append_suffix(once, suffix)
        self.assertEqual(once, twice)
        self.assertEqual(once.count(suffix), 1)

    def test_zoombench_question_format(self):
        query = "Question? (A) one (B) two"
        self.assertEqual(_question_format(query, "B"), "multiple_choice")
        self.assertEqual(_question_format("Question? A. one B. two", "B"), "multiple_choice")
        self.assertEqual(_question_format("What text is visible?", "OPENAI"), "open_question")

    def test_zoombench_reuses_frozen_local_parquet(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            repo_root = temp / "repo"
            config_dir = repo_root / "configs"
            config_dir.mkdir(parents=True)
            run_root = repo_root / "artifacts" / "runs" / "E-D5-001"
            data_root = temp / "benchmark_data"
            revision = "1" * 40
            config = {
                "protocol": {"status": "frozen"},
                "benchmarks": {
                    "zoombench": {
                        "dataset_repo_id": "inclusionAI/ZoomBench",
                        "dataset_revision": revision,
                        "split": "test",
                        "source_file": "data/test.parquet",
                        "expected_sample_count": 1,
                    },
                    "mmstar": {
                        "dataset_repo_id": "Lin-Chen/MMStar",
                        "dataset_revision": "2" * 40,
                        "split": "val",
                        "expected_sample_count": 1,
                    },
                    "vstar": {
                        "dataset_repo_id": "craigwu/vstar_bench",
                        "dataset_revision": "3" * 40,
                        "split": "test",
                        "expected_sample_count": 1,
                    },
                },
                "paths": {
                    "data_root": str(data_root),
                    "hf_cache": str(temp / "hf_cache"),
                    "run_root": str(run_root),
                    "dataset_manifest": str(run_root / "dataset_manifest.json"),
                    "raw_hash_manifest": str(run_root / "raw_data_sha256.txt"),
                    "converted_hash_manifest": str(run_root / "converted_data_sha256.txt"),
                },
            }
            config_path = config_dir / "benchmark_eval.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

            image_buffer = BytesIO()
            Image.new("RGB", (8, 8), "blue").save(image_buffer, format="PNG")
            image_bytes = image_buffer.getvalue()
            source = data_root / "raw" / "zoombench" / "data" / "test.parquet"
            source.parent.mkdir(parents=True)
            table = pa.Table.from_pylist([{
                "id": "zoom-1",
                "query": "Question? A. red B. blue",
                "response": "B",
                "bbox": [0.0, 1.0, 2.0, 3.0],
                "question_type": "mcq",
                "image": {"bytes": image_bytes, "path": None},
                "crop_image": {"bytes": image_bytes, "path": None},
            }])
            pq.write_table(table, source)
            marker = (
                data_root / "raw" / "zoombench" / ".cache"
                / "huggingface" / "trees" / f"{revision}.json"
            )
            marker.parent.mkdir(parents=True)
            marker.write_text("{}", encoding="utf-8")

            with patch("eval.frozen_benchmark_data.snapshot_download") as download:
                with patch("eval.frozen_benchmark_data.load_dataset") as remote_loader:
                    run_frozen_preparation(config_path, "zoombench")

            download.assert_not_called()
            remote_loader.assert_not_called()
            output = data_root / "converted" / "zoombench" / "zoombench.json"
            records = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(records[0]["question_format"], "multiple_choice")
            self.assertEqual(records[0]["official_question_type"], "mcq")
            self.assertEqual(records[0]["bbox"], [0.0, 1.0, 2.0, 3.0])
            self.assertEqual(records[0]["category"], "unavailable_official")
            self.assertTrue(Path(records[0]["crop_images"][0]).is_file())

    def test_download_receives_frozen_revision_and_writes_validated_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            repo_root = temp / "repo"
            config_dir = repo_root / "configs"
            config_dir.mkdir(parents=True)
            run_root = repo_root / "artifacts" / "runs" / "E-D5-001"
            data_root = temp / "benchmark_data"
            config = {
                "protocol": {"status": "frozen"},
                "benchmarks": {
                    "zoombench": {
                        "dataset_repo_id": "inclusionAI/ZoomBench",
                        "dataset_revision": "1" * 40,
                        "split": "test",
                        "expected_sample_count": 1,
                    },
                    "mmstar": {
                        "dataset_repo_id": "Lin-Chen/MMStar",
                        "dataset_revision": "2" * 40,
                        "split": "val",
                        "expected_sample_count": 1,
                    },
                    "vstar": {
                        "dataset_repo_id": "craigwu/vstar_bench",
                        "dataset_revision": "3" * 40,
                        "split": "test",
                        "expected_sample_count": 1,
                        "prompt_suffix": "Answer with the option's letter from the given choices directly.",
                    },
                },
                "paths": {
                    "data_root": str(data_root),
                    "hf_cache": str(temp / "hf_cache"),
                    "run_root": str(run_root),
                    "dataset_manifest": str(run_root / "dataset_manifest.json"),
                    "raw_hash_manifest": str(run_root / "raw_data_sha256.txt"),
                    "converted_hash_manifest": str(run_root / "converted_data_sha256.txt"),
                },
            }
            config_path = config_dir / "benchmark_eval.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            fake_dataset = [{
                "question_id": "q1",
                "text": "What color? (A) red (B) blue",
                "label": "A",
                "category": "direct_attributes",
                "image": Image.new("RGB", (8, 8), "red"),
            }]

            with patch("eval.frozen_benchmark_data.snapshot_download") as download:
                with patch("eval.frozen_benchmark_data.load_dataset", return_value=fake_dataset):
                    run_frozen_preparation(config_path, "vstar")

            download.assert_called_once_with(
                repo_id="craigwu/vstar_bench",
                repo_type="dataset",
                revision="3" * 40,
                local_dir=str(data_root / "raw" / "vstar"),
            )
            output = data_root / "converted" / "vstar" / "vstar.json"
            records = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["sample_uid"], "vstar:source_id:q1")
            self.assertEqual(records[0]["source_revision"], "3" * 40)
            validation = json.loads((run_root / "data_validation.json").read_text())
            self.assertEqual(validation["benchmarks"]["vstar"]["status"], "pass")


if __name__ == "__main__":
    unittest.main()
