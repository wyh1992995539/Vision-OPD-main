import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from scripts.prepare_grpo_data import (
    DATA_SOURCE,
    REWARD_ROUTE,
    convert_grpo_dataset,
)


class PrepareGrpoDataTest(unittest.TestCase):
    def make_source(self, root: Path, *, second_gold: str = "A") -> tuple[Path, Path]:
        image_root = root / "images"
        image_root.mkdir()
        rows = []
        for index, gold in enumerate(("D", second_gold)):
            image_path = image_root / f"image-{index}.png"
            Image.new("RGB", (12, 12), "white").save(image_path)
            sample_id = f"sample-{index}"
            rows.append(
                {
                    "data_source": "zwz_rl_vqa_bbox_teacher",
                    "prompt": [
                        {
                            "role": "user",
                            "content": (
                                "<image>\nWhat is shown?\n\n"
                                "A. alpha\nB. beta\nC. gamma\nD. delta\n\n"
                                "Answer with the option's letter from the given choices."
                            ),
                        }
                    ],
                    "images": [{"path": image_path.as_posix()}],
                    "bbox_images": [{"path": f"/unused/crop-{index}.png"}],
                    "ability": "visual_question_answering",
                    "reward_model": {"style": "none", "ground_truth": gold},
                    "extra_info": {
                        "answer": gold,
                        "question": "What is shown?",
                        "provenance": {
                            "sample_id": sample_id,
                            "source_id": f"source-{index}",
                            "source_row": index,
                            "group_id": f"group-{index}",
                            "question_type": "multiple_choice",
                        },
                    },
                }
            )
        source = root / "train.parquet"
        pq.write_table(pa.Table.from_pylist(rows), source)
        source_qa = root / "source_qa.json"
        source_qa.write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "record_count": 2,
                    "images_checked": 4,
                    "issue_count": 0,
                }
            ),
            encoding="utf-8",
        )
        return source, source_qa

    def test_conversion_preserves_policy_inputs_and_removes_teacher_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, source_qa = self.make_source(root)
            output = root / "grpo.parquet"
            run_dir = root / "evidence"
            report = convert_grpo_dataset(
                source,
                output,
                run_dir,
                expected_rows=2,
                source_qa=source_qa,
                command="test conversion",
            )

            self.assertEqual(report["status"], "PASS")
            self.assertFalse(report["pilot_authorized"])
            rows = pq.read_table(output).to_pylist()
            source_rows = pq.read_table(source).to_pylist()
            self.assertEqual(len(rows), 2)
            for index, row in enumerate(rows):
                self.assertEqual(row["data_source"], DATA_SOURCE)
                self.assertEqual(row["prompt"], source_rows[index]["prompt"])
                self.assertEqual(row["images"], source_rows[index]["images"])
                self.assertNotIn("bbox_images", row)
                self.assertNotIn("answer", row["extra_info"])
                self.assertEqual(row["extra_info"]["index"], index)
                self.assertEqual(row["extra_info"]["reward_route"], REWARD_ROUTE)
                self.assertEqual(row["extra_info"]["valid_options"], ["A", "B", "C", "D"])
                self.assertEqual(row["reward_model"]["style"], "rule")
            route_rows = [
                json.loads(line)
                for line in (run_dir / "reward_routes.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(route_rows), 2)
            self.assertTrue(all(item["data_scorable"] for item in route_rows))
            self.assertTrue((run_dir / "source_binding.json").is_file())
            self.assertTrue((run_dir / "grpo_parquet_sha256.txt").is_file())

    def test_invalid_gold_stops_before_output_is_written(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, source_qa = self.make_source(root, second_gold="E")
            output = root / "grpo.parquet"
            with self.assertRaisesRegex(ValueError, "ground truth must be one of"):
                convert_grpo_dataset(
                    source,
                    output,
                    root / "evidence",
                    expected_rows=2,
                    source_qa=source_qa,
                )
            self.assertFalse(output.exists())

    def test_existing_output_requires_explicit_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, source_qa = self.make_source(root)
            output = root / "grpo.parquet"
            run_dir = root / "evidence"
            convert_grpo_dataset(
                source,
                output,
                run_dir,
                expected_rows=2,
                source_qa=source_qa,
            )
            with self.assertRaisesRegex(FileExistsError, "use --overwrite"):
                convert_grpo_dataset(
                    source,
                    output,
                    run_dir,
                    expected_rows=2,
                    source_qa=source_qa,
                )


if __name__ == "__main__":
    unittest.main()
