import json
import tempfile
import unittest
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

from scripts.build_project_parquet import build_project_parquets


class BuildProjectParquetTest(unittest.TestCase):
    splits = {
        "train": ("train_2.jsonl", "train_2.parquet", 2),
        "eval": ("eval_1.jsonl", "eval_1.parquet", 1),
        "retention": ("retention_1.jsonl", "retention_1.parquet", 1),
    }

    def make_dataset(self, root: Path) -> tuple[Path, Path]:
        data_root = root / "data"
        manifest_dir = data_root / "manifests"
        (data_root / "images").mkdir(parents=True)
        (data_root / "teacher_images").mkdir()
        manifest_dir.mkdir()

        index = 0
        for split, (manifest_name, _output_name, count) in self.splits.items():
            rows = []
            for _ in range(count):
                index += 1
                full_name = f"image-{index}.png"
                crop_name = f"{index:06d}_image-{index}.png"
                Image.new("RGB", (20, 20), "white").save(data_root / "images" / full_name)
                Image.new("RGB", (10, 10), "red").save(
                    data_root / "teacher_images" / crop_name
                )
                rows.append(
                    {
                        "sample_id": f"sample-{index}",
                        "source_id": f"source-{index}",
                        "source_row": index,
                        "group_id": f"group-{index}",
                        "split": split,
                        "original_image_path": f"original_images/image-{index}.png",
                        "full_image_path": f"images/{full_name}",
                        "crop_image_path": f"teacher_images/{crop_name}",
                        "bbox": [0, 0, 10, 10],
                        "problem": "<image>\nWhat is shown?",
                        "question": "What is shown?",
                        "answer": "object",
                        "question_type": "short_answer",
                    }
                )
            (manifest_dir / manifest_name).write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
        return manifest_dir, data_root

    def test_builds_three_compatible_parquets_and_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_dir, data_root = self.make_dataset(root)
            report = build_project_parquets(
                manifest_dir,
                data_root,
                data_root,
                expected_splits=self.splits,
            )

            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["total_rows"], 4)
            self.assertTrue((manifest_dir / "parquet_build_report.json").is_file())
            self.assertEqual(
                len((manifest_dir / "parquet_sha256.txt").read_text().splitlines()), 3
            )

            table = pq.read_table(data_root / "train_2.parquet")
            self.assertEqual(table.num_rows, 2)
            row = table.to_pylist()[0]
            self.assertEqual(row["data_source"], "zwz_rl_vqa_bbox_teacher")
            self.assertEqual(row["prompt"][0]["role"], "user")
            self.assertTrue(Path(row["images"][0]["path"]).is_absolute())
            self.assertTrue(Path(row["bbox_images"][0]["path"]).is_file())
            self.assertEqual(row["extra_info"]["provenance"]["split"], "train")

    def test_rejects_windows_manifest_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_dir, data_root = self.make_dataset(root)
            manifest_path = manifest_dir / "train_2.jsonl"
            rows = [json.loads(line) for line in manifest_path.read_text().splitlines()]
            rows[0]["full_image_path"] = r"E:\VisionOPD-subset\images\image-1.png"
            manifest_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "Windows path"):
                build_project_parquets(
                    manifest_dir,
                    data_root,
                    data_root,
                    expected_splits=self.splits,
                )

    def test_rejects_cross_split_group_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_dir, data_root = self.make_dataset(root)
            train_record = json.loads(
                (manifest_dir / "train_2.jsonl").read_text().splitlines()[0]
            )
            eval_path = manifest_dir / "eval_1.jsonl"
            eval_record = json.loads(eval_path.read_text())
            eval_record["group_id"] = train_record["group_id"]
            eval_path.write_text(json.dumps(eval_record) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Duplicate group_id"):
                build_project_parquets(
                    manifest_dir,
                    data_root,
                    data_root,
                    expected_splits=self.splits,
                )


if __name__ == "__main__":
    unittest.main()
