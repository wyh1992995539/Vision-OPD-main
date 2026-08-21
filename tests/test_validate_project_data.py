import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.validate_project_data import validate_dataset


class ValidateProjectDataTest(unittest.TestCase):
    def make_dataset(self, root: Path, bbox=None):
        manifest_dir = root / "manifests"
        subset_root = root / "subset"
        (subset_root / "images").mkdir(parents=True)
        (subset_root / "teacher_images").mkdir(parents=True)
        manifest_dir.mkdir()

        Image.new("RGB", (100, 80), "white").save(subset_root / "images" / "abc.png")
        Image.new("RGB", (20, 20), "red").save(
            subset_root / "teacher_images" / "000001_abc.png"
        )
        record = {
            "sample_id": "sample-1",
            "group_id": "group-1",
            "split": "train",
            "full_image_path": "images/abc.png",
            "crop_image_path": "teacher_images/000001_abc.png",
            "bbox": bbox if bbox is not None else [10, 10, 30, 30],
            "problem": "<image>\nQuestion?\nA. x\nB. y",
            "answer": "A",
        }
        (manifest_dir / "train_1.jsonl").write_text(
            json.dumps(record) + "\n", encoding="utf-8"
        )
        return manifest_dir, subset_root

    def test_valid_dataset_passes_and_hashes_both_images(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest_dir, subset_root = self.make_dataset(Path(directory))
            validation, stats, hashes = validate_dataset(
                manifest_dir,
                subset_root,
                workers=1,
                expected_splits={"train_1.jsonl": ("train", 1)},
            )
            self.assertEqual(validation["status"], "PASS")
            self.assertEqual(validation["images_checked"], 2)
            self.assertEqual(stats["student_images"]["count"], 1)
            self.assertEqual(len(hashes), 2)

    def test_out_of_bounds_bbox_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest_dir, subset_root = self.make_dataset(
                Path(directory), bbox=[10, 10, 101, 30]
            )
            validation, _, _ = validate_dataset(
                manifest_dir,
                subset_root,
                workers=1,
                expected_splits={"train_1.jsonl": ("train", 1)},
            )
            self.assertEqual(validation["status"], "FAIL")
            self.assertEqual(validation["issue_counts"], {"bbox_out_of_bounds": 1})


if __name__ == "__main__":
    unittest.main()
