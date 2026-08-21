import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.prepare_manual_qa import add_area_ratio, select_records
from scripts.render_manual_qa import build_html


class ManualQAToolsTest(unittest.TestCase):
    def make_records(self, root: Path):
        records = []
        for split in ("train", "eval", "retention"):
            for index in range(9):
                name = f"{split}-{index}"
                full_relative = f"images/{name}.png"
                crop_relative = f"teacher_images/{index:06d}_{name}.png"
                full = root / full_relative
                crop = root / crop_relative
                full.parent.mkdir(parents=True, exist_ok=True)
                crop.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (100, 100), "white").save(full)
                Image.new("RGB", (10, 10), "red").save(crop)
                records.append(
                    {
                        "sample_id": name,
                        "split": split,
                        "full_image_path": full_relative,
                        "crop_image_path": crop_relative,
                        "bbox": [0, 0, 10 + index * 5, 10 + index * 5],
                    }
                )
        return records

    def test_selection_is_deterministic_and_stratified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = self.make_records(root)
            add_area_ratio(records, root)
            first = select_records(records, seed=42, quotas={"train": 3, "eval": 3, "retention": 3})
            second = select_records(records, seed=42, quotas={"train": 3, "eval": 3, "retention": 3})
            self.assertEqual([r["sample_id"] for r in first], [r["sample_id"] for r in second])
            for split in ("train", "eval", "retention"):
                buckets = {r["_area_bucket"] for r in first if r["split"] == split}
                self.assertEqual(buckets, {"small", "medium", "large"})

    def test_html_escapes_dataset_text_and_references_local_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = []
            for index in range(30):
                full = root / "images" / f"{index}.png"
                crop = root / "teacher_images" / f"{index:06d}_{index}.png"
                full.parent.mkdir(parents=True, exist_ok=True)
                crop.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (10, 10), "white").save(full)
                Image.new("RGB", (5, 5), "red").save(crop)
                records.append(
                    {
                        "qa_index": index + 1,
                        "sample_id": f"sample-{index}",
                        "split": "train",
                        "area_bucket": "small",
                        "full_image_path": f"images/{index}.png",
                        "crop_image_path": f"teacher_images/{index:06d}_{index}.png",
                        "problem": "<script>alert(1)</script>",
                        "answer": "A",
                        "bbox": [0, 0, 5, 5],
                        "note": "",
                    }
                )
            rendered = build_html(records, root)
            self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", rendered)
            self.assertIn(full.as_uri(), rendered)
            self.assertIn("导出 manual_qa_30.jsonl", rendered)


if __name__ == "__main__":
    unittest.main()
