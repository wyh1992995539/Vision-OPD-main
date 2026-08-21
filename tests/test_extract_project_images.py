import gzip
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from scripts.extract_project_images import ConcatenatedReader, load_targets, scan_or_extract


class ExtractProjectImagesTest(unittest.TestCase):
    def test_concatenated_reader_reconstructs_split_gzip_tar(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = io.BytesIO()
            with tarfile.open(fileobj=payload, mode="w") as archive:
                content = b"selected-image"
                member = tarfile.TarInfo("./selected.png")
                member.size = len(content)
                archive.addfile(member, io.BytesIO(content))

            compressed = gzip.compress(payload.getvalue())
            split = len(compressed) // 2
            parts = [root / "part00", root / "part01"]
            parts[0].write_bytes(compressed[:split])
            parts[1].write_bytes(compressed[split:])

            destination = root / "output"
            with ConcatenatedReader(parts) as raw_stream:
                with io.BufferedReader(raw_stream) as buffered:
                    with tarfile.open(fileobj=buffered, mode="r|gz") as archive:
                        result = scan_or_extract(
                            archive,
                            {"selected.png"},
                            destination,
                            label="test",
                            dry_run=False,
                            overwrite=False,
                        )

            self.assertEqual(result["missing"], [])
            self.assertEqual((destination / "selected.png").read_bytes(), b"selected-image")

    def test_manifest_counts_and_paths_are_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original_sizes = dict(__import__("scripts.extract_project_images", fromlist=["EXPECTED_SPLIT_SIZES"]).EXPECTED_SPLIT_SIZES)
            module = __import__("scripts.extract_project_images", fromlist=["EXPECTED_SPLIT_SIZES"])
            try:
                module.EXPECTED_SPLIT_SIZES = {"tiny.jsonl": 1}
                record = {
                    "sample_id": "sample-1",
                    "full_image_path": "images/full.png",
                    "crop_image_path": "teacher_images/crop.png",
                }
                (root / "tiny.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
                full, crop, total = load_targets(root)
                self.assertEqual(full, {"full.png"})
                self.assertEqual(crop, {"crop.png"})
                self.assertEqual(total, 1)
            finally:
                module.EXPECTED_SPLIT_SIZES = original_sizes


if __name__ == "__main__":
    unittest.main()
