import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts.day6_preflight import canonical_lf_sha256, controlled_cases


class Day6PreflightTest(unittest.TestCase):
    def test_canonical_lf_hash_is_cross_platform_stable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lf = root / "lf.txt"
            crlf = root / "crlf.txt"
            lf.write_bytes(b"a\nb\n")
            crlf.write_bytes(b"a\r\nb\r\n")
            expected = hashlib.sha256(b"a\nb\n").hexdigest()
            self.assertEqual(canonical_lf_sha256(lf), expected)
            self.assertEqual(canonical_lf_sha256(crlf), expected)

    def test_controlled_calibration_has_32_balanced_records(self):
        dataset = [
            {
                "question_format": "open_question",
                "response": str(index + 1),
                "sample_uid": f"z{index}",
                "query": f"How many objects are shown in sample {index}?",
            }
            for index in range(8)
        ]
        rows = controlled_cases(dataset)
        self.assertEqual(len(rows), 32)
        self.assertEqual(len({row["calibration_id"] for row in rows}), 32)
        self.assertTrue(all(row["human_label"] is None for row in rows))
        self.assertEqual(sum(row["coverage_type"] == "deterministic_numeric" for row in rows), 8)
        self.assertEqual(sum(row["coverage_type"] == "semantic_equivalent" for row in rows), 8)


if __name__ == "__main__":
    unittest.main()
