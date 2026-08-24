import json
import tempfile
import unittest
from pathlib import Path

from scripts.freeze_budget_inputs import write_frozen


class FreezeBudgetInputsTest(unittest.TestCase):
    def test_freeze_refuses_overwrite_and_writes_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "budget_inputs.json"
            document = {"schema_version": 1, "inputs": {"price": None}}
            hash_path = write_frozen(output, document, force=False)
            self.assertTrue(hash_path.is_file())
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), document)
            with self.assertRaises(FileExistsError):
                write_frozen(output, document, force=False)


if __name__ == "__main__":
    unittest.main()

