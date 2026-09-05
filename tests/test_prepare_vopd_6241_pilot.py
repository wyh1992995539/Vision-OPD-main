import unittest

from scripts.build_day11_static_gate import compact_preflight
from scripts.prepare_vopd_6241_pilot import select_samples


class PrepareVopd6241PilotTest(unittest.TestCase):
    def fixture(self):
        historical = {f"h-{index}" for index in range(8)}
        new = {f"n-{index}" for index in range(16)}
        sample_ids = sorted(historical | new)
        lengths = {
            sample_id: {
                "student": 1000 + index * 11,
                "teacher": 2000 + ((index * 7) % 23),
            }
            for index, sample_id in enumerate(sample_ids)
        }
        return sample_ids, lengths, historical

    def test_selection_is_unique_deterministic_and_nested(self):
        sample_ids, lengths, historical = self.fixture()
        selected_8 = select_samples(
            sample_ids, lengths, historical, count=8, seed=42
        )
        selected_16 = select_samples(
            sample_ids, lengths, historical, count=16, seed=42
        )
        ids_8 = [sample_id for sample_id, _reason in selected_8]
        ids_16 = [sample_id for sample_id, _reason in selected_16]
        self.assertEqual(ids_8, ids_16[:8])
        self.assertEqual(len(set(ids_16)), 16)
        self.assertTrue(set(ids_16) & historical)
        self.assertTrue(set(ids_16) - historical)
        self.assertEqual(
            selected_16,
            select_samples(sample_ids, lengths, historical, count=16, seed=42),
        )

    def test_selection_rejects_prompt_audit_mismatch(self):
        sample_ids, lengths, historical = self.fixture()
        lengths.pop(sample_ids[0])
        with self.assertRaisesRegex(ValueError, "prompt audit/source mismatch"):
            select_samples(sample_ids, lengths, historical, count=8, seed=42)


if __name__ == "__main__":
    unittest.main()
