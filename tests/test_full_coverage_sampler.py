import json
import tempfile
import unittest
from pathlib import Path
import importlib.util


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    import sys
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ROOT = Path(__file__).resolve().parents[1]
sampler_module = load_module(
    "full_coverage_sampler_standalone", ROOT / "verl/utils/dataset/full_coverage_sampler.py"
)
receipt_module = load_module(
    "full_coverage_receipt_standalone", ROOT / "verl/utils/dataset/full_coverage_receipt.py"
)
FullCoveragePaddingSampler = sampler_module.FullCoveragePaddingSampler
build_epoch_references = sampler_module.build_epoch_references
update_coverage_receipt = receipt_module.update_coverage_receipt


class FullCoverageSamplerTest(unittest.TestCase):
    def test_6241_contract_has_one_real_tail_and_seven_padding_rows(self):
        refs = build_epoch_references(6241, multiple=8, shuffle=False, seed=42, epoch=0)
        self.assertEqual(len(refs), 6248)
        self.assertEqual(sum(ref.sample_weight for ref in refs), 6241)
        self.assertEqual(sum(ref.is_padding for ref in refs), 7)
        self.assertEqual(sum(not ref.is_padding for ref in refs[-8:]), 1)
        self.assertEqual({ref.index for ref in refs if not ref.is_padding}, set(range(6241)))

    def test_shuffle_is_deterministic_and_epoch_state_is_restorable(self):
        first = FullCoveragePaddingSampler(range(17), multiple=8, shuffle=True, seed=42)
        epoch0 = list(first)
        state = first.state_dict()
        epoch1 = list(first)
        restored = FullCoveragePaddingSampler(range(17), multiple=8, shuffle=True, seed=42)
        restored.load_state_dict(state)
        self.assertEqual(epoch1, list(restored))
        self.assertNotEqual(epoch0[:17], epoch1[:17])

    def test_two_card_shards_keep_fixed_width_and_weighted_total(self):
        refs = build_epoch_references(6241, multiple=8, shuffle=True, seed=42, epoch=0)
        batches = [refs[index:index + 8] for index in range(0, len(refs), 8)]
        self.assertEqual(len(batches), 781)
        self.assertTrue(all(len(batch[:4]) == len(batch[4:]) == 4 for batch in batches))
        self.assertEqual(sum(ref.sample_weight for batch in batches for ref in batch), 6241)

    def test_receipt_rejects_duplicate_real_rows_and_passes_complete_run(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            infos1 = [{"provenance": {"sample_id": "a"}}, {"provenance": {"sample_id": "b"}}]
            update_coverage_receipt(
                path, extra_infos=infos1, sample_weights=[1.0, 1.0],
                global_step=1, expected_unique_samples=3, expected_padding_rows=1, final_step=False,
            )
            infos2 = [{"provenance": {"sample_id": "c"}}, {"provenance": {"sample_id": "a"}}]
            receipt = update_coverage_receipt(
                path, extra_infos=infos2, sample_weights=[1.0, 0.0],
                global_step=2, expected_unique_samples=3, expected_padding_rows=1, final_step=True,
            )
            self.assertEqual(receipt["status"], "PASS")
            self.assertEqual(json.loads(path.read_text())["dropped_rows"], 0)


if __name__ == "__main__":
    unittest.main()
