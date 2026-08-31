import json
import tempfile
import unittest
from pathlib import Path

from scripts.vopd_day8_reload import (
    SELECTION_ALGORITHM,
    stable_key,
    validate_merged_model,
    validate_static,
    verify_predictions,
)


class VopdDay8ReloadTest(unittest.TestCase):
    repository_root = Path(__file__).resolve().parents[1]
    config_path = repository_root / "configs/vopd_day8_reload.yaml"

    def test_repository_reload_config_passes_static_preflight(self):
        summary = validate_static(self.config_path, self.repository_root)
        self.assertEqual(summary["status"], "PASS", summary["errors"])
        self.assertEqual(len(summary["sample_ids"]), 5)
        self.assertTrue(summary["checks"]["source_and_merged_paths_are_separate"])

    def test_frozen_ids_are_in_stable_hash_order(self):
        import yaml

        config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        sample_ids = config["evaluation"]["sample_ids"]
        seed = config["experiment"]["seed"]
        self.assertEqual(config["evaluation"]["selection_algorithm"], SELECTION_ALGORITHM)
        self.assertEqual(sample_ids, sorted(sample_ids, key=lambda value: stable_key(seed, value)))

    def test_merged_model_contract_requires_weights_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertTrue(validate_merged_model(root))
            (root / "config.json").write_text("{}\n", encoding="utf-8")
            (root / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
            (root / "model.safetensors").write_bytes(b"weights")
            self.assertEqual(validate_merged_model(root), [])

    def test_prediction_gate_rejects_errors_and_empty_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ids = ["sample-1", "sample-2"]
            rows = [
                {
                    "sample_id": "sample-1",
                    "raw_prediction": "A",
                    "response_token_count": 1,
                    "finish_reason": "stop",
                    "inference_error": None,
                },
                {
                    "sample_id": "sample-2",
                    "raw_prediction": "",
                    "response_token_count": 0,
                    "finish_reason": "error",
                    "inference_error": "RuntimeError: failed",
                },
            ]
            (root / "predictions.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            (root / "summary.json").write_text(
                json.dumps({"total": 2, "unique_sample_ids": 2}) + "\n", encoding="utf-8"
            )
            result = verify_predictions(root, ids)
            self.assertEqual(result["status"], "FAIL")
            self.assertEqual(result["inference_error_count"], 1)
            self.assertEqual(result["nonempty_response_count"], 1)


if __name__ == "__main__":
    unittest.main()
