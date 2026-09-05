import copy
import hashlib
import tempfile
import unittest
from pathlib import Path

import pyarrow.parquet as pq

from scripts.generate_cached_prefix import (
    collect_audit_hashes,
    encode_cached_response,
    enforce_response_token_limit,
    extract_train_sample,
    smoke_paths,
    validate_cached_protocol,
    validate_cached_records,
    write_cache_artifacts,
)


class FakeTokenizer:
    eos_token_id = 99

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(character) for character in text]


class CachedPrefixContractTest(unittest.TestCase):
    @staticmethod
    def sha256(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def make_audit_fixture(self, root):
        model = root / "model"
        model.mkdir()
        weight = model / "model.safetensors"
        weight.write_bytes(b"frozen-model-weights")
        manifest = root / "base_model_sha256.txt"
        manifest.write_text(
            f"{self.sha256(weight)}  {weight.as_posix()}\n", encoding="utf-8"
        )
        template = root / "template.jinja"
        template.write_text("{{ messages }}\n", encoding="utf-8")
        input_parquet = root / "train.parquet"
        input_parquet.write_bytes(b"frozen-parquet")
        config = {
            "experiment": {
                "base_model_path": model.as_posix(),
                "base_model_sha256_file": manifest.as_posix(),
                "base_model_sha256_file_sha256": self.sha256(manifest),
            },
            "shared": {
                "chat_template": {
                    "file": template.as_posix(),
                    "sha256": self.sha256(template),
                }
            },
            "cached_prefix": {
                "input_parquet": input_parquet.as_posix(),
                "input_parquet_sha256": self.sha256(input_parquet),
            },
        }
        return config, model, manifest, template, input_parquet, weight

    def make_row(self, split="train"):
        return {
            "prompt": [{"role": "user", "content": "<image>\nQuestion?"}],
            "images": [{"path": "/data/images/sample.png"}],
            "bbox_images": [{"path": "/data/teacher_images/sample.png"}],
            "reward_model": {"style": "none", "ground_truth": "B"},
            "extra_info": {
                "answer": "B",
                "provenance": {
                    "sample_id": "sample-1",
                    "source_id": "source-1",
                    "split": split,
                },
            },
        }

    def test_extracts_only_allowed_generation_fields(self):
        sample = extract_train_sample(self.make_row())
        self.assertEqual(sample["sample_id"], "sample-1")
        self.assertEqual(sample["split"], "train")
        self.assertEqual(sample["image_path"].as_posix(), "/data/images/sample.png")
        self.assertNotIn("bbox_images", sample)
        self.assertNotIn("ground_truth", sample)
        self.assertNotIn("answer", sample)

    def test_rejects_non_train_rows(self):
        with self.assertRaisesRegex(ValueError, "only accepts train"):
            extract_train_sample(self.make_row(split="eval"))

    def test_protocol_requires_sampled_online_matched_generation(self):
        config = {
            "shared": {"student_image_key": "images"},
            "cached_prefix": {
                "generation": {
                    "do_sample": True,
                    "temperature": 1.0,
                    "num_return_sequences": 1,
                }
            },
        }
        _cached, generation = validate_cached_protocol(config)
        self.assertEqual(generation["temperature"], 1.0)
        config["cached_prefix"]["generation"]["temperature"] = 0.0
        with self.assertRaisesRegex(ValueError, "temperature"):
            validate_cached_protocol(config)

    def test_retokenizes_and_appends_eos_only_for_stop(self):
        tokenizer = FakeTokenizer()
        token_ids, appended = encode_cached_response(tokenizer, "A", "stop", True)
        self.assertEqual(token_ids, [ord("A"), 99])
        self.assertTrue(appended)
        token_ids, appended = encode_cached_response(tokenizer, "A", "length", True)
        self.assertEqual(token_ids, [ord("A")])
        self.assertFalse(appended)
    def test_caps_reencoded_ids_at_frozen_response_limit(self):
        token_ids, eos_appended, trimmed = enforce_response_token_limit(
            [1, 2, 99], True, 2
        )
        self.assertEqual(token_ids, [1, 2])
        self.assertFalse(eos_appended)
        self.assertTrue(trimmed)

    def test_validation_rejects_ids_above_frozen_response_limit(self):
        bad = self.make_record("one")
        bad["response_token_ids"] = [1] * 257
        with self.assertRaisesRegex(ValueError, "overlength_token_ids"):
            validate_cached_records([bad], ["one"], max_response_tokens=256)


    def make_record(self, sample_id):
        return {
            "sample_id": sample_id,
            "raw_response_text": "A",
            "response_token_ids": [1, 99],
            "finish_reason": "stop",
        }

    def test_validation_requires_exact_complete_cache(self):
        report = validate_cached_records(
            [self.make_record("one"), self.make_record("two")], ["one", "two"]
        )
        self.assertEqual(report["status"], "PASS")

    def test_validation_preserves_truncated_prefix_at_frozen_limit(self):
        truncated = self.make_record("one")
        truncated["finish_reason"] = "length"
        truncated["response_token_ids"] = [1] * 256
        report = validate_cached_records(
            [truncated], ["one"], max_response_tokens=256
        )
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["truncated_responses"], 1)
        self.assertEqual(
            report["truncation_policy"], "preserve_at_frozen_max_new_tokens"
        )

    def test_validation_still_rejects_empty_prefix(self):
        bad = self.make_record("one")
        bad["raw_response_text"] = ""
        bad["response_token_ids"] = []
        with self.assertRaisesRegex(ValueError, "validation failed"):
            validate_cached_records([bad], ["one"])

    def test_smoke_uses_separate_paths(self):
        output = Path("/data/cached_prefix_base_1024.parquet")
        smoke, report, hashes = smoke_paths(output, 8)
        self.assertEqual(smoke.name, "cached_prefix_base_1024_smoke_8.parquet")
        self.assertEqual(report.name, "cached_prefix_report_smoke_8.json")
        self.assertEqual(hashes.name, "cached_prefix_sha256_smoke_8.txt")

    def test_writes_parquet_report_and_hash(self):
        records = [self.make_record("one")]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "cache.parquet"
            report_path = root / "report.json"
            hash_path = root / "sha256.txt"
            final_report = write_cache_artifacts(
                output,
                records,
                report_path,
                hash_path,
                {"status": "PASS"},
            )
            self.assertTrue(output.is_file())
            self.assertTrue(report_path.is_file())
            self.assertTrue(hash_path.is_file())
            self.assertEqual(final_report["status"], "PASS")
            self.assertEqual(pq.read_table(output).num_rows, 1)
            self.assertIn(final_report["output_sha256"], hash_path.read_text())

    def test_collects_and_validates_all_frozen_input_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _model, manifest, template, input_parquet, _weight = (
                self.make_audit_fixture(root)
            )
            audit = collect_audit_hashes(config, root)
            self.assertEqual(audit["input_parquet_sha256"], self.sha256(input_parquet))
            self.assertEqual(audit["chat_template_sha256"], self.sha256(template))
            self.assertEqual(
                audit["base_model_sha256_manifest_sha256"], self.sha256(manifest)
            )
            self.assertEqual(audit["base_model_sha256_manifest_entries"], 1)
            self.assertEqual(audit["base_model_sha256_validation"], "PASS")

    def test_rejects_each_frozen_audit_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, *_paths = self.make_audit_fixture(root)
            cases = (
                (("cached_prefix", "input_parquet_sha256"), "input_parquet"),
                (("shared", "chat_template", "sha256"), "chat_template"),
                (
                    ("experiment", "base_model_sha256_file_sha256"),
                    "base_model_sha256_file",
                ),
            )
            for keys, message in cases:
                with self.subTest(field=".".join(keys)):
                    changed = copy.deepcopy(config)
                    target = changed
                    for key in keys[:-1]:
                        target = target[key]
                    target[keys[-1]] = "0" * 64
                    with self.assertRaisesRegex(ValueError, message):
                        collect_audit_hashes(changed, root)

    def test_rejects_tampered_base_model_file_from_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _model, _manifest, _template, _input, weight = (
                self.make_audit_fixture(root)
            )
            weight.write_bytes(b"tampered-model-weights")
            with self.assertRaisesRegex(ValueError, "Base model file SHA256 mismatch"):
                collect_audit_hashes(config, root)


if __name__ == "__main__":
    unittest.main()
