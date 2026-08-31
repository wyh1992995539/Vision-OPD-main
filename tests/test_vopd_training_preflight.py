import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from scripts.prepare_vopd_stability_subset import build_subset
from scripts.vopd_training_preflight import validate_config


class VopdTrainingPreflightTest(unittest.TestCase):
    def make_fixture(self, root: Path) -> Path:
        model = root / "model"
        model.mkdir()
        for name in (
            "config.json",
            "model.safetensors.index.json",
            "tokenizer_config.json",
            "preprocessor_config.json",
            "model-00001-of-00001.safetensors",
        ):
            (model / name).write_text("{}\n", encoding="utf-8")
        chat_template = root / "chat.jinja"
        chat_template.write_text("{{ messages }}\n", encoding="utf-8")

        rows = []
        for index in range(8):
            full = root / f"full-{index}.png"
            crop = root / f"crop-{index}.png"
            full.write_bytes(b"full")
            crop.write_bytes(b"crop")
            rows.append(
                {
                    "prompt": [{"role": "user", "content": f"question {index}"}],
                    "images": [{"path": str(full)}],
                    "bbox_images": [{"path": str(crop)}],
                    "extra_info": {
                        "provenance": {
                            "sample_id": f"sample-{index}",
                            "source_id": f"source-{index}",
                            "source_row": index,
                            "group_id": f"group-{index}",
                        }
                    },
                }
            )
        source = root / "source.parquet"
        pq.write_table(pa.Table.from_pylist(rows), source)
        train_file = root / "train-8.parquet"
        manifest = root / "selection.json"
        build_subset(source, train_file, manifest, count=8, seed=42)

        config = {
            "experiment": {"id": "E-D8-001", "prefix_source": "online", "seed": 42},
            "paths": {
                "model": str(model),
                "train_file": str(train_file),
                "selection_manifest": str(manifest),
                "chat_template": str(chat_template),
                "output_dir": str(root / "output"),
            },
            "data": {
                "expected_train_rows": 8,
                "train_batch_size": 8,
                "image_key": "images",
                "teacher_image_key": "bbox_images",
                "max_prompt_length": 8192,
                "max_response_length": 256,
                "truncation": "error",
                "shuffle": False,
            },
            "actor": {
                "parameter_offload": False,
                "optimizer_offload": False,
                "reference_parameter_offload": False,
            },
            "rollout": {"n": 1, "gpu_memory_utilization": 0.45},
            "resources": {"gpus_per_node": 2},
            "training": {
                "expected_samples": 8,
                "total_optimizer_steps": 1,
                "total_epochs": 1,
                "require_full_epoch": True,
                "save_frequency": -1,
                "test_frequency": -1,
            },
        }
        config_path = root / "config.yaml"
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        return config_path

    def test_accepts_audited_full_epoch_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = validate_config(self.make_fixture(root), root)
            self.assertEqual(summary["status"], "PASS", summary["errors"])
            self.assertEqual(summary["training_contract"]["total_optimizer_steps"], 1)
            self.assertEqual(len(summary["sample_ids"]), 8)

    def test_rejects_manifest_with_wrong_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = self.make_fixture(root)
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            config["experiment"]["seed"] = 7
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            summary = validate_config(config_path, root)
            self.assertEqual(summary["status"], "FAIL")
            self.assertIn(
                "selection manifest seed does not match experiment seed", summary["errors"]
            )

    def test_launcher_has_no_day7_training_constants(self):
        launcher = Path(__file__).resolve().parents[1] / "scripts/run_vopd_2gpu.sh"
        text = launcher.read_text(encoding="utf-8")
        self.assertNotIn("trainer.group_name=E-D7-001", text)
        self.assertNotIn("trainer.save_freq=-1", text)
        self.assertIn('trainer.total_training_steps="$TOTAL_OPTIMIZER_STEPS"', text)


if __name__ == "__main__":
    unittest.main()
