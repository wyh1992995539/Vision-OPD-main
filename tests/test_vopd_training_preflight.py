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

    def test_enforces_explicit_reference_offload_resource_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = self.make_fixture(root)
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            config["resources"]["reference_parameter_offload_required"] = True
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            summary = validate_config(config_path, root)
            self.assertEqual(summary["status"], "FAIL")
            self.assertFalse(
                summary["checks"]["reference_parameter_offload_matches_resource_contract"]
            )

            config["actor"]["reference_parameter_offload"] = True
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            summary = validate_config(config_path, root)
            self.assertEqual(summary["status"], "PASS", summary["errors"])

    def test_actor_offload_requires_explicit_boolean_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = self.make_fixture(root)
            config = yaml.safe_load(config_path.read_text())
            config["actor"]["parameter_offload"] = True
            config["actor"]["optimizer_offload"] = True
            config_path.write_text(yaml.safe_dump(config))
            summary = validate_config(config_path, root)
            self.assertFalse(summary["checks"]["actor_parameter_offload_matches_resource_contract"])
            self.assertFalse(summary["checks"]["actor_optimizer_offload_matches_resource_contract"])
            config["resources"].update(actor_parameter_offload_required=True, optimizer_offload_required=True)
            config_path.write_text(yaml.safe_dump(config))
            self.assertEqual(validate_config(config_path, root)["status"], "PASS")
            config["resources"]["actor_parameter_offload_required"] = "true"
            config_path.write_text(yaml.safe_dump(config))
            self.assertEqual(validate_config(config_path, root)["status"], "FAIL")

    def test_memory_profile_rejects_capture_memory_or_offload_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = self.make_fixture(root)
            config = yaml.safe_load(config_path.read_text())
            config["actor"].update(parameter_offload=True, optimizer_offload=True, reference_parameter_offload=True)
            config["resources"].update(
                memory_profile="offload_3way_graph4_v1", actor_parameter_offload_required=True,
                optimizer_offload_required=True, reference_parameter_offload_required=True,
                prelaunch_cgroup_minimum_bytes=192 * 1024**3,
            )
            config["rollout"].update(gpu_memory_utilization=0.40, engine_kwargs={
                "vllm": {"compilation_config": {"cudagraph_capture_sizes": [1, 2, 4, 8]}}
            })
            config_path.write_text(yaml.safe_dump(config))
            self.assertEqual(validate_config(config_path, root)["status"], "PASS")
            import copy
            cases = [
                ("actor", "optimizer_offload", False),
                ("rollout", "gpu_memory_utilization", 0.45),
                ("resources", "prelaunch_cgroup_minimum_bytes", 128 * 1024**3),
                ("rollout", "cudagraph_capture_sizes", [1, 2, 4, 8]),
                ("rollout", "engine_kwargs", {"vllm": {"compilation_config": {}}}),
            ]
            for section, key, value in cases:
                with self.subTest(section=section, key=key):
                    modified = copy.deepcopy(config)
                    modified[section][key] = value
                    config_path.write_text(yaml.safe_dump(modified))
                    summary = validate_config(config_path, root)
                    self.assertFalse(summary["checks"]["three_way_offload_memory_profile"])

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
        self.assertIn('actor_rollout_ref.actor.optim.lr_warmup_steps="$LR_WARMUP_STEPS"', text)
        self.assertIn('actor_rollout_ref.actor.clip_ratio_high="$CLIP_RATIO_HIGH"', text)
        self.assertIn('actor_rollout_ref.rollout.temperature="$ROLLOUT_TEMPERATURE"', text)
        self.assertIn('actor_rollout_ref.rollout.top_p="$ROLLOUT_TOP_P"', text)
        self.assertIn('actor_rollout_ref.rollout.top_k="$ROLLOUT_TOP_K"', text)
        self.assertIn('actor_rollout_ref.rollout.ignore_eos="$ROLLOUT_IGNORE_EOS"', text)
        self.assertIn(
            '+actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.pass_config.fuse_allreduce_rms="$VLLM_FUSE_ALLREDUCE_RMS"',
            text,
        )
        self.assertIn(
            '+actor_rollout_ref.rollout.engine_kwargs.vllm.kernel_config.enable_flashinfer_autotune="$VLLM_ENABLE_FLASHINFER_AUTOTUNE"',
            text,
        )
        self.assertIn('actor_rollout_ref.actor.self_distillation.max_reprompt_len="$MAX_REPROMPT_LENGTH"', text)


if __name__ == "__main__":
    unittest.main()
