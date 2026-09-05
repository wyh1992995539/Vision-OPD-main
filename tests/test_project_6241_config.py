import hashlib
import unittest
from pathlib import Path

import yaml

from scripts.build_project_parquet import configured_splits as parquet_splits
from scripts.extract_project_images import configured_splits as extraction_splits
from scripts.validate_project_data import configured_splits as validation_splits


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/project_6241.yaml"


class Project6241ConfigTest(unittest.TestCase):
    def test_only_train_is_active(self):
        config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(config["data"]["active_split_count"], 1)
        self.assertEqual(config["data"]["splits"], {"train": config["data"]["splits"]["train"]})
        self.assertEqual(config["data"]["splits"]["train"]["size"], 6241)
        self.assertTrue(all(item["historical_only"] for item in config["data"]["historical_splits"].values()))

    def test_all_data_tools_resolve_the_same_active_manifest(self):
        self.assertEqual(extraction_splits(CONFIG), {"train_6241.jsonl": 6241})
        self.assertEqual(validation_splits(CONFIG), {"train_6241.jsonl": ("train", 6241)})
        self.assertEqual(
            parquet_splits(CONFIG),
            {"train": ("train_6241.jsonl", "train_6241.parquet", 6241)},
        )

    def test_training_arithmetic_is_explicit(self):
        config = yaml.safe_load((ROOT / "configs/vopd_6241.yaml").read_text(encoding="utf-8"))
        policy = yaml.safe_load(
            (ROOT / "configs/vopd_6241_abort_policy.yaml").read_text(encoding="utf-8")
        )
        self.assertEqual(config["training"]["source_samples"], 6241)
        self.assertEqual(config["training"]["expected_samples"], 6240)
        self.assertEqual(config["training"]["total_optimizer_steps"], 780)
        self.assertEqual(config["training"]["padded_samples"], 6240)
        self.assertEqual(config["training"]["padding_rows"], 0)
        self.assertEqual(config["training"]["dropped_rows"], 1)
        self.assertFalse(config["data"].get("full_coverage_padding", {}).get("enabled", False))
        self.assertEqual(config["data"]["tail_policy"], "native_drop_last")
        save_frequency = config["training"]["save_frequency"]
        total_steps = config["training"]["total_optimizer_steps"]
        actual_save_steps = list(range(save_frequency, total_steps, save_frequency))
        actual_save_steps.append(total_steps)
        self.assertEqual(actual_save_steps, [390, 780])
        self.assertEqual(policy["checkpoint"]["allowed_save_steps"], actual_save_steps)
        self.assertEqual(policy["checkpoint"]["expected_final_step"], total_steps)

    def test_paper_core_and_resource_scaled_profile_are_explicit(self):
        config = yaml.safe_load((ROOT / "configs/vopd_6241.yaml").read_text(encoding="utf-8"))
        cached = yaml.safe_load((ROOT / "configs/cached_prefix_6241.yaml").read_text(encoding="utf-8"))
        reference = yaml.safe_load(
            (ROOT / "configs/vopd_6241_algorithm_aligned_2gpu.reference.yaml").read_text(
                encoding="utf-8"
            )
        )
        policy = yaml.safe_load(
            (ROOT / "configs/vopd_6241_abort_policy.yaml").read_text(encoding="utf-8")
        )
        self.assertEqual(config["paper_alignment"]["profile"], "paper_core_resource_scaled_2gpu")
        self.assertEqual(config["data"]["max_response_length"], 1024)
        self.assertEqual(config["actor"]["max_token_length_per_gpu"], 9216)
        self.assertEqual(config["actor"]["lr_warmup_steps"], 10)
        self.assertEqual(config["actor"]["clip_ratio_low"], 0.2)
        self.assertEqual(config["actor"]["clip_ratio_high"], 0.3)
        self.assertEqual(config["self_distillation"]["top_k"], 100)
        self.assertEqual(config["self_distillation"]["alpha"], 0.5)
        self.assertEqual(config["self_distillation"]["teacher_update_rate"], 0.05)
        self.assertEqual(config["resources"]["gpus_per_node"], 2)
        self.assertTrue(config["resources"]["reference_parameter_offload_required"])
        self.assertTrue(config["actor"]["parameter_offload"])
        self.assertTrue(config["actor"]["optimizer_offload"])
        self.assertTrue(config["actor"]["reference_parameter_offload"])
        self.assertEqual(config["data"]["train_batch_size"], 8)
        self.assertEqual(config["rollout"]["n"], 1)
        vllm = config["rollout"]["engine_kwargs"]["vllm"]
        self.assertEqual(config["rollout"]["gpu_memory_utilization"], 0.40)
        self.assertEqual(vllm["compilation_config"]["cudagraph_capture_sizes"], [1, 2, 4, 8])
        self.assertFalse(vllm["compilation_config"]["pass_config"]["fuse_allreduce_rms"])
        self.assertFalse(vllm["kernel_config"]["enable_flashinfer_autotune"])
        self.assertEqual(reference["status"], "reference_only_not_selected")
        self.assertEqual(reference["data"]["train_batch_size"], 96)
        self.assertEqual(reference["algorithm"]["ppo_mini_batch_size"], 96)
        self.assertEqual(reference["algorithm"]["rollout_n"], 8)
        self.assertEqual(reference["data"]["total_optimizer_steps"], 65)
        self.assertEqual(cached["cached_prefix"]["generation"]["max_new_tokens"], 1024)
        self.assertEqual(cached["serving"]["max_model_len"], 9216)
        self.assertTrue(policy["budget"]["require_pilot_reestimate_before_day12"])
        self.assertEqual(
            policy["memory"]["prelaunch_cgroup_minimum_bytes"], 192 * 1024**3
        )
        self.assertEqual(
            policy["budget"]["conservative_reservation_cny"],
            policy["runtime"]["max_wall_time_hours"]
            * policy["budget"]["hourly_dual_gpu_rate_cny"],
        )

    def test_cached_prefix_frozen_audit_hashes_match_inputs(self):
        cached = yaml.safe_load(
            (ROOT / "configs/cached_prefix_6241.yaml").read_text(encoding="utf-8")
        )

        def digest(path):
            return hashlib.sha256(path.read_bytes()).hexdigest()

        experiment = cached["experiment"]
        template = cached["shared"]["chat_template"]
        prefix = cached["cached_prefix"]
        manifest_path = ROOT / experiment["base_model_sha256_file"]
        template_path = ROOT / template["file"]
        input_path = Path(prefix["input_parquet"])
        self.assertEqual(
            experiment["base_model_sha256_file_sha256"], digest(manifest_path)
        )
        self.assertEqual(template["sha256"], digest(template_path))
        self.assertEqual(prefix["input_parquet_sha256"], digest(input_path))

    def test_pilot_configs_preserve_formal_algorithm_contract(self):
        formal = yaml.safe_load((ROOT / "configs/vopd_6241.yaml").read_text(encoding="utf-8"))
        pilot_16 = yaml.safe_load(
            (ROOT / "configs/vopd_6241_pilot_16.yaml").read_text(encoding="utf-8")
        )
        pilot_64 = yaml.safe_load(
            (ROOT / "configs/vopd_6241_pilot_64.yaml").read_text(encoding="utf-8")
        )
        for pilot, rows, steps in ((pilot_16, 16, 2), (pilot_64, 64, 8)):
            self.assertEqual(pilot["data"]["expected_train_rows"], rows)
            self.assertEqual(pilot["training"]["expected_samples"], rows)
            self.assertEqual(pilot["training"]["total_optimizer_steps"], steps)
            self.assertEqual(pilot["data"]["train_batch_size"], 8)
            self.assertEqual(pilot["data"]["max_response_length"], 1024)
            self.assertEqual(pilot["rollout"]["n"], 1)
            self.assertEqual(pilot["self_distillation"], formal["self_distillation"])
            self.assertEqual(pilot["actor"], formal["actor"])
            self.assertEqual(pilot["rollout"], formal["rollout"])
            self.assertTrue(pilot["training"]["require_full_epoch"])
            self.assertEqual(pilot["training"]["dropped_rows"], 0)
        self.assertEqual(pilot_16["pilot"]["stage"], 16)
        self.assertEqual(pilot_64["pilot"]["stage"], 64)
        self.assertFalse(pilot_16["pilot"]["may_start_formal_training"])
        self.assertFalse(pilot_64["pilot"]["may_start_formal_training"])


if __name__ == "__main__":
    unittest.main()
