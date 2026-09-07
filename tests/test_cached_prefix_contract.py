import hashlib
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tensordict import TensorDict
import yaml

from verl.protocol import DataProto
from verl.trainer.ppo.cached_prefix import CachedPrefixStore, TOKEN_IDS_SOURCE
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _object_array(value):
    result = np.empty(1, dtype=object)
    result[0] = value
    return result


def test_cached_store_binds_sample_id_prompt_image_and_response(tmp_path):
    image = tmp_path / "image.png"
    image.write_bytes(b"image")
    cache = tmp_path / "cache.parquet"
    prompt_text = "What is shown?"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "sample_id": "sample-1",
                    "split": "train",
                    "prompt_sha256": hashlib.sha256(prompt_text.encode()).hexdigest(),
                    "image_path": str(image),
                    "response_token_ids": [10, 11, 12],
                    "response_length": 3,
                    "finish_reason": "stop",
                    "inference_error": None,
                    "token_ids_source": TOKEN_IDS_SOURCE,
                    "generation_config_sha256": "generation-hash",
                    "model_path": "/model/base",
                }
            ]
        ),
        cache,
    )
    store = CachedPrefixStore.from_parquet(
        cache,
        expected_sha256=_sha(cache),
        expected_records=1,
        max_response_length=1024,
        expected_generation_config_sha256="generation-hash",
        expected_model_path="/model/base",
    )
    raw_prompt = [
        {
            "role": "user",
            "content": [
                {"type": "image", "path": str(image), "image": str(image)},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    batch = DataProto(
        batch=TensorDict({"dummy": torch.zeros((1, 1))}, batch_size=[1]),
        non_tensor_batch={
            "extra_info": _object_array({"provenance": {"sample_id": "sample-1"}}),
            "raw_prompt": _object_array(raw_prompt),
        },
    )
    metrics = store.bind(batch)
    assert batch.non_tensor_batch["cached_response_ids"][0] == [10, 11, 12]
    assert batch.non_tensor_batch["cached_sample_id"].tolist() == ["sample-1"]
    assert metrics == {
        "cached_prefix/resolved_records": 1.0,
        "cached_prefix/online_generation_calls": 0.0,
    }


class _Manager:
    def __init__(self):
        self.online_calls = 0
        self.cached_calls = 0

    def generate_sequences(self, batch):
        self.online_calls += 1
        return ("online", batch)

    def build_cached_sequences(self, batch):
        self.cached_calls += 1
        return ("cached", batch)


class _Store:
    def __init__(self):
        self.calls = 0

    def bind(self, batch):
        self.calls += 1
        return {"cached_prefix/online_generation_calls": 0.0}


def test_cached_dispatch_never_calls_online_generation():
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.prefix_source = "cached"
    trainer.cached_prefix_store = _Store()
    trainer.async_rollout_manager = _Manager()
    result, metrics = trainer._build_prefix_batch("batch")
    assert result == ("cached", "batch")
    assert trainer.cached_prefix_store.calls == 1
    assert trainer.async_rollout_manager.cached_calls == 1
    assert trainer.async_rollout_manager.online_calls == 0
    assert metrics["cached_prefix/online_generation_calls"] == 0.0


def test_online_dispatch_preserves_existing_generation_path():
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.prefix_source = "online"
    trainer.cached_prefix_store = None
    trainer.async_rollout_mode = True
    trainer.async_rollout_manager = _Manager()
    result, metrics = trainer._build_prefix_batch("batch")
    assert result == ("online", "batch")
    assert trainer.async_rollout_manager.online_calls == 1
    assert trainer.async_rollout_manager.cached_calls == 0
    assert metrics == {}


def test_frozen_6241_cache_and_training_contract_are_bound():
    online = yaml.safe_load((ROOT / "configs/vopd_6241.yaml").read_text())
    cached = yaml.safe_load((ROOT / "configs/cached_prefix_6241.yaml").read_text())
    cache = cached["cached_prefix"]
    assert online["experiment"]["prefix_source"] == "online"
    assert cached["experiment"]["prefix_source"] == "cached"
    assert cached["data"] == online["data"]
    assert cached["rollout"] == online["rollout"]
    assert cached["self_distillation"] == online["self_distillation"]
    assert cached["training"] == online["training"]
    assert {k: v for k, v in cached["actor"].items() if k != "memory_profile_dir"} == {
        k: v for k, v in online["actor"].items() if k != "memory_profile_dir"
    }
    cache_path = Path(cache["output_parquet"])
    assert _sha(cache_path) == cache["output_sha256"]
    store = CachedPrefixStore.from_parquet(
        cache_path,
        expected_sha256=cache["output_sha256"],
        expected_records=6241,
        max_response_length=1024,
        expected_generation_config_sha256=cache["generation_config_sha256"],
        expected_model_path=cached["paths"]["model"],
    )
    assert len(store) == 6241


def test_unified_launcher_forwards_prefix_contract():
    launcher = (ROOT / "scripts/run_vopd_2gpu.sh").read_text()
    for name in (
        "prefix_source", "cached_prefix_path", "cached_prefix_sha256",
        "cached_prefix_expected_records", "cached_prefix_token_ids_source",
        "cached_prefix_generation_config_sha256", "cached_prefix_base_model_path",
    ):
        assert f"actor_rollout_ref.rollout.{name}=" in launcher


def test_cached_pilot_guard_uses_estimate_only_accounting_and_runtime_ablation_gate():
    operations = yaml.safe_load((ROOT / "configs/cached_prefix_pilot_operations.yaml").read_text())
    policy = yaml.safe_load((ROOT / "configs/cached_prefix_6241_pilot_abort_policy.yaml").read_text())
    reload_config = yaml.safe_load((ROOT / "configs/cached_prefix_6241_pilot_64_reload.yaml").read_text())
    stage = policy["pilot"]["stage_contracts"]["64"]
    launcher = (ROOT / "scripts/run_cached_prefix_pilot.py").read_text()
    postflight = (ROOT / "scripts/audit_cached_prefix_pilot.py").read_text()

    assert operations["hourly_dual_gpu_rate_cny"] == 14.0
    assert operations["require_billing_observation"] is False
    assert operations["enforce_cumulative_cost_gate"] is False
    assert policy["budget"]["accounting_mode"] == "estimate_only"
    assert stage["experiment_id"] == "E-D13-6K-CACHED-PILOT-001"
    assert stage["expected_optimizer_steps"] == 8
    assert stage["require_cold_reload"] is True
    assert stage["max_incremental_cost_cny"] == 112.0
    assert reload_config["source_dir"].endswith("checkpoints/global_step_8")
    assert "--current-autodl-cost-cny" not in launcher
    assert "--billing-observed-at-utc" not in launcher
    assert '"online_generation_calls_zero_each_step"' in postflight
    assert '"cached_records_resolved_total_64"' in postflight
    assert 'report["strict_prefix_source_ablation"] = "PASS"' in postflight
