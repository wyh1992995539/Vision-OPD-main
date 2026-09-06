"""CPU-only checks; no model loading or GPU initialization."""

import ast
import contextlib
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import warnings
import weakref

import pytest

from scripts.monitor_vopd_training import read_memory_stat

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("shard_io", ROOT / "verl/utils/checkpoint/shard_io.py")
io = importlib.util.module_from_spec(spec)
spec.loader.exec_module(io)


def test_round_trip_model_optimizer_rng_and_shared_storage(tmp_path):
    import torch

    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-6)
    model(torch.ones(1, 3)).sum().backward()
    optimizer.step()
    tensor = torch.arange(12)
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "rng": torch.get_rng_state(), "base": tensor, "view": tensor[2:5]}
    path = tmp_path / "shard.pt"
    io.save_shard(state, path, flush_reclaim=True)
    restored = torch.load(path, weights_only=False)
    restored_model = torch.nn.Linear(3, 2)
    restored_model.load_state_dict(restored["model"])
    restored_optimizer = torch.optim.AdamW(restored_model.parameters())
    restored_optimizer.load_state_dict(restored["optimizer"])
    for old, new in zip(model.parameters(), restored_model.parameters(), strict=True):
        assert torch.equal(old, new)
    for key, value in optimizer.state_dict()["state"].items():
        for field, expected in value.items():
            assert torch.equal(expected, restored_optimizer.state_dict()["state"][key][field])
    assert torch.equal(restored["rng"], state["rng"])
    assert restored["base"].untyped_storage().data_ptr() == restored["view"].untyped_storage().data_ptr()
    assert torch.equal(tensor, torch.arange(12))  # no destructive mutation of live state


def test_fsync_precedes_file_scoped_advice(tmp_path):
    import torch

    calls = []
    with patch.object(io.os, "fsync", side_effect=lambda fd: calls.append(("fsync", fd))), patch.object(
        io.os, "posix_fadvise", create=True,
        side_effect=lambda fd, offset, length, advice: calls.append(("advise", fd, offset, length, advice)),
    ), patch.object(io.os, "POSIX_FADV_DONTNEED", 4, create=True):
        io.save_shard({"x": torch.ones(3)}, tmp_path / "model.pt", flush_reclaim=True)
    assert calls[0][0] == "fsync"
    assert calls[1] == ("advise", calls[0][1], 0, 0, 4)


def test_disabled_preserves_original_save_path(tmp_path):
    with patch.object(io.os, "fsync") as sync:
        io.save_shard({"x": 1}, tmp_path / "model.pt")
    sync.assert_not_called()


def test_advice_failure_warns_but_file_remains_loadable(tmp_path):
    import torch

    path = tmp_path / "model.pt"
    with patch.object(io.os, "posix_fadvise", side_effect=OSError("unsupported"), create=True), patch.object(
        io.os, "POSIX_FADV_DONTNEED", 4, create=True
    ), pytest.warns(RuntimeWarning, match="cache advice unavailable"):
        io.save_shard({"x": 7}, path, flush_reclaim=True)
    assert torch.load(path, weights_only=False)["x"] == 7


def test_fsync_failure_propagates_and_never_advises(tmp_path):
    with patch.object(io.os, "fsync", side_effect=OSError("disk failure")), patch.object(
        io.os, "posix_fadvise", create=True
    ) as advice, pytest.raises(OSError, match="disk failure"):
        io.save_shard({"x": 1}, tmp_path / "model.pt", flush_reclaim=True)
    advice.assert_not_called()


def test_serialize_failure_propagates_without_sync(tmp_path):
    import torch

    with patch.object(torch, "save", side_effect=OSError("write failed")), patch.object(
        io.os, "fsync"
    ) as sync, pytest.raises(OSError, match="write failed"):
        io.save_shard({}, tmp_path / "model.pt", flush_reclaim=True)
    sync.assert_not_called()


@pytest.mark.parametrize("enabled", [False, True])
def test_actual_manager_save_block_releases_model_dict_before_optimizer(tmp_path, enabled):
    # Execute the actual save block without importing FSDP/CUDA-dependent engines.
    tree = ast.parse((ROOT / "verl/utils/checkpoint/fsdp_checkpoint_manager.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FSDPCheckpointManager")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "save_checkpoint")
    block = next(n for n in method.body if isinstance(n, ast.With)
                 and ast.unparse(n.items[0].context_expr) == "warnings.catch_warnings()")
    saved = []
    refs = []

    class State(dict):
        pass

    def model_state():
        state = State(model=1)
        refs.append(weakref.ref(state))
        return state

    def optimizer_state():
        assert refs[0]() is None, "Model snapshot still live during optimizer serialization"
        return State(optimizer=2)

    worker = SimpleNamespace(
        rank=0, world_size=2, should_save_model=True, should_save_optimizer=True, should_save_extra=True,
        model=SimpleNamespace(state_dict=model_state), optimizer=SimpleNamespace(state_dict=optimizer_state),
        lr_scheduler=None, get_rng_state=lambda: {"seed": 1},
        checkpoint_config={"fsdp_flush_reclaim": enabled},
    )
    env = {"self": worker, "warnings": warnings, "os": os, "local_path": str(tmp_path),
           "state_dict_cfg": None, "optim_cfg": None, "StateDictType": SimpleNamespace(SHARDED_STATE_DICT=1),
           "get_fsdp_state_ctx": lambda *a: contextlib.nullcontext(), "logger": None,
           "log_with_rank": lambda *a, **k: None,
           "save_shard": lambda state, path, **kw: saved.append((Path(path).name, kw))}
    exec(compile(ast.Module(body=[block], type_ignores=[]), "save_checkpoint", "exec"), env)
    assert [name for name, _ in saved] == ["model_world_size_2_rank_0.pt", "optim_world_size_2_rank_0.pt",
                                         "extra_state_world_size_2_rank_0.pt"]
    assert all(kw == {"flush_reclaim": enabled} for _, kw in saved)


def test_memory_stat_preserves_raw_overlapping_counters(tmp_path):
    (tmp_path / "memory.stat").write_text("anon 100\nfile 200\nshmem 50\nfile_dirty 10\nfile_writeback 5\n")
    result = read_memory_stat(tmp_path)
    assert result["memory_stat"] == {"anon": 100, "file": 200, "shmem": 50, "file_dirty": 10, "file_writeback": 5}
    assert result["memory_stat_error"] is None


@pytest.mark.parametrize("contents", [None, "malformed", "anon invalid"])
def test_memory_stat_missing_or_bad_does_not_disable_total_memory_guard(tmp_path, contents):
    if contents is not None:
        (tmp_path / "memory.stat").write_text(contents)
    result = read_memory_stat(tmp_path)
    assert result["memory_stat"] is None
    assert result["memory_stat_error"]


def test_launcher_enables_io_strategy_without_algorithm_changes():
    text = (ROOT / "scripts/run_vopd_2gpu.sh").read_text()
    assert "++actor_rollout_ref.actor.checkpoint.fsdp_flush_reclaim=true" in text
