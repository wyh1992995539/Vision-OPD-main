import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from verl.utils.actor_memory import OptimizerResidency, StageMemoryRecorder, with_optimizer_residency


def test_disabled_recorder_never_calls_cuda():
    backend = Mock()
    recorder = StageMemoryRecorder(backend=backend)
    recorder.mark('disabled')
    assert not backend.mock_calls


def test_intervals_preserve_cumulative_peaks_and_context(tmp_path):
    backend = Mock()
    backend.is_available.return_value = True
    backend.memory_allocated.return_value = 10
    backend.memory_reserved.return_value = 20
    backend.max_memory_allocated.side_effect = [100, 50]
    backend.max_memory_reserved.side_effect = [120, 90]
    backend.mem_get_info.return_value = (1000, 2000)
    recorder = StageMemoryRecorder(tmp_path, backend=backend, rank=1)
    recorder.context = {'global_step': 2}
    recorder.mark('forward/before')
    recorder.mark('forward/after')
    rows = [json.loads(s) for s in recorder.path.read_text().splitlines()]
    assert rows[1]['interval_start'] == 'forward/before'
    assert rows[1]['global_step'] == 2
    assert (recorder.peak_allocated, recorder.peak_reserved) == (100, 120)
    assert backend.reset_peak_memory_stats.call_count == 2
    assert backend.synchronize.call_count == 2


@pytest.mark.parametrize('failure', ['load', 'step', 'offload', None])
def test_residency_cleans_partial_load_and_step_failure(failure):
    calls = []
    def load():
        calls.append('load')
        if failure == 'load':
            raise RuntimeError('load')
    def offload():
        calls.append('offload')
        if failure == 'offload':
            raise RuntimeError('offload')
    manager = OptimizerResidency(load, offload, StageMemoryRecorder())
    def run():
        with manager.step():
            calls.append('step')
            if failure == 'step':
                raise RuntimeError('step')
    if failure:
        with pytest.raises(RuntimeError, match=failure):
            run()
    else:
        run()
        run()
    assert calls[-1] == 'offload'
    assert not manager.active
    assert calls.count('load') == calls.count('offload')


def test_original_failure_not_hidden_by_cleanup_failure():
    def fail_load():
        raise ValueError('original')
    def fail_cleanup():
        raise RuntimeError('cleanup')
    with pytest.raises(ValueError, match='original') as exc:
        with OptimizerResidency(fail_load, fail_cleanup, StageMemoryRecorder()).step():
            pass
    assert 'cleanup' in exc.value.__notes__[0]


def test_real_actor_optimizer_step_matches_with_accumulation_warmup_and_ema():
    # Exercise the actual decorated actor method, not a duplicate optimizer implementation.
    from verl.workers.actor.dp_actor import DataParallelPPOActor
    from verl.utils.fsdp_utils import load_fsdp_optimizer, offload_fsdp_optimizer
    from verl.utils.runtime_evidence import update_ema_parameters
    torch.manual_seed(42)
    initial = torch.nn.Linear(3, 2).state_dict()
    results = []
    for deferred in (False, True):
        model, teacher = torch.nn.Linear(3, 2), torch.nn.Linear(3, 2)
        model.load_state_dict(initial)
        teacher.load_state_dict(initial)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
        actor = object.__new__(DataParallelPPOActor)
        actor.actor_module, actor.actor_optimizer = model, optimizer
        actor.config, actor.scaler = SimpleNamespace(grad_clip=1.0), None
        actor.memory_recorder = StageMemoryRecorder()
        transfers = []
        def load():
            transfers.append('load')
            load_fsdp_optimizer(optimizer, 'cpu')
        def offload():
            transfers.append('offload')
            offload_fsdp_optimizer(optimizer)
        actor.optimizer_residency = OptimizerResidency(load, offload, actor.memory_recorder) if deferred else None
        for step in range(4):
            optimizer.param_groups[0]['lr'] = 0.0 if step == 0 else 2e-6
            optimizer.zero_grad()
            for micro in range(2):
                x = torch.arange(6, dtype=torch.float32).reshape(2, 3) / (micro + 2)
                (model(x).square().mean() / 2).backward()
                assert len(transfers) == step * 2 if deferred else not transfers
            norm = actor._optimizer_step()
            assert torch.isfinite(norm)
            scheduler.step()
            update_ema_parameters(teacher, model, 0.05)
            assert all(p.grad is None for p in teacher.parameters())
        results.append((model.state_dict(), teacher.state_dict(), optimizer.state_dict(), scheduler.state_dict()))
        if deferred:
            assert transfers == ['load', 'offload'] * 4
    def equal(a, b):
        if isinstance(a, torch.Tensor):
            assert torch.equal(a, b)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for k in a:
                equal(a[k], b[k])
        elif isinstance(a, (list, tuple)):
            for x, y in zip(a, b, strict=True):
                equal(x, y)
        else:
            assert a == b
    equal(*results)


def test_nonfinite_grad_skips_real_optimizer_step_and_unloads(monkeypatch):
    from verl.workers.actor.dp_actor import DataParallelPPOActor
    monkeypatch.setattr(torch.distributed, 'get_rank', lambda: 0)
    model = torch.nn.Linear(1, 1)
    actor = object.__new__(DataParallelPPOActor)
    actor.actor_module = model
    actor.actor_optimizer = torch.optim.AdamW(model.parameters())
    actor.config, actor.scaler = SimpleNamespace(grad_clip=1.0), None
    load, unload = Mock(), Mock()
    actor.optimizer_residency = OptimizerResidency(load, unload, StageMemoryRecorder())
    saved = [p.detach().clone() for p in model.parameters()]
    for p in model.parameters():
        p.grad = torch.full_like(p, float('inf'))
    assert not torch.isfinite(actor._optimizer_step())
    assert not actor.actor_optimizer.state
    assert all(torch.equal(a, b) for a, b in zip(saved, model.parameters()))
    load.assert_called_once()
    unload.assert_called_once()
