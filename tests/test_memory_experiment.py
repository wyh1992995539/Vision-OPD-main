import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from scripts.vopd_memory_experiment import SOURCE_PATHS, ROOT, expected_overrides, memory_overrides, sha
from verl.utils.actor_memory import StageMemoryRecorder, OptimizerResidency


def test_selection_fork_preserves_rows_order_and_parent_evidence(tmp_path):
    from scripts.vopd_memory_experiment import fork_selection_manifest
    original = {'experiment_id': 'original', 'samples': [{'sample_id': 'b'}, {'sample_id': 'a'}],
                'source': {'sha256': 'source'}, 'output': {'sha256': 'data'}, 'selection': {'seed': 42}}
    path = tmp_path / 'selection.json'
    path.write_text(json.dumps(original))
    fork = fork_selection_manifest(path, 'E-D11-MEM-BASELINE')
    assert fork['experiment_id'] == 'E-D11-MEM-BASELINE'
    assert fork.pop('memory_ab_parent_manifest')['sha256'] == sha(path)
    fork['experiment_id'] = original['experiment_id']
    assert fork == original
    assert json.loads(path.read_text()) == original


def test_no_extension_keeps_historical_guard_commands_unchanged(tmp_path):
    assert memory_overrides({}, tmp_path / 'unused', tmp_path) == []


def test_experiment_checks_source_hash_and_output_binding(tmp_path):
    config = tmp_path / 'config.yaml'
    config.write_text('experiment:\n  id: E-D11-MEM-DEFERRED\n')
    output = tmp_path / 'run'
    manifest = dict(variant='deferred', formal_training_authorized=False,
                    config_sha256=sha(config), source_hashes={p: sha(ROOT/p) for p in SOURCE_PATHS},
                    overrides=expected_overrides('deferred', output))
    path = tmp_path / 'manifest.json'
    policy = {'memory_experiment': {'manifest': str(path), 'variant': 'deferred'},
              'pilot': {'postflight_script': 'scripts/audit_vopd_memory_ab.py',
                        'stage_contracts': {'64': {'require_cold_reload': False}}}}
    manifest['effective_policy'] = policy
    path.write_text(json.dumps(manifest))
    assert memory_overrides(policy, config, output) == expected_overrides('deferred', output)
    changed_policy = json.loads(json.dumps(policy))
    changed_policy['pilot']['stage_contracts']['64']['require_cold_reload'] = True
    with pytest.raises(ValueError, match='dedicated postflight'):
        memory_overrides(changed_policy, config, output)
    changed_policy = json.loads(json.dumps(policy))
    changed_policy['pilot']['postflight_script'] = 'scripts/audit_vopd_6241_pilot.py'
    with pytest.raises(ValueError, match='dedicated postflight'):
        memory_overrides(changed_policy, config, output)
    changed_policy = json.loads(json.dumps(policy))
    changed_policy['memory'] = {'gpu_used_ratio_abort': .999}
    with pytest.raises(ValueError, match='effective policy changed'):
        memory_overrides(changed_policy, config, output)
    with pytest.raises(ValueError, match='output path'):
        memory_overrides(policy, config, tmp_path / 'wrong')
    manifest['source_hashes'][SOURCE_PATHS[0]] = 'wrong'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='source hash changed'):
        memory_overrides(policy, config, output)


def test_deferred_scaler_call_order_uses_real_actor_method():
    from verl.workers.actor.dp_actor import DataParallelPPOActor
    actor = object.__new__(DataParallelPPOActor)
    actor.actor_module = torch.nn.Linear(1, 1)
    actor.actor_optimizer = torch.optim.AdamW(actor.actor_module.parameters())
    actor.config = SimpleNamespace(grad_clip=1.0)
    calls = []
    class Scaler:
        def unscale_(self, optimizer):
            calls.append('unscale')
        def step(self, optimizer):
            calls.append('step')
            optimizer.step()
        def update(self):
            calls.append('scaler_update')
    actor.scaler = Scaler()
    actor.optimizer_residency = OptimizerResidency(
        lambda: calls.append('load'), lambda: calls.append('offload'), StageMemoryRecorder())
    actor.actor_module(torch.ones(1, 1)).sum().backward()
    actor._optimizer_step()
    assert calls == ['load', 'unscale', 'step', 'scaler_update', 'offload']


def test_profiler_failure_before_offload_still_cleans_up():
    recorder = Mock()
    def mark(name):
        if name == 'optimizer_offload/before':
            raise OSError('disk full')
    recorder.mark.side_effect = mark
    offload = Mock()
    manager = OptimizerResidency(Mock(), offload, recorder)
    with pytest.raises(OSError, match='disk full'):
        with manager.step():
            pass
    offload.assert_called_once()
    assert not manager.active


def test_real_backward_helper_scales_loss_without_changing_result():
    from verl.workers.actor.dp_actor import DataParallelPPOActor
    actor = object.__new__(DataParallelPPOActor)
    actor.scaler = None
    x = torch.tensor(3.0, requires_grad=True)
    actor._backward_loss(x.square())
    assert x.grad.item() == 6
    actor.scaler = SimpleNamespace(scale=lambda loss: loss * 2)
    y = torch.tensor(3.0, requires_grad=True)
    actor._backward_loss(y.square())
    assert y.grad.item() == 12
