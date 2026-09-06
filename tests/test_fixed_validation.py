import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from tensordict import TensorDict
from verl import DataProto
from scripts import fixed_workload_io as io
from scripts.audit_memory_validation import pressure_coverage, compare_fixed
from scripts.memory_validation import exact_replace


def batch(rows=4):
    values = {k: torch.arange(rows*8).reshape(rows, 8) for k in (
        'input_ids', 'responses', 'position_ids', 'teacher_input_ids', 'teacher_position_ids')}
    values.update(response_mask=torch.ones(rows, 8), attention_mask=torch.ones(rows, 8),
                  teacher_attention_mask=torch.ones(rows, 8), teacher_response_start_idx=torch.zeros(rows),
                  old_log_probs=torch.zeros(rows, 8), self_distillation_mask=torch.ones(rows))
    mm = np.empty(rows, dtype=object)
    for i in range(rows):
        mm[i] = {'pixel_values': torch.ones(3, 2, 2)*i, 'image_grid_thw': np.array([1, 2, 2])}
    return DataProto(batch=TensorDict(values, batch_size=[rows]), non_tensor_batch={'multi_modal_inputs': mm},
                     meta_info={'global_steps': 1, 'temperature': 1., 'global_token_num': [8]*rows})


def test_full_actor_input_roundtrip_safe_and_lossless(tmp_path):
    original = batch()
    entry = io.save_batch(original, tmp_path/'batch.pt')
    restored = io.load_batch(entry)
    assert restored.meta_info == original.meta_info
    for k in original.batch.keys():
        assert torch.equal(original.batch[k], restored.batch[k])
    for a, b in zip(original.non_tensor_batch['multi_modal_inputs'], restored.non_tensor_batch['multi_modal_inputs']):
        assert torch.equal(a['pixel_values'], b['pixel_values'])
        assert np.array_equal(a['image_grid_thw'], b['image_grid_thw'])
    assert io.batch_summary(original) == io.batch_summary(restored)
    with pytest.raises(FileExistsError):
        io.save_batch(original, tmp_path/'batch.pt')


def test_tampered_payload_and_missing_fields_fail(tmp_path):
    entry = io.save_batch(batch(), tmp_path/'batch.pt')
    (tmp_path/'batch.pt').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='hash'):
        io.load_batch(entry)
    value = batch()
    del value.batch['teacher_input_ids']
    with pytest.raises(ValueError, match='Missing'):
        io.batch_summary(value)


def test_reject_nonbinary_mask_and_unsupported_python_objects():
    value = batch()
    value.batch['response_mask'][0, 0] = .5
    with pytest.raises(ValueError):
        io.batch_summary(value)
    with pytest.raises(TypeError):
        io.pack(object())


def test_image_roundtrip():
    from PIL import Image
    image = Image.new('RGB', (2, 3), (1, 2, 3))
    restored = io.unpack(io.pack(image))
    assert restored.mode == image.mode and restored.size == image.size and restored.tobytes() == image.tobytes()


def test_replay_mutates_caller_and_records_frozen_payload(tmp_path, monkeypatch):
    from types import SimpleNamespace
    original = batch()
    original.meta_info['global_steps'] = 1
    entry = dict(io.save_batch(original, tmp_path/'batch.pt'), step=1, mode='capture')
    (tmp_path/'bundle.json').write_text(json.dumps({'batches': [entry]}))
    cfg = dict(mode='replay', output_dir=str(tmp_path/'run'), bundle_manifest=str(tmp_path/'bundle.json'))
    (tmp_path/'run/preflight').mkdir(parents=True)
    (tmp_path/'run/preflight/validation_launch.json').write_text(json.dumps({'bundle_sha256': io.sha(tmp_path/'bundle.json')}))
    monkeypatch.setattr(io, 'active', lambda: cfg)
    fresh = batch()
    fresh.batch['responses'] += 100
    io.actor_input(SimpleNamespace(global_steps=1), fresh)
    assert torch.equal(fresh.batch['responses'], original.batch['responses'])
    receipt = json.loads((tmp_path/'run/evidence/fixed_workload/step0001.json').read_text())
    assert receipt['mode'] == 'replay' and receipt['sha256'] == entry['sha256']


@pytest.mark.parametrize('case', ['valid', 'warmup_only', 'short', 'one_rank', 'one_step'])
def test_pressure_coverage_requires_actual_long_postwarmup_on_both_ranks(case):
    steps = [11, 12] if case != 'warmup_only' else [9, 10]
    ranks = [0] if case == 'one_rank' else [0, 1]
    if case == 'one_step':
        steps = [11]
    records = [dict(step=s, rank=r, microbatches=[{'response_lengths':[389 if case=='short' else 1024]}])
               for s in steps for r in ranks]
    assert pressure_coverage(records)['passed'] is (case == 'valid')


def test_equal_actor_inputs_do_not_authorize_whole_run_causal_claim():
    b = dict(status='PASS_FIXED_ACTOR_RUN', stage='fixed_baseline', hardware=['GPU'], cpu_capacity_bytes=240,
             runtime_sha256={'code':'sha'}, microbatch_records=[{'token':'abc'}], actor_inputs=[{'sha256':'bundle'}])
    d = copy.deepcopy(b)
    d['stage'] = 'fixed_deferred'
    assert compare_fixed(b,d)['status'] == 'PASS_MATCHED_ACTOR_INPUTS_PENDING_MEMORY_REVIEW'
    assert compare_fixed(b,d)['optimization_validated'] is False
    assert compare_fixed(b,d)['whole_run_causal_claim_allowed'] is False
    d['microbatch_records'][0]['token'] = 'other'
    assert compare_fixed(b,d)['status'] == 'FAIL_FIXED_COMPARISON'


def test_overlay_anchor_must_match_exactly_once(tmp_path):
    p = tmp_path/'worker.py'
    p.write_text('anchor\nanchor\n')
    with pytest.raises(ValueError):
        exact_replace(p, 'anchor', 'hook')
    assert p.read_text() == 'anchor\nanchor\n'


def test_captured_batch_reproduces_actual_dynamic_partition(tmp_path):
    from verl.utils.seqlen_balancing import prepare_dynamic_batch
    original = batch()
    original.batch['attention_mask'] = original.batch['attention_mask'].to(torch.int64)
    original.batch['attention_mask'][0, 2:] = 0
    original.batch['attention_mask'][1, 5:] = 0
    restored = io.load_batch(io.save_batch(original, tmp_path/'batch.pt'))
    a, ai = prepare_dynamic_batch(original, max_token_len=16, same_micro_num_in_dp=False)
    b, bi = prepare_dynamic_batch(restored, max_token_len=16, same_micro_num_in_dp=False)
    assert ai == bi
    assert [io.batch_summary(x) for x in a] == [io.batch_summary(x) for x in b]


def test_prepared_trainer_hook_captures_real_dataproto(tmp_path, monkeypatch):
    import ast
    import sys
    from types import SimpleNamespace
    runtime = Path(__file__).resolve().parents[1]/'artifacts/runs/E-D11-6K-GATE-001/memory_optimization/fixed_validation_v1/capture/runtime'
    if not runtime.exists():
        pytest.skip('Run memory_validation.py prepare for isolated integration fixture')
    tree = ast.parse((runtime/'verl/trainer/ppo/ray_trainer.py').read_text())
    fn = next(x for x in ast.walk(tree) if isinstance(x, ast.FunctionDef) and x.name == '_update_actor')
    scope = {'DataProto': DataProto}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), 'actual_prepared_trainer_hook', 'exec'), scope)
    monkeypatch.setitem(sys.modules, 'verl.utils.fixed_workload', io)
    monkeypatch.setattr(io, 'active', lambda: dict(mode='capture', output_dir=str(tmp_path)))
    received = []
    trainer = SimpleNamespace(global_steps=1, use_legacy_worker_impl='enable',
        config=SimpleNamespace(actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(
            temperature=1., multi_turn=SimpleNamespace(enable=False)))),
        actor_rollout_wg=SimpleNamespace(update_actor=lambda value: received.append(value) or value))
    data = batch()
    scope['_update_actor'](trainer, data)
    assert len(received) == 1 and received[0] is data
    receipt = json.loads((tmp_path/'evidence/fixed_workload/step0001.json').read_text())
    restored = io.load_batch(receipt)
    assert restored.meta_info['global_steps'] == 1
    assert restored.meta_info['temperature'] == 1.


def test_bundle_mutation_after_launch_is_rejected(tmp_path, monkeypatch):
    from types import SimpleNamespace
    (tmp_path/'preflight').mkdir()
    (tmp_path/'preflight/validation_launch.json').write_text(json.dumps({'bundle_sha256': 'original'}))
    path = tmp_path/'bundle.json'
    path.write_text('{}')
    monkeypatch.setattr(io, 'active', lambda: dict(mode='replay', output_dir=str(tmp_path), bundle_manifest=str(path)))
    with pytest.raises(ValueError, match='changed after launch'):
        io.actor_input(SimpleNamespace(global_steps=1), batch())
