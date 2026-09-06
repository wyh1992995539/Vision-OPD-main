import copy
import json
from pathlib import Path

import pytest
import yaml

from scripts.audit_vopd_memory_ab import stage_summary, audit_memory
from scripts.compare_vopd_memory_ab import compare_reports
from scripts.vopd_memory_experiment import ROOT, AB_POSTFLIGHT

GIB = 1024**3


def traces(directory, variant='deferred', steps=8):
    directory.mkdir(parents=True, exist_ok=True)
    for rank in (0, 1):
        rows, previous = [], None
        for step in range(1, steps + 1):
            names = ['actor_update_entry_after_rollout', 'actor_parameter_load/after']
            if variant == 'baseline':
                names += ['optimizer_load/before', 'optimizer_load/after']
            for phase in ('student_forward', 'teacher_forward', 'backward'):
                names += [phase + '/before', phase + '/after']
            if variant == 'deferred':
                names += ['optimizer_load/before', 'optimizer_load/after']
            names += ['optimizer_step/before', 'optimizer_step/after']
            if variant == 'deferred':
                names += ['optimizer_offload/before', 'optimizer_offload/after']
            names += ['teacher_ema/before', 'teacher_ema/after', 'actor_update_exit']
            for name in names:
                rows.append(dict(event=name, interval_start=previous, pid=100 + rank,
                                 monotonic_seconds=len(rows), time_unix=100 + len(rows), global_step=step,
                                 synchronization_enabled=True, allocated_bytes=10, reserved_bytes=20,
                                 interval_peak_allocated_bytes=15, interval_peak_reserved_bytes=25,
                                 device_free_bytes=100, device_total_bytes=200,
                                 micro_batch_samples=1, sequence_width=50,
                                 max_unpadded_sequence_tokens=40, response_width=10))
                previous = name
        (directory / f'rank{rank}.pid{100+rank}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return directory


@pytest.mark.parametrize('variant', ['baseline', 'deferred'])
def test_complete_stage_traces_summarize_actual_phase_intervals(tmp_path, variant):
    result = stage_summary(traces(tmp_path, variant), 8, variant)
    assert len(result['ranks']) == 2
    assert result['phase_peaks']['0/student_forward'] == dict(allocated_bytes=15, reserved_bytes=25, intervals=8)
    assert len(result['forward_shapes']) == 32


@pytest.mark.parametrize('failure', ['rank', 'truncated', 'chain', 'nan', 'shape', 'step', 'pid', 'timing_order'])
def test_incomplete_or_malformed_stage_evidence_fails(tmp_path, failure):
    traces(tmp_path)
    if failure == 'rank':
        (tmp_path / 'rank1.pid101.jsonl').unlink()
    else:
        path = tmp_path / 'rank0.pid100.jsonl'
        rows = [json.loads(s) for s in path.read_text().splitlines()]
        if failure == 'truncated':
            rows.pop()
        elif failure == 'chain':
            rows[1]['interval_start'] = 'wrong'
        elif failure == 'nan':
            rows[1]['allocated_bytes'] = float('nan')
        elif failure == 'shape':
            rows[2].pop('sequence_width')
        elif failure == 'step':
            rows[1]['global_step'] = 2
        elif failure == 'pid':
            rows[0]['pid'] = 999
        elif failure == 'timing_order':
            rows[1]['monotonic_seconds'] = -1
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    with pytest.raises((ValueError, KeyError)):
        stage_summary(tmp_path, 8, 'deferred')


def pair():
    baseline = dict(status='PASS_MEMORY_AB_RUN', stage_gate_pass=True, variant='baseline',
                    comparison_config={'seed': 42}, source_hashes={'source': 'hash'},
                    train_sha256='data', sample_ids=['one'], hardware=[[0, 'gpu0', 100*GIB], [1, 'gpu1', 100*GIB]],
                    cpu_capacity_bytes=240*GIB, gpu_abort_ratio=.98, cpu_abort_ratio=.95,
                    run_start_unix=1, run_end_unix=2,
                    device_peak_bytes={'0': 95*GIB, '1': 95*GIB},
                    device_peak_ratios={'0': .95, '1': .95}, marker_peak_ratio=.95,
                    memory_stages={'phase_peaks': {'0/student_forward': {'allocated_bytes': 80*GIB, 'reserved_bytes': 90*GIB}}, 'forward_shapes': [[1, 20]]},
                    workload=[[1, 20, 10, 10]], cpu_peak_bytes=180*GIB, cpu_peak_ratio=.75,
                    wall_seconds=100, step_seconds=[10])
    deferred = copy.deepcopy(baseline)
    deferred.update(variant='deferred', run_start_unix=3, run_end_unix=4,
                    device_peak_bytes={'0': 93*GIB, '1': 93*GIB}, device_peak_ratios={'0': .93, '1': .93})
    return baseline, deferred


def test_observed_benefit_never_authorizes_formal_training():
    result = compare_reports(*pair())
    assert result['status'] == 'PASS_OBSERVED_MEMORY_REDUCTION'
    assert result['optimization_validated'] is True
    assert result['formal_training_authorized'] is False


@pytest.mark.parametrize('mutation,expected', [
    ('no_benefit', 'NO_CLEAR_MEMORY_BENEFIT'),
    ('local_only', 'NO_CLEAR_MEMORY_BENEFIT'),
    ('regression', 'FAIL_GPU_PEAK_REGRESSION'),
    ('shorter', 'REVIEW_WORKLOAD_DIFFERENCE'),
    ('gpu_line', 'FAIL_MEMORY_HEADROOM'),
    ('marker_line', 'FAIL_MEMORY_HEADROOM'),
    ('cpu_line', 'FAIL_MEMORY_HEADROOM'),
    ('hardware', 'FAIL_AB_COMPARABILITY'),
    ('overlap', 'FAIL_AB_COMPARABILITY'),
    ('config', 'FAIL_AB_COMPARABILITY'),
    ('source', 'FAIL_AB_COMPARABILITY'),
    ('failed_run', 'FAIL_AB_EVIDENCE'),
    ('not_run', 'WAITING_FOR_RUNS'),
])
def test_comparison_rejects_misleading_success(mutation, expected):
    b, d = pair()
    if mutation in ('no_benefit', 'local_only'):
        d['device_peak_bytes'] = b['device_peak_bytes'].copy()
        d['memory_stages']['phase_peaks']['0/student_forward']['allocated_bytes'] = 40*GIB
    elif mutation == 'regression':
        d['device_peak_bytes']['1'] = 96*GIB
    elif mutation == 'shorter':
        d['workload'] = [[1, 20, 1, 1]]
    elif mutation == 'gpu_line':
        d['device_peak_ratios']['0'] = .98
    elif mutation == 'marker_line':
        d['marker_peak_ratio'] = .98
    elif mutation == 'cpu_line':
        d['cpu_peak_ratio'] = .95
    elif mutation == 'hardware':
        d['hardware'][0][1] = 'different'
    elif mutation == 'overlap':
        d['run_start_unix'] = 1.5
    elif mutation == 'config':
        d['comparison_config']['seed'] = 43
    elif mutation == 'source':
        d['source_hashes']['source'] = 'changed'
    elif mutation == 'failed_run':
        d['status'] = 'FAIL_MEMORY_AB_RUN'
    elif mutation == 'not_run':
        d['status'] = 'NOT_RUN'
    result = compare_reports(b, d)
    assert result['status'] == expected
    assert result['optimization_validated'] is False
    assert result['formal_training_authorized'] is False


def test_failed_evidence_is_not_hidden_by_missing_other_run():
    b, d = pair()
    b['status'], d['status'] = 'FAIL_MEMORY_AB_RUN', 'NOT_RUN'
    assert compare_reports(b, d)['status'] == 'FAIL_AB_EVIDENCE'


@pytest.fixture
def audit_fixture(tmp_path, monkeypatch):
    """Real trace, policy, manifest, invocation, and telemetry; reuse a mocked generic training verdict."""
    import scripts.audit_vopd_memory_ab as module
    from scripts.run_vopd_6241_pilot_guarded import load_pilot_policy, sha256_file
    config = yaml.safe_load((ROOT / 'configs/vopd_6241_pilot_64.yaml').read_text())
    output = tmp_path / 'run'
    config['paths']['output_dir'] = str(output)
    config['experiment']['id'] = 'E-D11-MEM-DEFERRED'
    cfg = tmp_path / 'config.yaml'
    cfg.write_text(yaml.safe_dump(config))
    policy = yaml.safe_load((ROOT / 'configs/vopd_6241_pilot_abort_policy.yaml').read_text())
    policy['pilot']['postflight_script'] = AB_POSTFLIGHT
    policy['pilot']['stage_contracts']['64'].update(config=str(cfg), output_dir=str(output), experiment_id=config['experiment']['id'], require_cold_reload=False)
    manifest_path = tmp_path / 'manifest.json'
    policy['memory_experiment'] = {'variant': 'deferred', 'manifest': str(manifest_path)}
    pp = tmp_path / 'policy.yaml'
    pp.write_text(yaml.safe_dump(policy))
    effective, _ = load_pilot_policy(pp, '64')
    manifest = {'source_hashes': {'test': 'hash'}}
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(module, 'memory_overrides', lambda *args: ['expected'])
    (output / 'preflight').mkdir(parents=True)
    (output / 'logs').mkdir()
    log = output / 'logs/train.log'
    log.write_text('synthetic training passed')
    (output / 'preflight/run_invocation.json').write_text(json.dumps({'hydra_overrides': ['expected'], 'train_file_sha256': 'data', 'sample_ids': ['one']}))
    live = dict(status='PASS', policy_sha256=sha256_file(pp), config_sha256=sha256_file(cfg),
                experiment_id=config['experiment']['id'], effective_policy=effective, memory_experiment_manifest=manifest)
    (output / 'preflight/pilot_live_launch_gate.json').write_text(json.dumps(live))
    traces(output / 'evidence/memory_stages')
    tele = output / 'evidence/telemetry'
    tele.mkdir()
    gpu = [dict(index=i, uuid=f'gpu{i}', memory_used_bytes=90*GIB, memory_total_bytes=100*GIB) for i in (0, 1)]
    (tele / 'gpu.jsonl').write_text('\n'.join(json.dumps(dict(timestamp_utc=t, elapsed_seconds=s, gpus=gpu)) for t, s in [('1970-01-01T00:00:00+00:00', 0), ('1970-01-01T00:08:20+00:00', 500)]))
    cpu = dict(supported=True, memory_current_bytes=180*GIB, memory_max_bytes=240*GIB, memory_events={'oom': 0, 'oom_kill': 0})
    (tele / 'cgroup_memory.jsonl').write_text(json.dumps(cpu) + '\n' + json.dumps(cpu))
    (output / 'evidence/guard_summary.json').write_text(json.dumps({'finished_at_utc': '1970-01-01T00:08:22+00:00'}))
    generic = dict(training_gate_pass=True, stage_gate_pass=True, status='PASS',
                   inputs={'train_log': {'path': str(log)}}, telemetry={'max_observed_elapsed_seconds': 500},
                   steps=[dict(step=i, step_seconds=10, prompt_max_tokens=40, response_mean_tokens=10, response_max_tokens=10) for i in range(1, 9)])
    monkeypatch.setattr(module, 'pilot_audit', lambda *args: copy.deepcopy(generic))
    return pp, output, generic


def test_dedicated_audit_passes_without_reload_and_keeps_formal_blocked(audit_fixture):
    pp, _, _ = audit_fixture
    report = audit_memory(pp)
    assert report['status'] == 'PASS_MEMORY_AB_RUN', report['errors']
    assert report['cold_reload_required'] is False
    assert report['formal_training_authorized'] is False
    assert report['coverage']['post_warmup_steps'] == 0


@pytest.mark.parametrize('failure', ['training', 'override', 'launch_binding', 'empty_telemetry', 'oom'])
def test_dedicated_audit_fails_closed(audit_fixture, failure):
    pp, output, generic = audit_fixture
    if failure == 'training':
        generic['training_gate_pass'] = False
    elif failure == 'override':
        (output / 'preflight/run_invocation.json').write_text('{"hydra_overrides": []}')
    elif failure == 'launch_binding':
        p = output / 'preflight/pilot_live_launch_gate.json'
        r = json.loads(p.read_text())
        r['memory_experiment_manifest'] = {}
        p.write_text(json.dumps(r))
    elif failure == 'empty_telemetry':
        (output / 'evidence/telemetry/gpu.jsonl').write_text('')
    elif failure == 'oom':
        p = output / 'evidence/telemetry/cgroup_memory.jsonl'
        rows = [json.loads(s) for s in p.read_text().splitlines()]
        rows[-1]['memory_events']['oom_kill'] = 1
        p.write_text('\n'.join(json.dumps(r) for r in rows))
    result = audit_memory(pp)
    assert result['status'] == 'FAIL_MEMORY_AB_RUN'
    assert result['stage_gate_pass'] is False
