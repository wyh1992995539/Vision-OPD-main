import copy
import json
from pathlib import Path

import pytest
import yaml

from scripts import compare_vopd_memory_ab as compare
from scripts import vopd_memory_experiment as experiment
from scripts.run_vopd_6241_pilot_guarded import load_pilot_policy


def versioned_pair(tmp_path):
    current = {p: experiment.sha(experiment.ROOT / p) for p in experiment.SOURCE_PATHS}
    provenance = dict(offline_reaudit=False, source_revisions={}, evaluated_source_hashes=current,
                      training_source_hashes_unchanged=True, original_launch_manifest_preserved=True)
    candidate = dict(source_hashes=copy.deepcopy(current), audit_provenance=provenance)
    baseline = copy.deepcopy(candidate)
    baseline['audit_provenance']['offline_reaudit'] = True
    for i, source in enumerate(sorted(experiment.OFFLINE_AUDIT_SOURCES)):
        archived = tmp_path / str(i)
        archived.write_text('old audit source ' + source)
        old_hash = experiment.sha(archived)
        baseline['source_hashes'][source] = old_hash
        baseline['audit_provenance']['source_revisions'][source] = dict(
            launch_sha256=old_hash, archived_sha256=old_hash, archived_path=str(archived),
            audit_sha256=current[source])
    return baseline, candidate


def test_cross_audit_versions_require_complete_verified_receipts(tmp_path):
    b, d = versioned_pair(tmp_path)
    r = compare.source_compatibility(b, d)
    assert r['compatible'] is True
    assert r['mode'] == 'VERIFIED_AUDIT_ONLY_REVISION'
    assert set(r['differing_sources']) == experiment.OFFLINE_AUDIT_SOURCES


@pytest.mark.parametrize('change,expected', [(None, 'PASS_OBSERVED_MEMORY_REDUCTION'),
    ('gpu', 'FAIL_MEMORY_HEADROOM'), ('marker', 'FAIL_MEMORY_HEADROOM'),
    ('cpu', 'FAIL_MEMORY_HEADROOM'), ('workload', 'REVIEW_WORKLOAD_DIFFERENCE'),
    ('benefit', 'NO_CLEAR_MEMORY_BENEFIT')])
def test_verified_revision_does_not_bypass_benefit_or_safety_checks(tmp_path, change, expected):
    b, d = versioned_pair(tmp_path)
    gib = 1024**3
    common = dict(status='PASS_MEMORY_AB_RUN', stage_gate_pass=True, comparison_config={'seed': 42},
        train_sha256='data', sample_ids=['one'], hardware=[[0, 'gpu0', 100*gib], [1, 'gpu1', 100*gib]],
        cpu_capacity_bytes=240*gib, gpu_abort_ratio=.98, cpu_abort_ratio=.95,
        memory_stages={'phase_peaks': {}, 'forward_shapes': [[1, 20]]}, workload=[[1, 20, 10, 10]],
        cpu_peak_bytes=180*gib, cpu_peak_ratio=.75, wall_seconds=100, step_seconds=[10], marker_peak_ratio=.93)
    b.update(copy.deepcopy(common), variant='baseline', run_start_unix=1, run_end_unix=2,
             device_peak_bytes={'0': 96*gib, '1': 95*gib}, device_peak_ratios={'0': .96, '1': .95})
    d.update(copy.deepcopy(common), variant='deferred', run_start_unix=3, run_end_unix=4,
             device_peak_bytes={'0': 93*gib, '1': 93*gib}, device_peak_ratios={'0': .93, '1': .93})
    if change == 'gpu':
        d['device_peak_ratios']['0'] = .98
    elif change == 'marker':
        d['marker_peak_ratio'] = .98
    elif change == 'cpu':
        d['cpu_peak_ratio'] = .95
    elif change == 'workload':
        d['workload'] = [[1, 20, 1, 1]]
    elif change == 'benefit':
        d['device_peak_bytes'] = b['device_peak_bytes'].copy()
    result = compare.compare_reports(b, d)
    assert result['status'] == expected
    assert result['formal_training_authorized'] is False


@pytest.mark.parametrize('failure', ['runtime', 'collector', 'launcher', 'missing_source',
    'missing_receipt', 'extra_receipt', 'tampered_archive', 'missing_archive', 'old_hash',
    'new_hash', 'stale_evaluator', 'no_offline', 'no_provenance', 'manifest_unverified',
    'candidate_stale'])
def test_cross_version_comparison_fails_closed(tmp_path, failure):
    b, d = versioned_pair(tmp_path)
    p = b['audit_provenance']
    source = next(iter(p['source_revisions']))
    receipt = p['source_revisions'][source]
    if failure in ('runtime', 'collector', 'launcher'):
        key = dict(runtime='verl/workers/actor/dp_actor.py', collector='verl/utils/actor_memory.py',
                   launcher='scripts/run_vopd_6241_pilot_guarded.py')[failure]
        b['source_hashes'][key] = 'different'
    elif failure == 'missing_source':
        b['source_hashes'].pop(experiment.SOURCE_PATHS[0])
    elif failure == 'missing_receipt':
        p['source_revisions'].pop(source)
    elif failure == 'extra_receipt':
        p['source_revisions']['unbound'] = receipt
    elif failure == 'tampered_archive':
        Path(receipt['archived_path']).write_text('tampered')
    elif failure == 'missing_archive':
        Path(receipt['archived_path']).unlink()
    elif failure == 'old_hash':
        receipt['launch_sha256'] = 'wrong'
    elif failure == 'new_hash':
        receipt['audit_sha256'] = 'wrong'
    elif failure == 'stale_evaluator':
        p['evaluated_source_hashes'][source] = 'stale'
    elif failure == 'no_offline':
        p['offline_reaudit'] = False
    elif failure == 'no_provenance':
        b.pop('audit_provenance')
    elif failure == 'manifest_unverified':
        p['original_launch_manifest_preserved'] = False
    elif failure == 'candidate_stale':
        d['source_hashes'][source] = 'stale'
    assert compare.source_compatibility(b, d)['compatible'] is False


@pytest.fixture
def unlaunched(tmp_path, monkeypatch):
    import scripts.run_vopd_6241_pilot_guarded as guard
    monkeypatch.setattr(guard, 'static_preflight', lambda *args: dict(status='PASS', failed_checks=[]))
    old = tmp_path / 'old'
    old.mkdir()
    config = yaml.safe_load((experiment.ROOT / 'configs/vopd_6241_pilot_64.yaml').read_text())
    config['experiment']['id'] = 'E-D11-MEM-DEFERRED'
    config['paths'].update(output_dir=str(old / 'run'), selection_manifest=str(old / 'selection.json'))
    (old / 'selection.json').write_text('{"samples":[{"sample_id":"unchanged"}]}')
    cfg = old / 'config.yaml'
    cfg.write_text(yaml.safe_dump(config))
    policy = yaml.safe_load((experiment.ROOT / 'configs/vopd_6241_pilot_abort_policy.yaml').read_text())
    policy['pilot']['postflight_script'] = experiment.AB_POSTFLIGHT
    policy['pilot']['stage_contracts']['64'].update(
        config=str(cfg), output_dir=str(old / 'run'), experiment_id='E-D11-MEM-DEFERRED', require_cold_reload=False)
    policy['memory_experiment'] = dict(variant='deferred', manifest=str(old / 'manifest.json'))
    pp = old / 'policy.yaml'
    pp.write_text(yaml.safe_dump(policy))
    manifest = dict(variant='deferred', formal_training_authorized=False, config_sha256=experiment.sha(cfg),
        effective_policy=load_pilot_policy(pp, '64')[0],
        source_hashes={p: experiment.sha(experiment.ROOT / p) for p in experiment.SOURCE_PATHS},
        overrides=experiment.expected_overrides('deferred', old / 'run'))
    (old / 'manifest.json').write_text(json.dumps(manifest))
    return old, tmp_path / 'new', tmp_path / 'archive'


def test_preparation_preserves_original_and_changes_only_output_paths(unlaunched):
    old, new, archive = unlaunched
    before = {p: p.read_bytes() for p in old.iterdir()}
    result = experiment.prepare_deferred_revision(old / 'policy.yaml', new, archive)
    assert result['status'] == 'PASS' and result['training_started'] is False
    assert not (old / 'run').exists() and not (new / 'run').exists()
    assert all(p.read_bytes() == data for p, data in before.items())
    old_cfg = yaml.safe_load((old / 'config.yaml').read_text())
    new_cfg = yaml.safe_load((new / 'config.yaml').read_text())
    assert compare.comparable_config(old_cfg) == compare.comparable_config(new_cfg)
    assert (new / 'selection.json').read_bytes() == (old / 'selection.json').read_bytes()
    op, _ = load_pilot_policy(old / 'policy.yaml', '64')
    np, nc = load_pilot_policy(new / 'policy.yaml', '64')
    assert compare.comparable_policy(op) == compare.comparable_policy(np)
    assert experiment.memory_overrides(np, Path(nc['config']), Path(nc['output_dir']))
    with pytest.raises(FileExistsError):
        experiment.prepare_deferred_revision(old / 'policy.yaml', new, archive)


@pytest.mark.parametrize('failure', ['existing_run', 'runtime_change', 'policy_change', 'wrong_variant', 'static_fail'])
def test_preparation_rejects_unsafe_rebinding(unlaunched, monkeypatch, failure):
    old, new, archive = unlaunched
    if failure == 'existing_run':
        (old / 'run').mkdir()
    elif failure == 'runtime_change':
        p = old / 'manifest.json'
        m = json.loads(p.read_text())
        m['source_hashes'][experiment.SOURCE_PATHS[0]] = 'wrong'
        p.write_text(json.dumps(m))
    elif failure in ('policy_change', 'wrong_variant'):
        p = old / 'policy.yaml'
        policy = yaml.safe_load(p.read_text())
        if failure == 'policy_change':
            policy['memory']['gpu_used_ratio_abort'] = .999
        else:
            policy['memory_experiment']['variant'] = 'baseline'
        p.write_text(yaml.safe_dump(policy))
    else:
        import scripts.run_vopd_6241_pilot_guarded as guard
        monkeypatch.setattr(guard, 'static_preflight', lambda *args: dict(status='FAIL', failed_checks=['test']))
    with pytest.raises((ValueError, FileExistsError, RuntimeError)):
        experiment.prepare_deferred_revision(old / 'policy.yaml', new, archive)
    assert not (new / 'run').exists()


@pytest.mark.parametrize('field', ['gpu_used_ratio_abort', 'cgroup_used_ratio_abort', 'prelaunch_cgroup_minimum_bytes'])
def test_policy_comparison_does_not_ignore_resource_thresholds(field):
    original = yaml.safe_load((experiment.ROOT / 'configs/vopd_6241_pilot_abort_policy.yaml').read_text())
    changed = copy.deepcopy(original)
    changed['memory'][field] = -1
    assert compare.comparable_policy(original) != compare.comparable_policy(changed)


def test_effective_policy_identity_is_not_a_guard_difference():
    original = yaml.safe_load((experiment.ROOT / 'configs/vopd_6241_pilot_abort_policy.yaml').read_text())
    original['experiment_id'] = 'E-D11-MEM-BASELINE'
    changed = copy.deepcopy(original)
    changed['experiment_id'] = 'E-D11-MEM-DEFERRED'
    assert compare.comparable_policy(original) == compare.comparable_policy(changed)


@pytest.mark.parametrize('failure', [None, 'baseline', 'config', 'order', 'data', 'started'])
def test_preparation_check_never_confuses_cpu_readiness_with_training(unlaunched, monkeypatch, failure):
    old, _, archive = unlaunched
    policy_path = old / 'policy.yaml'
    config_path = old / 'config.yaml'
    cfg = yaml.safe_load(config_path.read_text())
    data_hash = experiment.sha(Path(cfg['paths']['train_file']))
    selection = dict(output={'sha256': data_hash}, samples=[{'sample_id': 'one'}])
    (old / 'selection.json').write_text(json.dumps(selection))
    baseline = dict(stage_gate_pass=True, comparison_config=compare.comparable_config(cfg),
        train_sha256=data_hash, sample_ids=['one'],
        source_hashes={p: experiment.sha(experiment.ROOT / p) for p in experiment.SOURCE_PATHS})
    monkeypatch.setattr(compare, 'audit_memory', lambda *args, **kwargs: baseline)
    monkeypatch.setattr(compare, 'static_preflight', lambda *args: dict(status='PASS'))
    if failure == 'baseline':
        baseline['stage_gate_pass'] = False
    elif failure == 'config':
        baseline['comparison_config']['actor']['learning_rate'] = .1
    elif failure == 'order':
        baseline['sample_ids'] = ['different']
    elif failure == 'data':
        baseline['train_sha256'] = 'different'
    elif failure == 'started':
        (old / 'run').mkdir()
    r = compare.check_preparation(policy_path, policy_path, archive)
    assert (r['status'] == 'PASS_COMPARISON_PREPARATION') is (failure is None)
    assert r['optimization_validated'] is False
    assert r['formal_training_authorized'] is False
    assert r['training_started'] is False
    assert r['live_gpu_validation_pending'] is True
