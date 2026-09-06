"""CPU-only regression and negative tests for diagnostic-to-formal evidence boundaries."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from scripts import day11_validation_evidence as evidence
from scripts import finalize_day11_candidate_gate as candidate_gate
from scripts import finalize_day11_preflight as gate


@pytest.fixture(scope='module')
def pressure():
    path = gate.PATHS['pressure']
    directory = path.parents[2]
    return (json.loads(path.read_text()), json.loads((directory/'manifest.json').read_text()),
            yaml.safe_load((directory/'config.yaml').read_text()))


def test_real_pressure_recomputes_both_rank_lengths_and_physical_peaks(pressure):
    result = evidence.pressure_metrics(evidence.Reader(), *pressure)
    assert result['response_count'] == 128
    assert result['minimum_response_tokens'] == result['maximum_response_tokens'] == 1024
    assert result['coverage']['per_rank_post_warmup_long_steps'] == {
        '0': [11, 12, 13, 14, 15, 16], '1': [11, 12, 13, 14, 15, 16]}
    assert result['marker_peak_ratio'] == pytest.approx(.9059110761503006)
    assert result['cpu_peak_bytes'] / gate.GIB == pytest.approx(183.20665740966797)


@pytest.mark.parametrize('mutation', [
    'missing_rank', 'duplicate_step', 'short_rank', 'bad_rows', 'nan_length',
    'global_disagreement', 'missing_first_gate', 'short_first_gate',
    'coverage_claim', 'peak_claim', 'nan_peak', 'wrong_eos', 'wrong_warmup',
])
def test_pressure_rejects_incomplete_or_inconsistent_claims(pressure, mutation):
    report, manifest, config = copy.deepcopy(pressure)
    if mutation == 'missing_rank':
        report['microbatch_records'].pop()
    elif mutation == 'duplicate_step':
        report['microbatch_records'][-1] = report['microbatch_records'][0]
    elif mutation == 'short_rank':
        report['microbatch_records'][-1]['microbatches'][0]['response_lengths'][0] = 10
    elif mutation == 'bad_rows':
        report['microbatch_records'][-1]['microbatches'][0]['rows'] += 1
    elif mutation == 'nan_length':
        report['microbatch_records'][0]['microbatches'][0]['response_lengths'][0] = float('nan')
    elif mutation == 'global_disagreement':
        report['actor_inputs'][-1]['response_lengths'][0] = 1
    elif mutation == 'missing_first_gate':
        report['first_batch_length_gate'] = {}
    elif mutation == 'short_first_gate':
        report['first_batch_length_gate']['response_lengths'][0] = 1
    elif mutation == 'coverage_claim':
        report['coverage']['per_rank_post_warmup_long_steps']['1'] = []
    elif mutation == 'peak_claim':
        report['gpu_peak_ratio'] = .1
    elif mutation == 'nan_peak':
        report['marker_peak_ratio'] = float('nan')
    elif mutation == 'wrong_eos':
        config['rollout']['ignore_eos'] = False
    elif mutation == 'wrong_warmup':
        config['actor']['lr_warmup_steps'] = 0
    with pytest.raises((ValueError, KeyError)):
        evidence.pressure_metrics(evidence.Reader(), report, manifest, config)


@pytest.mark.parametrize('kind', ['hardware', 'cpu_capacity', 'oom', 'gpu_trace', 'first_lengths',
                                 'negative_gpu', 'negative_cpu', 'unsynchronized', 'missing_trace_step'])
def test_pressure_checks_raw_telemetry_and_first_batch(pressure, kind):
    class ChangedReader(evidence.Reader):
        def jsonl(self, path):
            rows = super().jsonl(path)
            if Path(path).name == 'gpu.jsonl':
                if kind == 'hardware':
                    rows[-1]['gpus'][1]['uuid'] = 'other-gpu'
                if kind == 'gpu_trace':
                    return []
                if kind == 'negative_gpu':
                    rows[0]['gpus'][0]['memory_used_bytes'] = -1
            if Path(path).name == 'cgroup_memory.jsonl':
                if kind == 'cpu_capacity':
                    rows[-1]['memory_max_bytes'] //= 2
                if kind == 'oom':
                    rows[-1]['memory_events']['oom_kill'] += 1
                if kind == 'negative_cpu':
                    rows[0]['memory_current_bytes'] = -1
            if Path(path).parent.name == 'memory_stages':
                if kind == 'unsynchronized':
                    rows[0]['synchronization_enabled'] = False
                if kind == 'missing_trace_step':
                    rows = [r for r in rows if r['global_step'] != 16]
            return rows

        def json(self, path):
            result = super().json(path)
            if kind == 'first_lengths' and Path(path).name == 'first_batch_length_gate.json':
                result['response_lengths'] = [1] * 8
            return result

    with pytest.raises(ValueError):
        evidence.pressure_metrics(ChangedReader(), *pressure)


def test_hash_verifier_rejects_changed_or_missing_source(tmp_path):
    path = tmp_path/'source.py'
    path.write_text('original')
    expected = evidence.Reader().digest(path)
    path.write_text('changed')
    with pytest.raises(ValueError, match='Hash mismatch'):
        evidence.Reader().bound(path, expected)
    with pytest.raises(OSError):
        evidence.Reader().bound(tmp_path/'absent', expected)


@pytest.mark.parametrize('failure', ['manifest', 'stage', 'bare_pass', 'runtime_inputs'])
def test_run_integrity_rejects_partial_or_wrong_evidence(monkeypatch, failure):
    original = evidence.Reader.json

    def altered(reader, path):
        data = original(reader, path)
        if Path(path) == gate.PATHS['fixed_baseline']:
            if failure == 'manifest':
                data['manifest_sha256'] = '0'*64
            elif failure == 'stage':
                data['stage'] = 'pressure'
            elif failure == 'bare_pass':
                data['stage_gate_pass'] = False
        if failure == 'runtime_inputs' and Path(path).name == 'manifest.json':
            data['inputs'] = {}
        return data

    monkeypatch.setattr(evidence.Reader, 'json', altered)
    result = evidence.collect_validation_evidence(gate.PATHS)
    assert result['status'] == 'FAIL_VALIDATION_EVIDENCE'
    assert result['errors']
    assert result['formal_training_authorized'] is False


@pytest.fixture(scope='module')
def diagnostics():
    value = evidence.collect_validation_evidence(gate.PATHS)
    assert value['status'] == 'PASS_DIAGNOSTIC_EVIDENCE', value['errors']
    return value


def test_real_fixed_comparison_preserves_causal_boundary(diagnostics):
    assert diagnostics['fixed_comparison_status'] == 'PASS_MATCHED_ACTOR_INPUTS_PENDING_MEMORY_REVIEW'
    assert diagnostics['optimization_validated'] is False
    assert diagnostics['whole_run_causal_claim_allowed'] is False
    assert len(diagnostics['sources']) >= 1700
    assert 'verl/experimental/agent_loop/agent_loop.py' in diagnostics['fixed_deferred_to_pressure_runtime_differences']


@pytest.mark.parametrize('mutation', ['embedded_run', 'causal_claim', 'missing_checks', 'wrong_status',
                                     'payload', 'microbatch', 'sources', 'hardware'])
def test_fixed_comparison_does_not_trust_pass_flag(monkeypatch, pressure, mutation):
    comparison = json.loads(gate.PATHS['fixed_comparison'].read_text())
    runs = copy.deepcopy(comparison['runs'])
    if mutation == 'embedded_run':
        comparison['runs']['baseline']['cpu_capacity_bytes'] += 1
    elif mutation == 'causal_claim':
        comparison['whole_run_causal_claim_allowed'] = True
    elif mutation == 'missing_checks':
        comparison['checks'] = {}
    elif mutation == 'wrong_status':
        comparison['status'] = 'PASS'
    else:
        deferred = runs['deferred']
        if mutation == 'payload':
            deferred['actor_inputs'][0]['sha256'] = '0'*64
        elif mutation == 'microbatch':
            deferred['microbatch_records'][0]['step'] = 999
        elif mutation == 'sources':
            deferred['runtime_sha256']['fake.py'] = '0'*64
        elif mutation == 'hardware':
            deferred['hardware'][0][1] = 'other-gpu'
        comparison['runs'] = copy.deepcopy(runs)
    def fake_verify(reader, path, stage):
        if stage == 'pressure':
            return pressure
        return runs['baseline' if stage == 'fixed_baseline' else 'deferred'], {}, {}
    monkeypatch.setattr(evidence, 'verify_run', fake_verify)
    original = evidence.Reader.json
    monkeypatch.setattr(evidence.Reader, 'json', lambda reader, path:
                        comparison if Path(path) == gate.PATHS['fixed_comparison'] else original(reader, path))
    result = evidence.collect_validation_evidence(gate.PATHS)
    assert result['status'] == 'FAIL_VALIDATION_EVIDENCE'
    assert result['errors']


def test_gate_uses_new_coverage_but_never_promotes_diagnostics(monkeypatch, diagnostics, tmp_path):
    monkeypatch.setattr(candidate_gate, 'collect_validation_evidence', lambda _: diagnostics)
    paths = dict(gate.PATHS, promotion_receipt=tmp_path/'missing.json')
    value = gate.build_preflight(paths, disk_free_bytes=422*gate.GIB, cpu_capacity_bytes=2*gate.GIB)
    assert value['pilot_coverage']['post_warmup_steps_observed'] == 0
    assert value['checks']['at_least_two_post_warmup_steps_observed']
    assert value['checks']['long_response_training_pressure_observed']
    assert value['checks']['diagnostic_gpu_peaks_below_formal_abort_line']
    assert value['checks']['diagnostic_cpu_peak_below_formal_abort_line']
    assert value['checks']['formal_candidate_validation_bound'] is True
    assert value['evidence_checks']['candidate_validation_pass_and_bound'] is True
    assert value['formal_training_authorized'] is False
    assert value['ready_to_unblock_formal_config'] is False
    assert value['status'] == 'BLOCKED_ADDITIONAL_RESOURCE_VALIDATION'
    assert value['runtime_snapshot']['builder_process_capacity_is_launch_evidence'] is False


def test_changed_status_or_large_resources_cannot_bypass_candidate_gate(monkeypatch, diagnostics, tmp_path):
    monkeypatch.setattr(candidate_gate, 'collect_validation_evidence', lambda _: diagnostics)
    paths = dict(gate.PATHS)
    config = yaml.safe_load(paths['formal_config'].read_text())
    config['status'] = 'ready_for_formal_training'
    paths['promotion_receipt'] = tmp_path/'missing-promotion.json'
    config['resources']['prelaunch_cgroup_minimum_bytes'] = 240*gate.GIB
    paths['formal_config'] = tmp_path/'formal.yaml'
    paths['formal_config'].write_text(yaml.safe_dump(config))
    value = gate.build_preflight(paths, disk_free_bytes=600*gate.GIB, cpu_capacity_bytes=240*gate.GIB)
    assert value['checks']['formal_config_released'] is False
    assert value['checks']['formal_candidate_validation_bound'] is True
    assert value['formal_training_authorized'] is False
    assert value['status'] == 'BLOCKED_ADDITIONAL_RESOURCE_VALIDATION'


def test_bad_diagnostics_do_not_fall_back_to_old_coverage(monkeypatch):
    monkeypatch.setattr(candidate_gate, 'collect_validation_evidence', lambda _: {
        'status': 'FAIL_VALIDATION_EVIDENCE', 'errors': ['changed source']})
    value = gate.build_preflight(gate.PATHS, disk_free_bytes=600*gate.GIB, cpu_capacity_bytes=240*gate.GIB)
    assert value['status'] == 'FAIL_EVIDENCE_INTEGRITY'
    assert not value['checks']['long_response_training_pressure_observed']
    assert not value['checks']['at_least_two_post_warmup_steps_observed']


def test_missing_diagnostic_report_fails_closed(tmp_path):
    paths = dict(gate.PATHS, pressure=tmp_path/'missing.json')
    value = gate.build_preflight(paths, disk_free_bytes=600*gate.GIB, cpu_capacity_bytes=240*gate.GIB)
    assert value['status'] == 'FAIL_MISSING_INPUTS'
    assert value['formal_training_authorized'] is False


def test_refresh_archives_old_report_without_touching_other_evidence(monkeypatch, tmp_path):
    output = tmp_path/'preflight.json'
    old = {'.json': '{"old": true}\n', '.md': 'old markdown\n', '.sha256': 'old digest\n'}
    for suffix, content in old.items():
        output.with_suffix(suffix).write_text(content)
    sentinel = tmp_path/'postflight.json'
    sentinel.write_text('preserve frozen evidence')
    monkeypatch.setattr(sys, 'argv', ['gate', '--output', str(output), '--disk-path', str(tmp_path)])
    monkeypatch.setattr(gate, 'build_preflight', lambda *a, **kw: {
        'status': 'BLOCKED_ADDITIONAL_RESOURCE_VALIDATION', 'artifact_status': 'COMPLETE',
        'formal_training_authorized': False})
    assert gate.main() == 0
    report = json.loads(output.read_text())
    archive = Path(report['previous_report_archive'])
    for suffix, content in old.items():
        assert (archive/output.with_suffix(suffix).name).read_text() == content
    assert sentinel.read_text() == 'preserve frozen evidence'
    assert report['formal_training_authorized'] is False


def test_import_has_no_gpu_or_tensor_runtime_dependencies():
    code = '''
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'vllm', 'ray', 'numpy'}:
        raise RuntimeError('Unexpected heavy import: '+name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from scripts import finalize_day11_preflight
'''
    result = subprocess.run([sys.executable, '-c', code], cwd=gate.ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
