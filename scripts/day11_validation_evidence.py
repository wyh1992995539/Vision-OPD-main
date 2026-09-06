"""CPU-only verification of historical diagnostics; never promotes a formal candidate.

Only JSON/YAML, source bytes and streamed hashes are read. No torch, vLLM,
checkpoint deserialization or GPU initialization is needed.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import yaml


def require(condition, message):
    if not condition:
        raise ValueError(message)


class Reader:
    def __init__(self):
        self.sources = {}

    def digest(self, path):
        path = Path(path).resolve()
        key = str(path)
        if key not in self.sources:
            h = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    h.update(block)
            self.sources[key] = h.hexdigest()
        return self.sources[key]

    def bound(self, path, expected):
        require(bool(expected) and self.digest(path) == expected, f'Hash mismatch: {path}')

    def json(self, path):
        self.digest(path)
        return json.loads(Path(path).read_text())

    def jsonl(self, path):
        self.digest(path)
        return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def verify_run(reader, report_path, stage):
    report_path = Path(report_path)
    directory = report_path.parents[2]
    manifest_path = directory / 'manifest.json'
    manifest = reader.json(manifest_path)
    report = reader.json(report_path)
    checksum_path = report_path.with_suffix('.sha256')
    reader.digest(checksum_path)
    checksum = checksum_path.read_text().strip().split(maxsplit=1)
    require(len(checksum) == 2 and Path(checksum[1]).resolve() == report_path.resolve(),
            'Postflight checksum target mismatch')
    reader.bound(report_path, checksum[0])
    output = directory / 'run'
    require(Path(manifest['output_dir']).resolve() == output.resolve(), 'Run directory mismatch')
    expected_status = 'PASS_PRESSURE_DIAGNOSTIC' if stage == 'pressure' else 'PASS_FIXED_ACTOR_RUN'
    require(manifest['stage'] == stage and report['stage'] == stage, 'Wrong diagnostic stage')
    require(report['status'] == expected_status and report['stage_gate_pass'] is True
            and report['formal_training_authorized'] is False and not report['errors'], 'Diagnostic did not pass')
    reader.bound(manifest_path, report['manifest_sha256'])
    require(manifest['inputs'] and manifest['formal_training_authorized'] is False, 'Missing frozen inputs')
    for path, expected in manifest['inputs'].items():
        reader.bound(path, expected)
    runtime = Path(manifest['runtime'])
    require(runtime.resolve() == (directory / 'runtime').resolve(), 'Wrong isolated runtime')
    runtime_hashes = {str(Path(p).relative_to(runtime)): h for p, h in manifest['inputs'].items()
                      if Path(p).is_relative_to(runtime) and not p.endswith('validation_active.json')}
    require(runtime_hashes and runtime_hashes == report['runtime_sha256'], 'Runtime binding mismatch')
    for name in ('config.yaml', 'policy.yaml', 'selection.json'):
        require(str(directory / name) in manifest['inputs'], f'Unbound {name}')
    config = yaml.safe_load((directory / 'config.yaml').read_text())
    policy = yaml.safe_load((directory / 'policy.yaml').read_text())
    launch = reader.json(output / 'preflight/validation_launch.json')
    require(launch['manifest_sha256'] == reader.digest(manifest_path), 'Launch manifest mismatch')
    invocation = reader.json(output / 'preflight/run_invocation.json')
    require(Path(invocation['config']).resolve() == (directory / 'config.yaml').resolve()
            and Path(invocation['train_file']).resolve() == Path(config['paths']['train_file']).resolve(),
            'Executed config/data path mismatch')
    reader.bound(directory / 'config.yaml', invocation['config_sha256'])
    reader.bound(invocation['train_file'], invocation['train_file_sha256'])
    require(invocation['hydra_overrides'] == policy['validation_overrides'], 'Executed overrides changed')
    deferred = stage != 'fixed_baseline'
    require('++actor_rollout_ref.actor.defer_optimizer_state_load=' + str(deferred).lower()
            in invocation['hydra_overrides'], 'Wrong optimizer loading variant')
    live = reader.json(output / 'preflight/pilot_live_launch_gate.json')
    require(live['status'] == 'PASS', 'Live launch did not pass')
    reader.bound(directory / 'policy.yaml', live['policy_sha256'])
    reader.bound(directory / 'config.yaml', live['config_sha256'])
    audit = report['training_audit']
    require(audit['training_gate_pass'] is True and audit['checks']
            and all(v is True for v in audit['checks'].values()) and not audit['failed_checks'],
            'Mechanism/checkpoint audit failed')
    required_inputs = {'train_log', 'guard_summary', 'run_invocation', 'preflight', 'selection_manifest'}
    require(required_inputs <= audit['inputs'].keys(), 'Missing training audit bindings')
    for entry in audit['inputs'].values():
        reader.bound(entry['path'], entry['sha256'])
    guard = reader.json(output / 'evidence/guard_summary.json')
    require(guard['status'] == 'PASS' and guard['return_code'] == 0 and guard['trigger'] is None,
            'Guard failure')
    steps = manifest['expected_steps']
    require(steps == (16 if stage == 'pressure' else 8), 'Unexpected diagnostic step count')
    require(audit['observed_steps'] == steps and audit['expected_steps'] == steps
            and [s['step'] for s in audit['steps']] == list(range(1, steps + 1)), 'Incomplete optimizer steps')
    require(guard['checkpoint']['status'] == 'PASS'
            and int(guard['checkpoint']['marker_value']) == steps, 'Final checkpoint marker mismatch')
    require(all(v == 0 for v in audit['signals'].values()), 'Training error signals')
    # Historical model files need not be loaded. Their saved audit is not a resume guarantee.
    expected_receipts = {str(output / f'evidence/fixed_workload/step{s:04d}{suffix}.json')
                         for s in range(1, steps + 1) for suffix in ('', '.rank0', '.rank1')}
    require(len(report['inputs']) == len(expected_receipts)
            and {e['path'] for e in report['inputs']} == expected_receipts, 'Incomplete input receipt bindings')
    for entry in report['inputs']:
        reader.bound(entry['path'], entry['sha256'])
    actor, records = [], []
    for step in range(1, steps + 1):
        actor.append(reader.json(output / f'evidence/fixed_workload/step{step:04d}.json'))
        for rank in (0, 1):
            records.append(reader.json(output / f'evidence/fixed_workload/step{step:04d}.rank{rank}.json'))
    require(actor == report['actor_inputs'] and records == report['microbatch_records'],
            'Embedded receipts disagree with originals')
    if stage != 'pressure':
        require(manifest['mode'] == 'replay', 'Fixed run is not replay')
        bundle = reader.json(manifest['bundle_manifest'])
        reader.bound(manifest['bundle_manifest'], launch['bundle_sha256'])
        reader.bound(bundle['capture_manifest_path'], bundle['capture_manifest_sha256'])
        require(bundle['status'] == 'SEALED_CAPTURE' and len(bundle['batches']) == steps, 'Unsealed bundle')
        for entry in bundle['batches']:
            reader.bound(entry['path'], entry['sha256'])
        require([r['sha256'] for r in actor] == [e['sha256'] for e in bundle['batches']], 'Replay payload mismatch')
    return report, manifest, config


def pressure_metrics(reader, report, manifest, config):
    require(manifest['mode'] == 'observe' and config['rollout']['ignore_eos'] is True,
            'Not a forced-length diagnostic')
    warmup = config['actor']['lr_warmup_steps']
    limit = config['data']['max_response_length']
    require(warmup == 10 and limit == 1024, 'Unexpected pressure configuration')
    records = report['microbatch_records']
    expected_pairs = {(s, r) for s in range(1, 17) for r in (0, 1)}
    require(len(records) == 32 and {(r['step'], r['rank']) for r in records} == expected_pairs,
            'Duplicate or missing rank/step')
    eligible = {0: [], 1: []}
    lengths = []
    for r in records:
        local = [n for b in r['microbatches'] for n in b['response_lengths']]
        require(len(local) == 4 and sum(b['rows'] for b in r['microbatches']) == 4
                and all(b['rows'] == len(b['response_lengths']) and b['response_width'] == limit
                        for b in r['microbatches'])
                and all(type(n) is int and 1 <= n <= limit for n in local), 'Invalid rank lengths')
        lengths.extend(local)
        if r['step'] > warmup and any(n >= 1000 for n in local):
            eligible[r['rank']].append(r['step'])
    for entry in report['actor_inputs']:
        local = [n for r in records if r['step'] == entry['step']
                 for b in r['microbatches'] for n in b['response_lengths']]
        require(entry['rows'] == 8 and sorted(entry['response_lengths']) == sorted(local),
                'Global/rank lengths disagree')
    output = Path(manifest['output_dir'])
    first = reader.json(output / 'evidence/first_batch_length_gate.json')
    contract = dict(expected_rows=8, minimum_tokens=1000, maximum_tokens=1024)
    require(first == report['first_batch_length_gate'] and first['status'] == 'PASS_FIRST_BATCH_LENGTH'
            and first['contract'] == contract and manifest['first_batch_length_gate'] == contract
            and first['step'] == 1 and first['response_width'] == limit
            and first['checked_before'] == 'balance_ref_logprob_actor_update' and not first['errors']
            and len(first['response_lengths']) == 8
            and all(type(n) is int and 1000 <= n <= limit for n in first['response_lengths'])
            and sorted(first['response_lengths']) == sorted(report['actor_inputs'][0]['response_lengths']),
            'First-batch length gate mismatch')
    coverage = {'passed': all(len(set(v)) >= 2 for v in eligible.values()),
                'per_rank_post_warmup_long_steps': {str(k): v for k, v in eligible.items()},
                'minimum_response_tokens': 1000}
    require(coverage == report['coverage'], 'Pressure coverage summary mismatch')
    gpu = reader.jsonl(output / 'evidence/telemetry/gpu.jsonl')
    cpu = reader.jsonl(output / 'evidence/telemetry/cgroup_memory.jsonl')
    hardware = tuple(tuple(g) for g in report['hardware'])
    require(len(hardware) == 2 and {g[0] for g in hardware} == {0, 1}, 'Missing GPU identity')
    require(gpu and cpu, 'Empty telemetry')
    require(all(tuple(sorted((g['index'], g['uuid'], g['memory_total_bytes']) for g in row['gpus'])) == hardware
                for row in gpu), 'GPU identity/capacity changed')
    require(all(type(g['memory_used_bytes']) is int and 0 <= g['memory_used_bytes'] <= g['memory_total_bytes']
                for row in gpu for g in row['gpus']), 'Invalid physical GPU samples')
    require(all(row['memory_max_bytes'] == report['cpu_capacity_bytes'] > 0 for row in cpu), 'CPU capacity changed')
    require(all(type(row['memory_current_bytes']) is int
                and 0 <= row['memory_current_bytes'] <= row['memory_max_bytes'] for row in cpu), 'Invalid CPU samples')
    require(all(cpu[-1]['memory_events'][k] - cpu[0]['memory_events'][k] == 0
                for k in ('oom', 'oom_kill')), 'cgroup OOM events increased')
    marker_files = sorted((output / 'evidence/memory_stages').glob('*.jsonl'))
    require(len(marker_files) == 2, 'Missing or extra rank stage traces')
    require({p.name.split('.')[0] for p in marker_files} == {'rank0', 'rank1'}, 'Missing rank stage trace')
    traces = [reader.jsonl(p) for p in marker_files]
    require(all({row['global_step'] for row in trace} == set(range(1, 17)) for trace in traces),
            'Incomplete stage trace steps')
    markers = [row for trace in traces for row in trace]
    require(all(row['synchronization_enabled'] is True and type(row['device_free_bytes']) is int
                and 0 <= row['device_free_bytes'] <= row['device_total_bytes']
                and row['device_total_bytes'] > 0 for row in markers), 'Invalid/unsynchronized physical markers')
    values = dict(gpu_peak_ratio=max(g['memory_used_bytes'] / g['memory_total_bytes'] for row in gpu for g in row['gpus']),
                  marker_peak_ratio=max(1 - row['device_free_bytes'] / row['device_total_bytes'] for row in markers),
                  cpu_peak_ratio=max(row['memory_current_bytes'] / row['memory_max_bytes'] for row in cpu))
    for key, value in values.items():
        require(math.isfinite(value) and 0 <= value <= 1 and math.isclose(value, report[key], abs_tol=1e-12),
                f'Physical telemetry summary mismatch: {key}')
    return dict(**values, coverage=coverage, observed_steps=16, response_count=len(lengths),
                minimum_response_tokens=min(lengths), maximum_response_tokens=max(lengths),
                all_responses_at_limit=all(n == limit for n in lengths),
                cpu_capacity_bytes=report['cpu_capacity_bytes'],
                cpu_peak_bytes=max(row['memory_current_bytes'] for row in cpu), hardware=report['hardware'])


def collect_validation_evidence(paths):
    reader = Reader()
    result = dict(status='FAIL_VALIDATION_EVIDENCE', errors=[], formal_training_authorized=False,
                  whole_run_causal_claim_allowed=False, optimization_validated=False)
    try:
        baseline, _, _ = verify_run(reader, paths['fixed_baseline'], 'fixed_baseline')
        deferred, _, _ = verify_run(reader, paths['fixed_deferred'], 'fixed_deferred')
        comparison = reader.json(paths['fixed_comparison'])
        require(comparison['runs'] == {'baseline': baseline, 'deferred': deferred}, 'Comparison run binding mismatch')
        checks = dict(variants=baseline['stage'] == 'fixed_baseline' and deferred['stage'] == 'fixed_deferred',
                      hardware=baseline['hardware'] == deferred['hardware'],
                      cpu_capacity=baseline['cpu_capacity_bytes'] == deferred['cpu_capacity_bytes'],
                      runtime_sources=baseline['runtime_sha256'] == deferred['runtime_sha256'],
                      microbatches=baseline['microbatch_records'] == deferred['microbatch_records'],
                      full_actor_payloads=[e['sha256'] for e in baseline['actor_inputs']] ==
                                          [e['sha256'] for e in deferred['actor_inputs']])
        require(all(checks.values()) and checks == comparison['checks'], 'Fixed Actor inputs do not match')
        require(comparison['status'] == 'PASS_MATCHED_ACTOR_INPUTS_PENDING_MEMORY_REVIEW'
                and all(comparison[k] is False for k in ('formal_training_authorized',
                        'optimization_validated', 'whole_run_causal_claim_allowed')), 'Unsupported causal/release claim')
        pressure, manifest, config = verify_run(reader, paths['pressure'], 'pressure')
        require(pressure['hardware'] == deferred['hardware']
                and pressure['cpu_capacity_bytes'] == deferred['cpu_capacity_bytes'], 'Pressure hardware mismatch')
        metrics = pressure_metrics(reader, pressure, manifest, config)
        # Pressure v2 is a repaired isolated runtime. Expose differences rather
        # than treating it as the same runtime as fixed replay or the formal code.
        b_sources, p_sources = deferred['runtime_sha256'], pressure['runtime_sha256']
        differences = sorted(k for k in b_sources.keys() | p_sources.keys()
                             if b_sources.get(k) != p_sources.get(k))
        result.update(status='PASS_DIAGNOSTIC_EVIDENCE', fixed_comparison_status=comparison['status'],
                      pressure=metrics, pressure_manifest_sha256=pressure['manifest_sha256'],
                      fixed_deferred_to_pressure_runtime_differences=differences,
                      scope='Historical diagnostics only; not final-candidate validation or training-resume proof.')
    except (OSError, ValueError, KeyError, TypeError, IndexError, ZeroDivisionError) as exc:
        result['errors'].append(str(exc))
    result['sources'] = {p: h for p, h in sorted(reader.sources.items())}
    return result
