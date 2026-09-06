#!/usr/bin/env python3
"""Compare fresh A/B audits without promoting a diagnostic run to formal training."""
import argparse
import copy
import json
from pathlib import Path
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.audit_vopd_memory_ab import audit_memory, comparable_config
from scripts.monitor_vopd_training import utc_now, write_json
from scripts.run_vopd_6241_pilot_guarded import resolve, sha256_file, load_pilot_policy, static_preflight
from scripts.vopd_memory_experiment import BASE, ROOT, SOURCE_PATHS, OFFLINE_AUDIT_SOURCES

# Predeclared engineering criterion, not a paper hyperparameter or a statistical claim.
MIN_REDUCTION_BYTES = 512 * 1024**2


def source_compatibility(baseline, deferred):
    """Accept audit-only revisions only with complete, freshly verifiable provenance.

    The CLI always re-audits raw runs. This helper additionally rejects stale or
    incomplete source receipts instead of merely deleting audit keys from equality.
    """
    result = dict(compatible=False, mode='REJECTED', differing_sources=[], errors=[])
    try:
        b, d = baseline['source_hashes'], deferred['source_hashes']
        if b == d:
            result.update(compatible=True, mode='EXACT_LAUNCH_SOURCES')
            return result
        if set(b) != set(SOURCE_PATHS) or set(d) != set(SOURCE_PATHS):
            raise ValueError('Incomplete launch source coverage')
        differences = sorted(k for k in b if b[k] != d[k])
        result['differing_sources'] = differences
        if not set(differences) <= OFFLINE_AUDIT_SOURCES:
            raise ValueError('Training/collector/launcher source differs')
        current = {p: sha256_file(ROOT / p) for p in SOURCE_PATHS}
        for report in (baseline, deferred):
            launched = report['source_hashes']
            provenance = report['audit_provenance']
            if (provenance['evaluated_source_hashes'] != current
                    or provenance['training_source_hashes_unchanged'] is not True
                    or provenance['original_launch_manifest_preserved'] is not True):
                raise ValueError('Stale evaluator or unverified launch provenance')
            revised = {k for k in launched if launched[k] != current[k]}
            receipts = provenance['source_revisions']
            if (not revised <= OFFLINE_AUDIT_SOURCES or set(receipts) != revised
                    or (revised and provenance['offline_reaudit'] is not True)):
                raise ValueError('Missing or excessive audit revision receipts')
            for key in revised:
                entry = receipts[key]
                if not (entry['launch_sha256'] == entry['archived_sha256'] == launched[key]
                        == sha256_file(Path(entry['archived_path']))
                        and entry['audit_sha256'] == current[key]):
                    raise ValueError(f'Invalid archived audit source: {key}')
        result.update(compatible=True, mode='VERIFIED_AUDIT_ONLY_REVISION', evaluated_source_hashes=current)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result['errors'].append(str(exc))
    return result


def comparable_policy(policy):
    result = copy.deepcopy(policy)
    result.pop('experiment_id', None)  # load_pilot_policy injects the selected stage identity
    result.pop('memory_experiment', None)
    for stage in result['pilot']['stage_contracts'].values():
        for field in ('experiment_id', 'config', 'output_dir'):
            stage.pop(field, None)
    return result


def check_preparation(baseline_policy, deferred_policy, snapshot):
    """CPU-only readiness of the pairing, not live resources or optimization success."""
    result = dict(schema_version=1, generated_at_utc=utc_now(), status='FAIL_COMPARISON_PREPARATION',
                  formal_training_authorized=False, optimization_validated=False, training_started=False,
                  live_gpu_validation_pending=True, checks={}, errors=[])
    try:
        baseline = audit_memory(baseline_policy, audit_source_snapshot=snapshot)
        result['baseline_audit'] = baseline
        result['checks']['baseline_evidence_pass'] = baseline['stage_gate_pass']
        if not baseline['stage_gate_pass']:
            return result
        bp, _ = load_pilot_policy(baseline_policy, '64')
        dp, contract = load_pilot_policy(deferred_policy, '64')
        static = static_preflight(deferred_policy, '64')
        result['deferred_static_preflight'] = static
        result['checks']['deferred_static_pass'] = static['status'] == 'PASS'
        result['checks']['deferred_not_started'] = not resolve(contract['output_dir']).exists()
        result['checks']['deferred_variant'] = dp['memory_experiment']['variant'] == 'deferred'
        cfg = yaml.safe_load(resolve(contract['config']).read_text())
        manifest = json.loads(Path(dp['memory_experiment']['manifest']).read_text())
        candidate = dict(source_hashes=manifest['source_hashes'], audit_provenance=dict(
            offline_reaudit=False, source_revisions={},
            evaluated_source_hashes={p: sha256_file(ROOT / p) for p in SOURCE_PATHS},
            training_source_hashes_unchanged=static['status'] == 'PASS', original_launch_manifest_preserved=True))
        compatibility = source_compatibility(baseline, candidate)
        result['source_compatibility'] = compatibility
        result['checks']['source_compatible'] = compatibility['compatible']
        result['checks']['algorithm_config_match'] = baseline['comparison_config'] == comparable_config(cfg)
        result['checks']['guard_policy_match'] = comparable_policy(bp) == comparable_policy(dp)
        selection = json.loads(resolve(cfg['paths']['selection_manifest']).read_text())
        result['checks']['train_bytes_match'] = baseline['train_sha256'] == sha256_file(resolve(cfg['paths']['train_file'])) == selection['output']['sha256']
        result['checks']['sample_order_match'] = baseline['sample_ids'] == [s['sample_id'] for s in selection['samples']]
        result['deferred_manifest'] = dict(path=dp['memory_experiment']['manifest'], sha256=sha256_file(Path(dp['memory_experiment']['manifest'])))
        if all(result['checks'].values()):
            result['status'] = 'PASS_COMPARISON_PREPARATION'
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result['errors'].append(str(exc))
    return result


def compare_reports(baseline, deferred):
    result = dict(schema_version=1, generated_at_utc=utc_now(), status='FAIL_AB_EVIDENCE',
                  formal_training_authorized=False, optimization_validated=False,
                  criteria=dict(minimum_worst_device_reduction_bytes=MIN_REDUCTION_BYTES,
                                no_per_device_peak_regression=True, workload_must_match=True), checks={})
    result['runs'] = {'baseline': baseline, 'deferred': deferred}
    if any(r['status'] not in ('NOT_RUN', 'PASS_MEMORY_AB_RUN') for r in (baseline, deferred)):
        return result
    if any(r['status'] == 'NOT_RUN' for r in (baseline, deferred)):
        result['status'] = 'WAITING_FOR_RUNS'
        return result
    if not all(r['status'] == 'PASS_MEMORY_AB_RUN' and r['stage_gate_pass'] for r in (baseline, deferred)):
        return result
    b, d = baseline, deferred
    checks = result['checks']
    for key in ('comparison_config', 'train_sha256', 'sample_ids', 'hardware',
                'cpu_capacity_bytes', 'gpu_abort_ratio', 'cpu_abort_ratio'):
        checks[key + '_match'] = b[key] == d[key]
    result['source_compatibility'] = source_compatibility(b, d)
    checks['source_hashes_match_or_verified_audit_revision'] = result['source_compatibility']['compatible']
    checks['variant_order'] = b['variant'] == 'baseline' and d['variant'] == 'deferred'
    checks['sequential_run_windows'] = b['run_end_unix'] < d['run_start_unix']
    if not all(checks.values()):
        result['status'] = 'FAIL_AB_COMPARABILITY'
        return result
    result['device_reduction_bytes'] = {k: b['device_peak_bytes'][k] - d['device_peak_bytes'][k] for k in b['device_peak_bytes']}
    result['worst_device_reduction_bytes'] = max(b['device_peak_bytes'].values()) - max(d['device_peak_bytes'].values())
    result['phase_reduction_bytes'] = {
        key: {metric: b['memory_stages']['phase_peaks'][key][metric] - value[metric]
              for metric in ('allocated_bytes', 'reserved_bytes')}
        for key, value in d['memory_stages']['phase_peaks'].items() if key in b['memory_stages']['phase_peaks']}
    result['cpu_peak_increase_bytes'] = d['cpu_peak_bytes'] - b['cpu_peak_bytes']
    result['wall_time_increase_seconds'] = d['wall_seconds'] - b['wall_seconds']
    result['step_time_increase_seconds'] = sum(d['step_seconds']) - sum(b['step_seconds'])
    checks['matching_observed_workload'] = b['workload'] == d['workload'] and b['memory_stages']['forward_shapes'] == d['memory_stages']['forward_shapes']
    checks['candidate_below_memory_abort_lines'] = (
        max(d['device_peak_ratios'].values()) < d['gpu_abort_ratio'] and
        d['marker_peak_ratio'] < d['gpu_abort_ratio'] and d['cpu_peak_ratio'] < d['cpu_abort_ratio'])
    checks['no_gpu_regression'] = all(v >= 0 for v in result['device_reduction_bytes'].values())
    checks['meaningful_observed_reduction'] = result['worst_device_reduction_bytes'] >= MIN_REDUCTION_BYTES
    if not checks['candidate_below_memory_abort_lines']:
        result['status'] = 'FAIL_MEMORY_HEADROOM'
    elif not checks['matching_observed_workload']:
        result['status'] = 'REVIEW_WORKLOAD_DIFFERENCE'
    elif not checks['no_gpu_regression']:
        result['status'] = 'FAIL_GPU_PEAK_REGRESSION'
    elif not checks['meaningful_observed_reduction']:
        result['status'] = 'NO_CLEAR_MEMORY_BENEFIT'
    else:
        result['status'] = 'PASS_OBSERVED_MEMORY_REDUCTION'
        result['optimization_validated'] = True
    result['validation_scope'] = 'Observed memory benefit on the matched 64-row diagnostic runs only'
    result['limitations'] = [
        'A single pair is not a causal or statistical proof; matching length summaries is not matching generated tokens.',
        'NVML samples may miss spikes; actor phase allocator peaks and synchronized device markers are separate measurements.',
        'Cold reload, post-warmup and long-response training gates remain required outside this diagnostic comparison.',
        'Timing includes synchronized profiling and must not replace the formal budget estimate.']
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-policy', type=Path, default=BASE / 'ab/baseline/policy.yaml')
    parser.add_argument('--deferred-policy', type=Path, default=BASE / 'ab/deferred/policy.yaml')
    parser.add_argument('--baseline-audit-source-snapshot', type=Path,
                        help='Explicit launch-time audit archive for historical baseline re-audit')
    parser.add_argument('--preflight-only', action='store_true', help='CPU-only planned pairing checks; no training')
    parser.add_argument('--output', type=Path, default=BASE / 'ab/comparison.json')
    args = parser.parse_args()
    # Re-read raw evidence; a stale manually edited PASS report is never sufficient.
    if args.preflight_only:
        result = check_preparation(resolve(args.baseline_policy), resolve(args.deferred_policy), args.baseline_audit_source_snapshot)
    else:
        baseline = audit_memory(resolve(args.baseline_policy), audit_source_snapshot=args.baseline_audit_source_snapshot)
        # Candidate must pass current-source binding; no offline exception on launch/candidate.
        result = compare_reports(baseline, audit_memory(resolve(args.deferred_policy)))
    path = resolve(args.output)
    if any(p.exists() for p in (path, path.with_suffix('.md'), path.with_suffix('.sha256'))):
        raise FileExistsError('Refusing to overwrite comparison evidence; choose a new --output')
    write_json(path, result)
    md = path.with_suffix('.md')
    lines = [f"# Memory A/B comparison\n\nStatus: `{result['status']}`", '\nFormal training authorized: `false`',
             '\nChecks:\n']
    lines.extend(f"- {key}: {value}" for key, value in result['checks'].items())
    if 'device_reduction_bytes' in result:
        lines.append('\n| GPU | Baseline peak GiB | Deferred peak GiB | Reduction GiB |\n| --- | ---: | ---: | ---: |')
        for key, delta in result['device_reduction_bytes'].items():
            b, d = result['runs']['baseline'], result['runs']['deferred']
            lines.append(f"| {key} | {b['device_peak_bytes'][key]/1024**3:.3f} | {d['device_peak_bytes'][key]/1024**3:.3f} | {delta/1024**3:.3f} |")
        lines.append(f"\nCPU peak increase: {result['cpu_peak_increase_bytes']/1024**3:.3f} GiB; wall time increase: {result['wall_time_increase_seconds']:.2f} seconds.")
    lines.extend('\n- ' + item for item in result.get('limitations', []))
    md.write_text('\n'.join(lines) + '\n')
    path.with_suffix('.sha256').write_text(''.join(f'{sha256_file(p)}  {p}\n' for p in (path, md)))
    print(f"MEMORY_AB_COMPARISON={result['status']}\nOUTPUT={path}")
    return 0 if result['optimization_validated'] or result['status'] == 'PASS_COMPARISON_PREPARATION' else 2 if result['status'] == 'WAITING_FOR_RUNS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
