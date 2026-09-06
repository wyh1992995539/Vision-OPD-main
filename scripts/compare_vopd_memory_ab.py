#!/usr/bin/env python3
"""Compare fresh A/B audits without promoting a diagnostic run to formal training."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.audit_vopd_memory_ab import audit_memory
from scripts.monitor_vopd_training import utc_now, write_json
from scripts.run_vopd_6241_pilot_guarded import resolve, sha256_file
from scripts.vopd_memory_experiment import BASE

# Predeclared engineering criterion, not a paper hyperparameter or a statistical claim.
MIN_REDUCTION_BYTES = 512 * 1024**2


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
    for key in ('comparison_config', 'source_hashes', 'train_sha256', 'sample_ids', 'hardware',
                'cpu_capacity_bytes', 'gpu_abort_ratio', 'cpu_abort_ratio'):
        checks[key + '_match'] = b[key] == d[key]
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
    parser.add_argument('--output', type=Path, default=BASE / 'ab/comparison.json')
    args = parser.parse_args()
    # Re-read raw evidence; a stale manually edited PASS report is never sufficient.
    result = compare_reports(audit_memory(resolve(args.baseline_policy)), audit_memory(resolve(args.deferred_policy)))
    path = resolve(args.output)
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
    return 0 if result['optimization_validated'] else 2 if result['status'] == 'WAITING_FOR_RUNS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
