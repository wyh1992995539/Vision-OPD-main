#!/usr/bin/env python3
"""Audit one diagnostic memory run; never authorizes formal training."""
import argparse
import copy
import datetime as dt
import json
import math
from pathlib import Path
import re
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_vopd_6241_pilot import audit as pilot_audit, read_jsonl
from scripts.run_vopd_6241_pilot_guarded import load_pilot_policy, resolve, sha256_file
from scripts.monitor_vopd_training import utc_now, write_json
from scripts.vopd_memory_experiment import memory_overrides

PAIRED = ('student_forward', 'teacher_forward', 'backward', 'optimizer_load', 'optimizer_step', 'teacher_ema')
BYTE_FIELDS = ('allocated_bytes', 'reserved_bytes', 'interval_peak_allocated_bytes',
               'interval_peak_reserved_bytes', 'device_free_bytes', 'device_total_bytes')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def stage_summary(directory, steps, variant):
    """Validate complete per-rank actor-update intervals, ignoring valid outside-update markers."""
    ranks, peaks, shapes = {}, {}, []
    files = sorted(directory.glob('*.jsonl'))
    require(bool(files), 'Missing memory stage JSONL')
    for path in files:
        match = re.fullmatch(r'rank(\d+)\.pid(\d+)\.jsonl', path.name)
        require(match is not None, f'Unexpected stage filename: {path.name}')
        rank, pid = map(int, match.groups())
        require(rank not in ranks, f'Multiple processes for rank {rank}')
        rows = read_jsonl(path)
        require(bool(rows), f'Empty stage file: {path}')
        previous, last_time, active, completed = None, -1, None, []
        for row in rows:
            name = row['event']
            stamp = row['monotonic_seconds']
            require(row['pid'] == pid and row['synchronization_enabled'] is True, 'PID or synchronization mismatch')
            require(isinstance(stamp, (int, float)) and math.isfinite(stamp) and stamp >= last_time, 'Invalid stage time')
            require(row['interval_start'] == previous, 'Broken memory interval chain')
            require(all(type(row[k]) is int and row[k] >= 0 for k in BYTE_FIELDS), 'Invalid memory bytes')
            require(0 <= row['allocated_bytes'] <= row['reserved_bytes'] <= row['device_total_bytes'], 'Invalid memory ordering')
            require(row['interval_peak_allocated_bytes'] >= row['allocated_bytes'] and
                    row['interval_peak_reserved_bytes'] >= row['reserved_bytes'] and
                    row['device_total_bytes'] > 0 and row['device_free_bytes'] <= row['device_total_bytes'], 'Invalid memory peaks')
            require(not name.endswith('/error'), 'Error marker in memory trace')
            previous, last_time = name, stamp
            if name == 'actor_update_entry_after_rollout':
                require(active is None, 'Nested actor update')
                require(type(row['global_step']) is int, 'Missing global step')
                active = []
            if active is None:
                continue
            active.append(row)
            require(row['global_step'] == active[0]['global_step'], 'Step changed within update')
            if name == 'actor_update_exit':
                step = row['global_step']
                names = [r['event'] for r in active]
                pairs = PAIRED + (('optimizer_offload',) if variant == 'deferred' else ())
                for phase in pairs:
                    opened, count = False, 0
                    for event in names:
                        if event == phase + '/before':
                            require(not opened, f'Nested {phase}')
                            opened = True
                        elif event == phase + '/after':
                            require(opened, f'Unpaired {phase}')
                            opened, count = False, count + 1
                    require(not opened and count > 0, f'Missing/incomplete {phase}: rank={rank}, step={step}')
                require(names.index('student_forward/before') < names.index('teacher_forward/before') <
                        names.index('backward/before') < names.index('optimizer_step/before') <
                        names.index('teacher_ema/before'), 'Invalid phase order')
                if variant == 'deferred':
                    require(names.index('backward/after') < names.index('optimizer_load/before') and
                            names.index('optimizer_step/after') < names.index('optimizer_offload/before') <
                            names.index('teacher_ema/before'), 'Deferred state residency order violated')
                else:
                    require(names.index('optimizer_load/after') < names.index('student_forward/before'), 'Baseline load is not eager')
                for event in active:
                    event_name = event['event']
                    # /after interval contains the actual phase peak; never label a boundary gap as that phase.
                    if event_name.endswith('/after') and event['interval_start'] == event_name[:-6] + '/before':
                        key = f'{rank}/{event_name[:-6]}'
                        record = peaks.setdefault(key, dict(allocated_bytes=0, reserved_bytes=0, intervals=0))
                        record['allocated_bytes'] = max(record['allocated_bytes'], event['interval_peak_allocated_bytes'])
                        record['reserved_bytes'] = max(record['reserved_bytes'], event['interval_peak_reserved_bytes'])
                        record['intervals'] += 1
                    if event_name in ('student_forward/before', 'teacher_forward/before'):
                        fields = ('micro_batch_samples', 'sequence_width', 'max_unpadded_sequence_tokens', 'response_width')
                        require(all(type(event.get(k)) is int and event[k] > 0 for k in fields), 'Missing forward shape context')
                        shapes.append([rank, step, event_name, *[event[k] for k in fields]])
                completed.append(step)
                active = None
        require(active is None and completed == list(range(1, steps + 1)), f'Incomplete/duplicate steps for rank {rank}')
        ranks[rank] = dict(pid=pid, completed_steps=completed, path=str(path), sha256=sha256_file(path))
    require(set(ranks) == {0, 1}, 'Expected exactly ranks 0 and 1')
    return dict(ranks=ranks, phase_peaks=peaks, forward_shapes=shapes)


def comparable_config(config):
    value = copy.deepcopy(config)
    for key in ('id', 'name', 'group_name'):
        value['experiment'].pop(key, None)
    for key in ('output_dir', 'selection_manifest'):
        value['paths'].pop(key, None)
    return value


def audit_memory(policy_path):
    report = dict(schema_version=1, generated_at_utc=utc_now(), status='FAIL_MEMORY_AB_RUN',
                  training_gate_pass=False, stage_gate_pass=False, formal_training_authorized=False,
                  cold_reload_required=False, cold_reload_validated=False,
                  optimization_validated=False, errors=[])
    try:
        policy, contract = load_pilot_policy(policy_path, '64')
        config_path, output = resolve(contract['config']), resolve(contract['output_dir'])
        expected = memory_overrides(policy, config_path, output)
        require(bool(expected), 'Not an isolated memory experiment')
        variant = policy['memory_experiment']['variant']
        report.update(variant=variant, experiment_id=contract['experiment_id'], output_dir=str(output))
        config = yaml.safe_load(config_path.read_text())
        if not (output / 'preflight/run_invocation.json').exists() and not (output / 'logs/train.log').exists():
            report['status'] = 'NOT_RUN'
            return report
        generic = pilot_audit('64', policy_path)
        report['training_audit'] = generic
        report['training_gate_pass'] = generic['training_gate_pass']
        require(generic['training_gate_pass'], f"Training audit failed: {generic.get('failed_checks', generic.get('missing_inputs'))}")
        invocation = json.loads((output / 'preflight/run_invocation.json').read_text())
        require(invocation.get('hydra_overrides') == expected, 'Executed Hydra overrides mismatch')
        live = json.loads((output / 'preflight/pilot_live_launch_gate.json').read_text())
        require(live['status'] == 'PASS' and live['policy_sha256'] == sha256_file(policy_path) and
                live['config_sha256'] == sha256_file(config_path) and live['experiment_id'] == contract['experiment_id'], 'Live launch binding mismatch')
        require(live.get('effective_policy') == policy, 'Live effective policy mismatch')
        trace = stage_summary(output / 'evidence/memory_stages', int(contract['expected_optimizer_steps']), variant)
        gpu_rows = read_jsonl(output / 'evidence/telemetry/gpu.jsonl')
        cpu_rows = read_jsonl(output / 'evidence/telemetry/cgroup_memory.jsonl')
        require(bool(gpu_rows) and bool(cpu_rows), 'Empty runtime telemetry')
        device_peaks, hardware = {}, set()
        last_elapsed = -1
        for sample in gpu_rows:
            require(math.isfinite(sample['elapsed_seconds']) and sample['elapsed_seconds'] >= last_elapsed, 'Invalid GPU sample time')
            last_elapsed = sample['elapsed_seconds']
            require({g['index'] for g in sample['gpus']} == {0, 1} and len(sample['gpus']) == 2, 'Incomplete GPU sample')
            identity = []
            for g in sample['gpus']:
                require(type(g['memory_used_bytes']) is int and 0 <= g['memory_used_bytes'] <= g['memory_total_bytes'] and g['memory_total_bytes'] > 0, 'Invalid device memory')
                identity.append((g['index'], g['uuid'], g['memory_total_bytes']))
                key = str(g['index'])
                device_peaks[key] = max(device_peaks.get(key, 0), g['memory_used_bytes'])
            hardware.add(tuple(sorted(identity)))
        require(len(hardware) == 1, 'GPU identity changed during run')
        capacities, cpu_ratios, cpu_peak = set(), [], 0
        for r in cpu_rows:
            used, cap = r['memory_current_bytes'], r['memory_max_bytes']
            require(r['supported'] is True and type(used) is int and type(cap) is int and 0 <= used <= cap and cap > 0, 'Invalid cgroup sample')
            capacities.add(cap)
            cpu_peak = max(cpu_peak, used)
            cpu_ratios.append(used / cap)
        require(len(capacities) == 1, 'CPU quota changed during run')
        for event in ('oom', 'oom_kill'):
            counts = [r['memory_events'][event] for r in cpu_rows]
            require(len(set(counts)) == 1, f'cgroup {event} counter changed')
        hardware = list(next(iter(hardware)))
        peak_ratios = {str(i): device_peaks[str(i)] / total for i, _, total in hardware}
        # Rank-to-physical-device mapping is not assumed. Keep marker values separate.
        marker_peak_ratio = 0
        for info in trace['ranks'].values():
            rows = read_jsonl(Path(info['path']))
            for row in rows:
                marker_peak_ratio = max(marker_peak_ratio, 1 - row['device_free_bytes'] / row['device_total_bytes'])
        steps = generic['steps']
        manifest_path = Path(policy['memory_experiment']['manifest'])
        manifest = json.loads(manifest_path.read_text())
        require(live.get('memory_experiment_manifest') == manifest, 'Launch-time source manifest mismatch')
        start = dt.datetime.fromisoformat(gpu_rows[0]['timestamp_utc']).timestamp()
        guard = json.loads((output / 'evidence/guard_summary.json').read_text())
        end = dt.datetime.fromisoformat(guard['finished_at_utc']).timestamp()
        require(start < end, 'Invalid runtime window')
        require(all(start <= r['time_unix'] <= end for v in trace['ranks'].values()
                    for r in read_jsonl(Path(v['path']))), 'Stage trace outside runtime window')
        require(all(isinstance(r['step_seconds'], (int, float)) and math.isfinite(r['step_seconds']) and r['step_seconds'] > 0
                    for r in steps), 'Missing/invalid step timing')
        paths = [policy_path, config_path, manifest_path, output / 'preflight/pilot_live_launch_gate.json',
                 *[Path(v['path']) for v in generic['inputs'].values()],
                 *sorted((output / 'evidence/telemetry').glob('*.jsonl')),
                 *sorted((output / 'evidence/memory_stages').glob('*.jsonl'))]
        report.update(status='PASS_MEMORY_AB_RUN', stage_gate_pass=True, memory_stages=trace,
                      comparison_config=comparable_config(config), source_hashes=manifest['source_hashes'],
                      train_sha256=invocation['train_file_sha256'], sample_ids=invocation['sample_ids'],
                      hardware=hardware, cpu_capacity_bytes=next(iter(capacities)),
                      run_start_unix=start, run_end_unix=end, marker_peak_ratio=marker_peak_ratio,
                      device_peak_bytes=device_peaks, device_peak_ratios=peak_ratios,
                      cpu_peak_bytes=cpu_peak, cpu_peak_ratio=max(cpu_ratios),
                      gpu_abort_ratio=policy['memory']['gpu_used_ratio_abort'], cpu_abort_ratio=policy['memory']['cgroup_used_ratio_abort'],
                      wall_seconds=generic['telemetry']['max_observed_elapsed_seconds'],
                      step_seconds=[r['step_seconds'] for r in steps],
                      workload=[[r['step'], r['prompt_max_tokens'], r['response_mean_tokens'], r['response_max_tokens']] for r in steps],
                      coverage=dict(post_warmup_steps=sum(r['step'] > config['actor']['lr_warmup_steps'] for r in steps),
                                    max_response_tokens=max(r['response_max_tokens'] for r in steps)),
                      inputs=[dict(path=str(p.resolve()), sha256=sha256_file(p)) for p in sorted(set(paths))],
                      limitations=['Sampled whole-device peaks may miss spikes; stage peaks are process allocator intervals.',
                                   'Cold reload and formal long-response/post-warmup safety remain separate gates.',
                                   'Synchronized profiling affects throughput; no formal budget projection is authorized.'])
        report['training_audit'].pop('projection_780', None)
    except (OSError, ValueError, KeyError, TypeError, AssertionError) as exc:
        report['errors'].append(str(exc))
    return report


def write_report(path, report):
    write_json(path, report)
    md = path.with_suffix('.md')
    lines = [f"# Memory A/B audit\n\nStatus: `{report['status']}`", '\nFormal training authorized: `false`',
             f"\nTraining passed: `{report['training_gate_pass']}`; memory evidence passed: `{report['stage_gate_pass']}`.",
             '\nCold reload is not required for this diagnostic run; optimization is decided by the two-run comparison.']
    lines.extend('\n- ' + e for e in report.get('errors', []))
    if report['stage_gate_pass']:
        lines.append('\n| GPU | Observed peak GiB | Used ratio |\n| --- | ---: | ---: |')
        lines.extend(f"| {k} | {v / 1024**3:.3f} | {report['device_peak_ratios'][k]:.4f} |" for k, v in report['device_peak_bytes'].items())
    md.write_text('\n'.join(lines) + '\n')
    path.with_suffix('.sha256').write_text(''.join(f'{sha256_file(p)}  {p.resolve()}\n' for p in (path, md)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['64'], default='64')
    parser.add_argument('--policy', type=Path, required=True)
    args = parser.parse_args()
    policy_path = resolve(args.policy)
    policy, contract = load_pilot_policy(policy_path, '64')
    # Prevent this dedicated writer from targeting historical Pilot outputs.
    require(bool(memory_overrides(policy, resolve(contract['config']), resolve(contract['output_dir']))), 'Not an A/B policy')
    report = audit_memory(policy_path)
    if report['status'] == 'NOT_RUN':
        print('MEMORY_AB_POSTFLIGHT=NOT_RUN; no run artifacts written')
        return 2
    path = resolve(contract['output_dir']) / 'evidence/postflight.json'
    write_report(path, report)
    print(f"MEMORY_AB_POSTFLIGHT={report['status']}\nOUTPUT={path}")
    return 0 if report['stage_gate_pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
