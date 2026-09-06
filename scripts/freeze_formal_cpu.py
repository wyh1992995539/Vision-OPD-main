#!/usr/bin/env python3
"""Bind the reviewed 240 GiB CPU floor without changing historical run evidence."""
import argparse
import copy
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.day11_validation_evidence import Reader, require

GIB = 1024**3
FLOOR = 240 * GIB
RUN = ROOT/'artifacts/runs/E-D11-6K-GATE-001'
DIRECTORY = RUN/'resource_refreeze_v1'
REPORT = DIRECTORY/'cpu_freeze.json'
CONFIGS = ('vopd_6241.yaml', 'vopd_6241_candidate.yaml', 'vopd_6241_abort_policy.yaml')
RUNS = ('pilot/16', 'pilot/64', 'memory_optimization/fixed_validation_v1/capture/run',
        'memory_optimization/fixed_validation_v1/fixed_baseline/run',
        'memory_optimization/fixed_validation_v1/fixed_deferred/run',
        'memory_optimization/fixed_validation_v1/pressure_v2/run')


def only_cpu_change(before, after, name):
    value = copy.deepcopy(after)
    section = 'memory' if name.endswith('abort_policy.yaml') else 'resources'
    require(before[section]['prelaunch_cgroup_minimum_bytes'] == 192*GIB, 'Unexpected old floor')
    require(after[section]['prelaunch_cgroup_minimum_bytes'] == FLOOR, 'New floor must be 240 GiB')
    value[section]['prelaunch_cgroup_minimum_bytes'] = before[section]['prelaunch_cgroup_minimum_bytes']
    if name == 'vopd_6241_candidate.yaml':
        require(after['candidate']['resource_floor_refreeze_required'] is False, 'Candidate floor still pending')
        require(after['candidate']['cpu_freeze'] == str(REPORT.relative_to(ROOT)), 'Wrong CPU freeze reference')
        value['candidate'].pop('cpu_freeze')
        value['candidate']['resource_floor_refreeze_required'] = True
    require(value == before, 'Non-CPU semantic change in '+name)


def cpu_review(reader):
    rows = []
    for relative in RUNS:
        directory = RUN/relative
        report = reader.json(directory/'evidence/postflight.json')
        require(report['status'] in ('PASS', 'PASS_CAPTURE', 'PASS_FIXED_ACTOR_RUN', 'PASS_PRESSURE_DIAGNOSTIC'),
                'Unsuccessful CPU evidence run')
        audit = report.get('training_audit', report)
        require(audit['training_gate_pass'] is True, 'Training audit failed')
        for entry in audit['inputs'].values():
            reader.bound(entry['path'], entry['sha256'])
        samples = reader.jsonl(directory/'evidence/telemetry/cgroup_memory.jsonl')
        require(samples and all(x['memory_max_bytes'] == FLOOR for x in samples), 'Unverified/different runtime capacity')
        require(all(type(x['memory_current_bytes']) is int and 0 < x['memory_current_bytes'] <= FLOOR for x in samples),
                'Invalid CPU samples')
        require(all(samples[-1]['memory_events'][k] == samples[0]['memory_events'][k] for k in ('oom', 'oom_kill')),
                'OOM events changed')
        peak = max(x['memory_current_bytes'] for x in samples)
        require(peak < .95*FLOOR, 'CPU peak reaches abort floor')
        rows.append(dict(run=relative, peak_bytes=peak, peak_gib=peak/GIB, capacity_bytes=FLOOR,
                         samples=len(samples), oom_delta=0, oom_kill_delta=0))
    peak = max(row['peak_bytes'] for row in rows)
    return dict(runs=rows, maximum_peak_bytes=peak, maximum_peak_gib=peak/GIB,
                minimum_bytes=FLOOR, minimum_gib=240, abort_ratio=.95,
                abort_threshold_gib=228, headroom_to_capacity_gib=(FLOOR-peak)/GIB,
                headroom_to_abort_gib=(.95*FLOOR-peak)/GIB,
                basis='Freeze the actual successful 240 GiB capacity; do not claim 224/220 GiB is validated.',
                latest_pressure_is_not_only_peak_source=True)


def build():
    reader = Reader()
    evidence = cpu_review(reader)
    changes = []
    for name in CONFIGS:
        before, after = DIRECTORY/'before'/name, ROOT/'configs'/name
        only_cpu_change(yaml.safe_load(before.read_text()), yaml.safe_load(after.read_text()), name)
        changes.append(dict(name=name, before_path=str(before), before_sha256=reader.digest(before),
                            after_path=str(after), after_sha256=reader.digest(after)))
    reader.digest(Path(__file__))
    return dict(status='PASS_CPU_FLOOR_FROZEN', formal_training_authorized=False, gpu_used=False,
                review=evidence, changes=changes, sources=reader.sources,
                current_cpu_capacity_is_launch_evidence=False,
                limits=['Historical quota is not current quota; guarded launch must freshly read its own cgroup.',
                        'GPU safety, disk, final-candidate validation and budget are separate gates.',
                        'Historical static/config/budget freezes are preserved; changed source bindings need final refresh.'])


def verify(path=REPORT):
    try:
        report = json.loads(Path(path).read_text())
        require(report['status'] == 'PASS_CPU_FLOOR_FROZEN' and report['formal_training_authorized'] is False,
                'Wrong CPU freeze status')
        reader = Reader()
        require(report['sources'], 'Missing source bindings')
        for source, expected in report['sources'].items():
            reader.bound(source, expected)
        require(report['review'] == cpu_review(reader), 'CPU review does not recompute')
        require([c['name'] for c in report['changes']] == list(CONFIGS), 'CPU config set changed')
        for c in report['changes']:
            name = c['name']
            require(Path(c['before_path']) == DIRECTORY/'before'/name
                    and Path(c['after_path']) == ROOT/'configs'/name, 'CPU change target mismatch')
            reader.bound(c['before_path'], c['before_sha256'])
            reader.bound(c['after_path'], c['after_sha256'])
            only_cpu_change(yaml.safe_load(Path(c['before_path']).read_text()),
                            yaml.safe_load(Path(c['after_path']).read_text()), name)
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=REPORT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve old CPU freeze; choose a new output')
    value = build()
    with args.output.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    with args.output.with_suffix('.sha256').open('x') as stream:
        stream.write(f'{Reader().digest(args.output)}  {args.output}\n')
    print(value['status'])
    print(json.dumps(value['review'], ensure_ascii=False))


if __name__ == '__main__':
    main()
