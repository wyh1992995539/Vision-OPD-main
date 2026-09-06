#!/usr/bin/env python3
"""Fail-closed audit of isolated fixed workload and long-response pressure runs."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.memory_validation import check, write
from scripts.fixed_workload_io import sha, load_batch
from scripts.audit_vopd_6241_pilot import audit as pilot_audit, read_jsonl
from scripts.audit_vopd_memory_ab import stage_summary


def pressure_coverage(records, warmup=10, minimum_tokens=1000, minimum_steps=2):
    eligible = {}
    for rank in (0, 1):
        eligible[rank] = sorted(r['step'] for r in records if r['rank'] == rank and r['step'] > warmup
                                and any(n >= minimum_tokens for b in r['microbatches'] for n in b['response_lengths']))
    return dict(passed=all(len(set(steps)) >= minimum_steps for steps in eligible.values()),
                per_rank_post_warmup_long_steps=eligible, minimum_response_tokens=minimum_tokens)


def audit(policy):
    report = dict(status='FAIL_VALIDATION', stage_gate_pass=False, formal_training_authorized=False, errors=[])
    try:
        manifest = check(policy.parent)
        output = Path(manifest['output_dir'])
        launch = json.loads((output/'preflight/validation_launch.json').read_text())
        if launch['manifest_sha256'] != sha(policy.parent/'manifest.json'):
            raise ValueError('Launch manifest binding changed')
        import yaml
        effective = yaml.safe_load(policy.read_text())
        live = json.loads((output/'preflight/pilot_live_launch_gate.json').read_text())
        invocation = json.loads((output/'preflight/run_invocation.json').read_text())
        if (live['status'] != 'PASS' or live['policy_sha256'] != sha(policy)
                or live['config_sha256'] != sha(Path(effective['pilot']['stage_contracts']['64']['config']))
                or invocation['hydra_overrides'] != effective['validation_overrides']):
            raise ValueError('Executed config/overrides do not match the validation contract')
        if manifest['mode']=='replay' and launch['bundle_sha256'] != sha(manifest['bundle_manifest']):
            raise ValueError('Replay bundle changed since launch')
        first_gate = None
        if manifest.get('first_batch_length_gate') is not None:
            from scripts.pressure_runtime import CONTRACT, validate_receipt
            if manifest['first_batch_length_gate'] != CONTRACT:
                raise ValueError('Changed first-batch length contract')
            first_gate = json.loads((output/'evidence/first_batch_length_gate.json').read_text())
            report['first_batch_length_gate'] = first_gate
            if not validate_receipt(first_gate):
                raise ValueError('First-batch actual response length gate failed')
        generic = pilot_audit('64', policy)
        report['training_audit'] = generic
        if not generic['training_gate_pass']:
            raise ValueError('Training/mechanism/checkpoint audit failed')
        stage = manifest['stage']
        variant = 'deferred' if stage in ('fixed_deferred', 'pressure') else 'baseline'
        trace = stage_summary(output/'evidence/memory_stages', manifest['expected_steps'], variant)
        records, global_receipts = [], []
        for step in range(1, manifest['expected_steps']+1):
            receipt = json.loads((output/f'evidence/fixed_workload/step{step:04d}.json').read_text())
            if receipt['step'] != step or receipt['mode'] != manifest['mode'] or receipt['rows'] != 8:
                raise ValueError('Actor input receipt mismatch')
            if manifest['mode'] in ('capture', 'replay'):
                load_batch(receipt)
            global_receipts.append(receipt)
            for rank in (0, 1):
                r = json.loads((output/f'evidence/fixed_workload/step{step:04d}.rank{rank}.json').read_text())
                if r['rank'] != rank or r['step'] != step or sum(b['rows'] for b in r['microbatches']) != 4:
                    raise ValueError('Incomplete rank-local microbatch plan')
                if any(b['response_width'] != 1024 or any(n < 1 or n > 1024 for n in b['response_lengths']) for b in r['microbatches']):
                    raise ValueError('Invalid actual response lengths')
                records.append(r)
        if first_gate is not None and sorted(first_gate['response_lengths']) != sorted(global_receipts[0]['response_lengths']):
            raise ValueError('First-batch gate disagrees with actual actor input lengths')
        gpu = read_jsonl(output/'evidence/telemetry/gpu.jsonl')
        cpu = read_jsonl(output/'evidence/telemetry/cgroup_memory.jsonl')
        hardware = {tuple(sorted((g['index'], g['uuid'], g['memory_total_bytes']) for g in row['gpus'])) for row in gpu}
        if len(hardware) != 1:
            raise ValueError('GPU identity changed')
        peak = max(g['memory_used_bytes']/g['memory_total_bytes'] for row in gpu for g in row['gpus'])
        marker_peak = max(1-r['device_free_bytes']/r['device_total_bytes']
                          for f in (output/'evidence/memory_stages').glob('*.jsonl') for r in read_jsonl(f))
        cpu_peak = max(row['memory_current_bytes']/row['memory_max_bytes'] for row in cpu)
        coverage = pressure_coverage(records)
        if stage == 'pressure' and not (coverage['passed'] and peak < .98 and marker_peak < .98 and cpu_peak < .95):
            raise ValueError('Pressure coverage/headroom failed: need >=1000 tokens on both ranks in >=2 post-warmup steps')
        report.update(status='PASS_CAPTURE' if stage=='capture' else 'PASS_PRESSURE_DIAGNOSTIC' if stage=='pressure' else 'PASS_FIXED_ACTOR_RUN',
                      stage_gate_pass=True, stage=stage, microbatch_records=records, actor_inputs=global_receipts,
                      memory_stages=trace, gpu_peak_ratio=peak, marker_peak_ratio=marker_peak, cpu_peak_ratio=cpu_peak,
                      hardware=list(next(iter(hardware))), cpu_capacity_bytes=cpu[0]['memory_max_bytes'],
                      coverage=coverage, scope=manifest['runtime_hook_scope'],
                      manifest_sha256=sha(policy.parent/'manifest.json'),
                      runtime_sha256={str(Path(k).relative_to(manifest['runtime'])):v for k,v in manifest['inputs'].items()
                                      if str(k).startswith(manifest['runtime']+'/') and not k.endswith('validation_active.json')},
                      inputs=[dict(path=str(p), sha256=sha(p)) for p in sorted((output/'evidence/fixed_workload').glob('*.json'))])
    except (OSError, KeyError, ValueError, TypeError) as exc:
        report['errors'].append(str(exc))
    return report


def compare_fixed(baseline, deferred):
    # Whole-run VRAM cannot be attributed: rollout before actor replay is still stochastic.
    result = dict(status='FAIL_FIXED_COMPARISON', formal_training_authorized=False,
                  optimization_validated=False, whole_run_causal_claim_allowed=False)
    if any(r['status'] != 'PASS_FIXED_ACTOR_RUN' for r in (baseline, deferred)):
        return result
    b, d = baseline, deferred
    checks = dict(variants=b['stage']=='fixed_baseline' and d['stage']=='fixed_deferred',
        hardware=b['hardware']==d['hardware'], cpu_capacity=b['cpu_capacity_bytes']==d['cpu_capacity_bytes'],
        runtime_sources=b['runtime_sha256']==d['runtime_sha256'], microbatches=b['microbatch_records']==d['microbatch_records'],
        full_actor_payloads=[x['sha256'] for x in b['actor_inputs']]==[x['sha256'] for x in d['actor_inputs']])
    result['checks'] = checks
    if all(checks.values()):
        result['status'] = 'PASS_MATCHED_ACTOR_INPUTS_PENDING_MEMORY_REVIEW'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', default='64')
    parser.add_argument('--policy', type=Path)
    parser.add_argument('--baseline-policy', type=Path)
    parser.add_argument('--deferred-policy', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.baseline_policy or args.deferred_policy:
        if not (args.baseline_policy and args.deferred_policy and args.output) or args.policy:
            parser.error('Comparison requires both policies and a new --output')
        if args.output.exists():
            raise FileExistsError('Refusing to overwrite comparison')
        baseline, deferred = audit(args.baseline_policy.resolve()), audit(args.deferred_policy.resolve())
        report = compare_fixed(baseline, deferred)
        report['runs'] = dict(baseline=baseline, deferred=deferred)
        write(args.output, report)
        print(report['status'])
        return 0 if report['status']=='PASS_MATCHED_ACTOR_INPUTS_PENDING_MEMORY_REVIEW' else 1
    if not args.policy:
        parser.error('--policy is required for a single-run audit')
    report = audit(args.policy.resolve())
    import yaml
    policy = yaml.safe_load(args.policy.read_text())
    path = Path(policy['pilot']['stage_contracts']['64']['output_dir'])/'evidence/postflight.json'
    if path.exists():
        raise FileExistsError('Refusing to overwrite validation audit')
    write(path, report)
    path.with_suffix('.md').write_text(f"# Isolated validation\n\nStatus: `{report['status']}`\n\nFormal training authorized: false\n\n"+'\n'.join(report['errors'])+'\n')
    path.with_suffix('.sha256').write_text(f'{sha(path)}  {path}\n')
    print(report['status'])
    return 0 if report['stage_gate_pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
