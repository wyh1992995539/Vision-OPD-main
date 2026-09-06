#!/usr/bin/env python3
"""Prepare/check isolated fixed actor-input replay and long-response diagnostics (CPU default)."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.fixed_workload_io import sha, load_batch
from scripts.run_vopd_6241_pilot_guarded import load_pilot_policy, static_preflight

STAGES = ('capture', 'fixed_baseline', 'fixed_deferred', 'pressure')
DEFAULT = ROOT / 'artifacts/runs/E-D11-6K-GATE-001/memory_optimization/fixed_validation_v1'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def exact_replace(path, old, new):
    text = path.read_text()
    if text.count(old) != 1:
        raise ValueError(f'Runtime hook anchor changed: {path}')
    path.write_text(text.replace(old, new))


def prepare(destination):
    from scripts.prepare_vopd_6241_pilot import build_subset
    from scripts.vopd_training_preflight import validate_config
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError('Refusing to overwrite prepared validation')
    source = ROOT / 'artifacts/runs/E-D11-6K-GATE-001/memory_optimization/ab/deferred_v2'
    original = yaml.safe_load((source/'config.yaml').read_text())
    original_policy = yaml.safe_load((source/'policy.yaml').read_text())
    original_selection = json.loads((source/'selection.json').read_text())
    destination.mkdir(parents=True)
    long_selection = build_subset(Path(original_selection['source']['path']),
        Path(original_selection['prompt_length_audit']['path']),
        [Path(p['path']) for p in original_selection['historical_manifests']],
        destination/'data/train_128.parquet', destination/'data/selection_128.json',
        experiment_id='E-D11-VAL-PRESSURE', count=128)
    # The existing tail-aware selection algorithm must retain the original 64-row prefix.
    if [x['sample_id'] for x in long_selection['samples'][:64]] != [x['sample_id'] for x in original_selection['samples']]:
        raise ValueError('128-row selection lost the existing 64-row prefix')
    results = {}
    for stage in STAGES:
        directory = destination/stage
        runtime = directory/'runtime'
        runtime.mkdir(parents=True)
        for folder in ('verl', 'scripts', 'configs', 'chat_templates'):
            shutil.copytree(ROOT/folder, runtime/folder,
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.pytest_cache'))
        shutil.copy2(ROOT/'scripts/fixed_workload_io.py', runtime/'verl/utils/fixed_workload.py')
        exact_replace(runtime/'verl/trainer/ppo/ray_trainer.py',
            '        batch.meta_info["global_steps"] = self.global_steps\n        # update actor',
            '        batch.meta_info["global_steps"] = self.global_steps\n'
            '        from verl.utils.fixed_workload import actor_input\n'
            '        batch = actor_input(self, batch)\n        # update actor')
        exact_replace(runtime/'verl/workers/actor/dp_actor.py',
            '                self.actor_optimizer.zero_grad()\n\n                for micro_batch in micro_batches:',
            '                from verl.utils.fixed_workload import microbatch_plan\n'
            '                microbatch_plan(self, micro_batches)\n'
            '                self.actor_optimizer.zero_grad()\n\n                for micro_batch in micro_batches:')
        exact_replace(runtime/'scripts/run_vopd_6241_pilot_guarded.py',
            '    command.extend(memory_overrides(policy, Path(result["config"]), output_dir))',
            '    command.extend(memory_overrides(policy, Path(result["config"]), output_dir))\n'
            '    command.extend(policy["validation_overrides"])')
        # Scoped diagnostic exception only in the isolated runtime, never the formal validator.
        exact_replace(runtime/'scripts/vopd_training_preflight.py',
            'and rollout["ignore_eos"] is False',
            'and (rollout["ignore_eos"] is False or (experiment["id"] == "E-D11-VAL-PRESSURE"\n'
            '                         and config.get("diagnostic_generation") == "forced_length_not_paper_sampling"))')
        config, policy = copy.deepcopy(original), copy.deepcopy(original_policy)
        steps = 16 if stage == 'pressure' else 8
        experiment_id = 'E-D11-VAL-' + stage.upper().replace('_', '-')
        config['experiment'].update(id=experiment_id, name=stage, group_name='E-D11-FIXED-VALIDATION')
        config['paths'].update(output_dir=str(directory/'run'), selection_manifest=str(directory/'selection.json'),
                               chat_template=str(ROOT/original['paths']['chat_template']))
        config['status'] = 'diagnostic_only_pending_gpu'
        selection = copy.deepcopy(long_selection if stage == 'pressure' else original_selection)
        selection['experiment_id'] = experiment_id
        write(directory/'selection.json', selection)
        if stage == 'pressure':
            config['paths']['train_file'] = str(destination/'data/train_128.parquet')
            config['data']['expected_train_rows'] = 128
            for key in ('source_samples', 'expected_samples', 'padded_samples'):
                config['training'][key] = 128
            config['training']['total_optimizer_steps'] = 16
            config['rollout']['ignore_eos'] = True
            config['diagnostic_generation'] = 'forced_length_not_paper_sampling'
            config['paper_alignment']['profile'] = 'diagnostic_forced_length_not_paper_sampling'
        config['pilot']['purpose'] = 'diagnostic_only_not_formal_or_quality_evaluation'
        (directory/'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
        policy.pop('memory_experiment', None)
        policy['pilot']['static_gate'] = str(ROOT/policy['pilot']['static_gate'])
        contract = policy['pilot']['stage_contracts']['64']
        contract.update(config=str(directory/'config.yaml'), output_dir=str(directory/'run'),
                        experiment_id=experiment_id, expected_rows=steps*8, expected_optimizer_steps=steps,
                        prerequisite_postflight=str(ROOT/contract['prerequisite_postflight']), require_cold_reload=False)
        policy['pilot']['postflight_script'] = str(runtime/'scripts/audit_memory_validation.py')
        deferred = stage in ('fixed_deferred', 'pressure')
        policy['validation_overrides'] = [
            '++actor_rollout_ref.actor.defer_optimizer_state_load='+str(deferred).lower(),
            '++actor_rollout_ref.actor.memory_profile_dir='+str(directory/'run/evidence/memory_stages')]
        policy['validation_manifest'] = str(directory/'manifest.json')
        (directory/'policy.yaml').write_text(yaml.safe_dump(policy, sort_keys=False))
        mode = 'capture' if stage == 'capture' else 'observe' if stage == 'pressure' else 'replay'
        write(runtime/'validation_active.json', dict(stage=stage, mode=mode, output_dir=str(directory/'run'),
                                                     bundle_manifest=str(destination/'fixed_bundle.json')))
        files = sorted(p for p in runtime.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
        files += [directory/'config.yaml', directory/'policy.yaml', directory/'selection.json']
        manifest = dict(schema_version=1, stage=stage, expected_steps=steps, mode=mode,
            output_dir=str(directory/'run'), runtime=str(runtime), source_root=str(ROOT),
            bundle_manifest=str(destination/'fixed_bundle.json'), formal_training_authorized=False,
            min_post_warmup_steps=2, long_response_min_tokens=1000, minimum_long_steps_per_rank=2,
            runtime_hook_scope='actor_update_only; online rollout/ref/logprob stages are NOT fixed',
            pressure_generation_change='ignore_eos=true; diagnostic only' if stage=='pressure' else None,
            inputs={str(p): sha(p) for p in files})
        write(directory/'manifest.json', manifest)
        # Exercise the exact isolated launcher/validator that will later own the run.
        process = subprocess.run([sys.executable, str(runtime/'scripts/run_vopd_6241_pilot_guarded.py'),
                                  '--stage', '64', '--policy', str(directory/'policy.yaml'), '--preflight-only'],
                                 cwd=runtime, capture_output=True, text=True)
        checked_path = directory/'run/preflight/pilot_guard_preflight.json'
        if not checked_path.exists():
            raise RuntimeError(f'Isolated preflight did not produce evidence: {process.stderr[-2000:]}')
        checked = json.loads(checked_path.read_text())
        compile((runtime/'verl/trainer/ppo/ray_trainer.py').read_text(), 'trainer_overlay', 'exec')
        compile((runtime/'verl/workers/actor/dp_actor.py').read_text(), 'actor_overlay', 'exec')
        if process.returncode != 0 or checked['status'] != 'PASS':
            raise ValueError(f'Static validation failed: {stage}: {checked.get("failed_checks")}')
        results[stage] = dict(cpu_static='PASS', gpu_run_started=False,
                              launch_dependency='sealed_capture_bundle' if mode=='replay' else 'live_resource_and_billing_gate')
    write(destination/'cpu_preparation.json', dict(status='PASS_CPU_PREPARATION_PENDING_GPU',
        formal_training_authorized=False, stages=results, source_bound_runtime_isolated=True,
        replay_is_offline_actor_diagnostic_not_on_policy_training=True,
        projected_checkpoint_bytes=4*int(original_policy['disk']['checkpoint_estimate_bytes']),
        reserve_for_each_launch_bytes=int(original_policy['disk']['prelaunch_required_bytes'])))
    return results


def check(directory):
    directory = directory.resolve()
    manifest = json.loads((directory/'manifest.json').read_text())
    if manifest['formal_training_authorized'] is not False:
        raise ValueError('Formal authorization is forbidden')
    for path, expected in manifest['inputs'].items():
        if sha(path) != expected:
            raise ValueError(f'Changed validation source/config: {path}')
    if manifest['mode'] == 'replay':
        bundle = json.loads(Path(manifest['bundle_manifest']).read_text())
        if bundle['status'] != 'SEALED_CAPTURE' or len(bundle['batches']) != manifest['expected_steps']:
            raise ValueError('Missing/incomplete sealed capture')
        if sha(bundle['capture_manifest_path']) != bundle['capture_manifest_sha256']:
            raise ValueError('Capture provenance changed')
        for entry in bundle['batches']:
            load_batch(entry)
    return manifest


def seal(directory):
    directory = directory.resolve()
    manifest = check(directory)
    if manifest['mode'] != 'capture':
        raise ValueError('Only capture can seal a bundle')
    output = Path(manifest['output_dir'])
    receipt = json.loads((output/'evidence/exit_receipt.json').read_text())
    if receipt['guard_exit_code'] != 0 or receipt['postflight_status'] != 'PASS_CAPTURE':
        raise ValueError('Capture did not pass guarded postflight')
    entries = []
    for step in range(1, manifest['expected_steps']+1):
        entry = json.loads((output/f'evidence/fixed_workload/step{step:04d}.json').read_text())
        load_batch(entry)
        entries.append(entry)
    path = Path(manifest['bundle_manifest'])
    if path.exists():
        raise FileExistsError('Bundle already sealed')
    write(path, dict(status='SEALED_CAPTURE', batches=entries, capture_manifest_path=str(directory/'manifest.json'),
                     capture_manifest_sha256=sha(directory/'manifest.json'),
                     formal_training_authorized=False))
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'check', 'seal', 'launch'])
    parser.add_argument('--directory', type=Path, default=DEFAULT)
    parser.add_argument('--current-autodl-cost-cny', type=float)
    parser.add_argument('--billing-observed-at-utc')
    args = parser.parse_args()
    if args.action == 'prepare':
        print(json.dumps(prepare(args.directory), indent=2)); return 0
    if args.action == 'seal':
        print(seal(args.directory)); return 0
    manifest = check(args.directory)
    if args.action == 'check':
        print(json.dumps(dict(status='PASS_STATIC_DEPENDENCIES', gpu_checked=False, stage=manifest['stage']))); return 0
    if args.current_autodl_cost_cny is None or not args.billing_observed_at_utc:
        parser.error('launch requires fresh billing amount and timestamp')
    receipt = Path(manifest['output_dir'])/'preflight/validation_launch.json'
    if receipt.exists():
        raise FileExistsError('Launch already recorded; preserve attempt and prepare a new directory')
    write(receipt, dict(manifest_sha256=sha(args.directory/'manifest.json'),
                       bundle_sha256=sha(manifest['bundle_manifest']) if manifest['mode']=='replay' else None))
    # Original monitor and live guards run in a separately source-bound runtime.
    command = [sys.executable, str(Path(manifest['runtime'])/'scripts/run_vopd_6241_pilot_guarded.py'),
               '--stage', '64', '--policy', str(args.directory.resolve()/'policy.yaml'), '--run',
               '--current-autodl-cost-cny', str(args.current_autodl_cost_cny),
               '--billing-observed-at-utc', args.billing_observed_at_utc]
    return subprocess.call(command, cwd=manifest['runtime'])


if __name__ == '__main__':
    raise SystemExit(main())
