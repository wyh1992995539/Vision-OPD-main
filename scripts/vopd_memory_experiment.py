#!/usr/bin/env python3
"""Prepare isolated, guarded baseline/deferred 64-row memory experiments. CPU only."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BASE = ROOT / 'artifacts/runs/E-D11-6K-GATE-001/memory_optimization'
SOURCE_PATHS = (
    'verl/utils/actor_memory.py', 'verl/workers/actor/dp_actor.py',
    'verl/workers/fsdp_workers.py', 'verl/workers/config/actor.py',
    'verl/utils/fsdp_utils.py', 'scripts/run_vopd_2gpu.sh',
    'scripts/run_vopd_6241_pilot_guarded.py', 'scripts/vopd_memory_experiment.py',
    'configs/vopd_6241_pilot_64.yaml', 'configs/vopd_6241_pilot_abort_policy.yaml',
    'artifacts/runs/E-D11-6K-GATE-001/pilot/64/preflight/selection.json',
    'scripts/audit_vopd_memory_ab.py', 'scripts/compare_vopd_memory_ab.py',
    'scripts/audit_vopd_6241_pilot.py', 'scripts/monitor_vopd_training.py',
)
AB_POSTFLIGHT = 'scripts/audit_vopd_memory_ab.py'
# Only offline re-audits may use archived versions of these audit-only files.
# Actor, collector, launcher, policies and training sources must still match live hashes.
OFFLINE_AUDIT_SOURCES = frozenset((AB_POSTFLIGHT, 'scripts/vopd_memory_experiment.py',
                                   'scripts/compare_vopd_memory_ab.py'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_overrides(variant, output):
    if variant not in ('baseline', 'deferred'):
        raise ValueError('Unknown memory variant')
    return [
        '++actor_rollout_ref.actor.defer_optimizer_state_load=' + str(variant == 'deferred').lower(),
        '++actor_rollout_ref.actor.memory_profile_dir=' + str(output / 'evidence/memory_stages'),
    ]


def fork_selection_manifest(source_path, experiment_id):
    selection = json.loads(source_path.read_text())
    selection['experiment_id'] = experiment_id
    selection['memory_ab_parent_manifest'] = {
        'path': str(source_path), 'sha256': sha(source_path),
        'sample_order_and_data_unchanged': True,
    }
    return selection


def memory_overrides(policy, config_path, output, *, audit_source_snapshot=None):
    extension = policy.get('memory_experiment')
    if extension is None:
        return []
    manifest = json.loads(Path(extension['manifest']).read_text())
    variant = extension['variant']
    if manifest['variant'] != variant or manifest['formal_training_authorized'] is not False:
        raise ValueError('Invalid memory experiment manifest')
    if (policy.get('pilot', {}).get('postflight_script') != AB_POSTFLIGHT
            or policy['pilot']['stage_contracts']['64'].get('require_cold_reload') is not False):
        raise ValueError('Memory runs require their dedicated postflight and no cold-reload gate')
    if manifest.get('effective_policy') != policy:
        raise ValueError('Memory effective policy changed')
    if manifest['config_sha256'] != sha(config_path):
        raise ValueError('Memory config hash changed')
    for path, digest in manifest['source_hashes'].items():
        if sha(ROOT / path) != digest:
            if (audit_source_snapshot is None or path not in OFFLINE_AUDIT_SOURCES
                    or sha(Path(audit_source_snapshot) / path) != digest):
                raise ValueError(f'Memory source hash changed: {path}')
    if set(manifest['source_hashes']) != set(SOURCE_PATHS):
        raise ValueError('Incomplete memory source binding')
    expected = expected_overrides(variant, output)
    if manifest['overrides'] != expected:
        raise ValueError('Unexpected memory overrides or output path')
    config = yaml.safe_load(config_path.read_text())
    if config['experiment']['id'] != 'E-D11-MEM-' + variant.upper():
        raise ValueError('Memory overrides may only target isolated experiments')
    return expected


def prepare_deferred_revision(policy_path, directory, audit_source_snapshot):
    """Fork an unlaunched binding. Never rewrite a historical manifest or run directory."""
    from scripts.run_vopd_6241_pilot_guarded import static_preflight, load_pilot_policy, resolve
    policy_path, directory = resolve(policy_path), resolve(directory)
    policy, contract = load_pilot_policy(policy_path, '64')
    config_path, old_output = resolve(contract['config']), resolve(contract['output_dir'])
    if policy.get('memory_experiment', {}).get('variant') != 'deferred':
        raise ValueError('Only unlaunched deferred bindings may be forked')
    if old_output.exists():
        raise FileExistsError(f'Refusing to rebind an existing run: {old_output}')
    if directory.exists():
        raise FileExistsError(f'Refusing to overwrite experiment: {directory}')
    # Every non-audit source, effective policy, config and override remains bound.
    memory_overrides(policy, config_path, old_output, audit_source_snapshot=audit_source_snapshot)
    old_manifest_path = Path(policy['memory_experiment']['manifest'])
    manifest = json.loads(old_manifest_path.read_text())
    config = yaml.safe_load(config_path.read_text())
    selection_path = resolve(config['paths']['selection_manifest'])
    selection = selection_path.read_bytes()
    prior = {p: p.read_bytes() for p in (policy_path, config_path, old_manifest_path, selection_path)}
    directory.mkdir(parents=True)
    archive = directory / 'previous_binding'
    archive.mkdir()
    for p, content in prior.items():
        (archive / p.name).write_bytes(content)
    (directory / 'selection.json').write_bytes(selection)
    config['paths']['selection_manifest'] = str(directory / 'selection.json')
    config['paths']['output_dir'] = str(directory / 'run')
    new_config_path, new_policy_path = directory / 'config.yaml', directory / 'policy.yaml'
    new_config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    policy['pilot']['stage_contracts']['64'].update(config=str(new_config_path), output_dir=config['paths']['output_dir'])
    policy['memory_experiment']['manifest'] = str(directory / 'manifest.json')
    new_policy_path.write_text(yaml.safe_dump(policy, sort_keys=False))
    manifest.update(config_sha256=sha(new_config_path), effective_policy=load_pilot_policy(new_policy_path, '64')[0],
                    source_hashes={p: sha(ROOT / p) for p in SOURCE_PATHS},
                    overrides=expected_overrides('deferred', directory / 'run'),
                    previous_binding=dict(policy_path=str(policy_path), policy_sha256=sha(policy_path),
                                          manifest_path=str(old_manifest_path), manifest_sha256=sha(old_manifest_path),
                                          selection_sha256=sha(selection_path), original_run_absent=True))
    (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    report = static_preflight(new_policy_path, '64')
    (directory / 'static_preflight.json').write_text(json.dumps(report, indent=2) + '\n')
    if any(p.read_bytes() != content for p, content in prior.items()):
        raise RuntimeError('Original binding changed during preparation')
    if report['status'] != 'PASS':
        raise RuntimeError(f"Deferred static preflight failed: {report['failed_checks']}")
    return dict(status='PASS', policy=str(new_policy_path), original_binding_preserved=True,
                training_started=False, gpu_validation_pending=True)


def prepare():
    from scripts.run_vopd_6241_pilot_guarded import static_preflight, load_pilot_policy
    source = yaml.safe_load((ROOT / 'configs/vopd_6241_pilot_64.yaml').read_text())
    base_policy = yaml.safe_load((ROOT / 'configs/vopd_6241_pilot_abort_policy.yaml').read_text())
    reports = {}
    for variant in ('baseline', 'deferred'):
        directory = BASE / 'ab' / variant
        if directory.exists():
            raise FileExistsError(f'Refusing to overwrite experiment: {directory}')
    for variant in ('baseline', 'deferred'):
        directory = BASE / 'ab' / variant
        if directory.exists():
            raise FileExistsError(f'Refusing to overwrite experiment: {directory}')
        directory.mkdir(parents=True)
        config = copy.deepcopy(source)
        config['experiment']['id'] = 'E-D11-MEM-' + variant.upper()
        config['experiment']['name'] = 'memory-ab-' + variant
        config['experiment']['group_name'] = 'E-D11-MEM-AB'
        config['paths']['output_dir'] = str(directory / 'run')
        config['pilot']['purpose'] = 'diagnostic_memory_ab_not_formal_safety_signoff'
        selection_path = directory / 'selection.json'
        selection = fork_selection_manifest(
            ROOT / source['paths']['selection_manifest'], config['experiment']['id'])
        selection_path.write_text(json.dumps(selection, indent=2) + '\n')
        config['paths']['selection_manifest'] = str(selection_path)
        config_path = directory / 'config.yaml'
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        manifest_path = directory / 'manifest.json'
        policy = copy.deepcopy(base_policy)
        stage = policy['pilot']['stage_contracts']['64']
        stage['experiment_id'] = config['experiment']['id']
        stage['config'] = str(config_path)
        stage['output_dir'] = config['paths']['output_dir']
        stage['require_cold_reload'] = False
        policy['pilot']['postflight_script'] = AB_POSTFLIGHT
        policy['memory_experiment'] = {'variant': variant, 'manifest': str(manifest_path)}
        (directory / 'policy.yaml').write_text(yaml.safe_dump(policy, sort_keys=False))
        manifest = dict(variant=variant, config_sha256=sha(config_path),
                        effective_policy=load_pilot_policy(directory / 'policy.yaml', '64')[0],
                        source_hashes={path: sha(ROOT / path) for path in SOURCE_PATHS},
                        formal_training_authorized=False,
                        diagnostic_sync_overhead=True,
                        cpu_tests_do_not_validate_cuda_transfers=True,
                        algorithm_config_source_sha256=sha(ROOT / 'configs/vopd_6241_pilot_64.yaml'),
                        overrides=expected_overrides(variant, Path(config['paths']['output_dir'])))
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
        report = static_preflight(directory / 'policy.yaml', '64')
        (directory / 'static_preflight.json').write_text(json.dumps(report, indent=2) + '\n')
        reports[variant] = {'status': report['status'], 'failed_checks': report['failed_checks']}
    print(json.dumps(reports, indent=2))
    if any(v['status'] != 'PASS' for v in reports.values()):
        raise RuntimeError('Memory A/B static preflight failed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-deferred-from', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--audit-source-snapshot', type=Path)
    args = parser.parse_args()
    if args.prepare_deferred_from:
        if not args.output_dir or not args.audit_source_snapshot:
            parser.error('Deferred revision requires --output-dir and --audit-source-snapshot')
        print(json.dumps(prepare_deferred_revision(args.prepare_deferred_from, args.output_dir,
                                                   args.audit_source_snapshot), indent=2))
    else:
        if args.output_dir or args.audit_source_snapshot:
            parser.error('Revision options require --prepare-deferred-from')
        prepare()
