#!/usr/bin/env python3
"""Record CPU candidate preparation, never authorize or launch training."""
import argparse
import copy
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.day11_validation_evidence import Reader

CANDIDATE = ROOT / 'configs/vopd_6241_candidate.yaml'
OUTPUT = ROOT / 'artifacts/runs/E-D11-6K-GATE-001/formal_candidate_v1/cpu_preparation.json'
SOURCES = [
    'scripts/run_vopd_2gpu.sh', 'scripts/vopd_training_preflight.py',
    'scripts/audit_formal_candidate.py', 'scripts/day11_validation_evidence.py',
    'verl/experimental/agent_loop/agent_loop.py',
    'verl/experimental/agent_loop/single_turn_agent_loop.py',
    'verl/workers/rollout/vllm_rollout/vllm_async_server.py',
    'verl/workers/actor/dp_actor.py', 'verl/workers/fsdp_workers.py',
    'verl/workers/config/actor.py', 'verl/utils/actor_memory.py',
    'verl/trainer/ppo/ray_trainer.py', 'verl/utils/checkpoint/fsdp_checkpoint_manager.py',
    'verl/utils/checkpoint/shard_io.py', 'scripts/checkpoint_io_contract.py',
]


def candidate_checks(candidate, base):
    normalized = copy.deepcopy(candidate)
    normalized.pop('candidate', None)
    normalized['status'] = base['status']
    normalized['paper_alignment']['pending_gates'] = base['paper_alignment']['pending_gates']
    normalized['actor'].pop('defer_optimizer_state_load', None)
    normalized['actor'].pop('memory_profile_dir', None)
    normalized['resources']['memory_profile'] = base['resources']['memory_profile']
    return {
        'only_declared_candidate_changes': normalized == base,
        'candidate_remains_blocked': candidate['status'] == 'blocked_pending_formal_candidate_validation',
        'deferred_enabled_with_optimizer_offload': candidate['actor'].get('defer_optimizer_state_load') is True
            and candidate['actor']['optimizer_offload'] is True,
        'natural_eos_preserved': candidate['rollout']['ignore_eos'] is False,
        'online_not_replay_or_pressure': candidate['experiment']['prefix_source'] == 'online'
            and 'diagnostic_generation' not in candidate and 'pilot' not in candidate,
        'memory_trace_separate_from_diagnostics': candidate['actor'].get('memory_profile_dir') ==
            candidate['paths']['output_dir'] + '/evidence/memory_stages',
        'resource_floor_contract': (
            candidate['candidate']['resource_floor_refreeze_required'] is True
            or (candidate['candidate']['resource_floor_refreeze_required'] is False
                and candidate['resources']['prelaunch_cgroup_minimum_bytes'] == 240*1024**3
                and bool(candidate['candidate'].get('cpu_freeze')))
        ),
        'natural_validation_explicitly_pending': candidate['candidate']['natural_generation_validation_required'] is True,
        'no_training_authorization': candidate['candidate']['formal_training_authorized'] is False,
    }


def audit(candidate_path=CANDIDATE):
    reader = Reader()
    result = dict(status='FAIL_CPU_CANDIDATE_PREPARATION', gpu_used=False, formal_training_authorized=False,
                  gpu_validation_completed=False, full_vllm_integration_executed=False, errors=[])
    try:
        reader.digest(candidate_path)
        candidate = yaml.safe_load(candidate_path.read_text())
        base_path = ROOT / candidate['candidate']['base_config']
        promoted_backup = (ROOT / 'artifacts/runs/E-D11-6K-GATE-001/formal_promotion_v1'
                           / 'previous_vopd_6241.yaml')
        if promoted_backup.is_file():
            base_path = promoted_backup
        reader.digest(base_path)
        base = yaml.safe_load(base_path.read_text())
        checks = candidate_checks(candidate, base)
        if candidate['candidate']['resource_floor_refreeze_required'] is False:
            from scripts.freeze_formal_cpu import verify
            checks['cpu_floor_freeze_bound'] = verify(ROOT/candidate['candidate']['cpu_freeze'])
            if not checks['cpu_floor_freeze_bound'] and promoted_backup.is_file():
                from scripts.promote_vopd_6241_candidate import verify_receipt
                checks['cpu_floor_freeze_bound'] = verify_receipt() and (
                    candidate['resources']['prelaunch_cgroup_minimum_bytes'] == 240*1024**3
                )
        manifest_path = ROOT / candidate['candidate']['diagnostic_evidence']
        manifest = reader.json(manifest_path)
        for path, expected in manifest['inputs'].items():
            reader.bound(path, expected)
        runtime = Path(manifest['runtime'])
        # These implementation files are unchanged from the successful pressure run.
        for name in ('verl/workers/fsdp_workers.py', 'verl/utils/actor_memory.py',
                     'verl/experimental/agent_loop/agent_loop.py',
                     'verl/workers/rollout/vllm_rollout/vllm_async_server.py'):
            checks['validated_source_' + name] = reader.digest(ROOT/name) == reader.digest(runtime/name)
        actor = (runtime/'verl/workers/actor/dp_actor.py').read_text()
        diagnostic_hook = ('                from verl.utils.fixed_workload import microbatch_plan\n'
                           '                microbatch_plan(self, micro_batches)\n')
        checks['actor_diff_is_only_removed_diagnostic_hook'] = (
            actor.count(diagnostic_hook) == 1
            and actor.replace(diagnostic_hook, '') == (ROOT/'verl/workers/actor/dp_actor.py').read_text())
        trainer = (ROOT/'verl/trainer/ppo/ray_trainer.py').read_text()
        checks['no_forced_length_or_fixed_replay_trainer_hooks'] = all(
            s not in trainer for s in ('verl.utils.fixed_workload', 'verl.utils.pressure_runtime', 'first_batch_check('))
        for name in SOURCES:
            reader.digest(ROOT/name)
        reader.digest(ROOT/'configs/vopd_6241_abort_policy.yaml')
        result['checks'] = checks
        result['errors'] = [name for name, passed in checks.items() if not passed]
        if not result['errors']:
            result['status'] = 'PASS_CPU_CANDIDATE_PREPARATION_PENDING_GPU'
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result['errors'].append(str(exc))
    result['sources'] = reader.sources
    result['limits'] = [
        'This is CPU source/configuration preparation, not final-candidate runtime validation.',
        'CPU floor must be pending or bound to a valid freeze; disk, budget and final promotion are separate gates.',
        'Short natural-EOS outputs are allowed. No forced minimum response length is installed.',
    ]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CANDIDATE)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or output.with_suffix('.sha256').exists():
        raise FileExistsError('Preserve the preparation receipt; choose a new --output')
    value = audit(args.config.resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    with output.with_suffix('.sha256').open('x') as stream:
        stream.write(f'{Reader().digest(output)}  {output}\n')
    print(value['status'])
    print(json.dumps(value['errors'], ensure_ascii=False))
    print(output)
    return 0 if not value['errors'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
