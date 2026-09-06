#!/usr/bin/env python3
"""CPU-only preparation of a new pressure attempt; never edit old bound runtimes."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.memory_validation import check, exact_replace, write
from scripts.fixed_workload_io import sha
from scripts.pressure_runtime import CONTRACT

DEFAULT_SOURCE = ROOT/'artifacts/runs/E-D11-6K-GATE-001/memory_optimization/fixed_validation_v1/pressure'
DEFAULT_DESTINATION = DEFAULT_SOURCE.parent/'pressure_v2'


def patch_runtime(runtime):
    if 'ignore_eos=config.ignore_eos,' in (runtime/'verl/experimental/agent_loop/agent_loop.py').read_text():
        raise ValueError('Runtime hook anchor changed: pressure repair already applied')
    exact_replace(runtime/'verl/experimental/agent_loop/agent_loop.py',
                  '            logprobs=config.calculate_log_probs,\n',
                  '            logprobs=config.calculate_log_probs,\n'
                  '            ignore_eos=config.ignore_eos,\n')
    # Preserve explicit per-request false values, including future validation overrides.
    exact_replace(runtime/'verl/workers/rollout/vllm_rollout/vllm_async_server.py',
                  '        sampling_params.setdefault("repetition_penalty", self.config.get("repetition_penalty", 1.0))',
                  '        sampling_params.setdefault("repetition_penalty", self.config.get("repetition_penalty", 1.0))\n'
                  '        sampling_params.setdefault("ignore_eos", self.config.get("ignore_eos", False))')
    exact_replace(runtime/'verl/trainer/ppo/ray_trainer.py',
                  '                    # Balance the number of valid tokens across DP ranks.',
                  '                    from verl.utils.pressure_runtime import first_batch_check\n'
                  '                    first_batch_check(self, batch)\n'
                  '                    # Balance the number of valid tokens across DP ranks.')
    exact_replace(runtime/'scripts/run_vopd_6241_pilot_guarded.py',
                  '    if exit_code == 0 or policy.get("memory_experiment"):',
                  '    if exit_code == 0 or policy.get("memory_experiment") or policy.get("validation_manifest"):')


def prepare(source, destination):
    source, destination = source.resolve(), destination.resolve()
    old = check(source)
    if old['stage'] != 'pressure' or old['mode'] != 'observe':
        raise ValueError('Only an existing pressure diagnostic can be repaired')
    if destination.exists():
        raise FileExistsError('Preserve existing attempts; choose a new destination')
    runtime = destination/'runtime'
    shutil.copytree(Path(old['runtime']), runtime,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.pytest_cache'))
    for name in ('pressure_runtime.py', 'audit_memory_validation.py'):
        shutil.copy2(ROOT/'scripts'/name, runtime/'scripts'/name)
    shutil.copy2(ROOT/'scripts/pressure_runtime.py', runtime/'verl/utils/pressure_runtime.py')
    patch_runtime(runtime)
    config = yaml.safe_load((source/'config.yaml').read_text())
    policy = yaml.safe_load((source/'policy.yaml').read_text())
    config['paths'].update(output_dir=str(destination/'run'), selection_manifest=str(destination/'selection.json'))
    config['experiment']['name'] = 'pressure_v2_sampling_repair'
    # Retain the diagnostic experiment ID: the scoped paper-contract exception requires it.
    assert config['experiment']['id'] == 'E-D11-VAL-PRESSURE'
    assert config['rollout']['ignore_eos'] is True
    assert config['diagnostic_generation'] == 'forced_length_not_paper_sampling'
    (destination/'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    shutil.copy2(source/'selection.json', destination/'selection.json')
    policy['pilot']['stage_contracts']['64'].update(config=str(destination/'config.yaml'), output_dir=str(destination/'run'))
    policy['pilot']['postflight_script'] = str(runtime/'scripts/audit_memory_validation.py')
    policy['validation_manifest'] = str(destination/'manifest.json')
    policy['validation_overrides'] = [x.replace(str(source/'run'), str(destination/'run')) for x in policy['validation_overrides']]
    (destination/'policy.yaml').write_text(yaml.safe_dump(policy, sort_keys=False))
    cfg = json.loads((runtime/'validation_active.json').read_text())
    cfg.update(output_dir=str(destination/'run'), first_batch_length_gate=CONTRACT)
    write(runtime/'validation_active.json', cfg)
    manifest = copy.deepcopy(old)
    manifest.update(schema_version=2, output_dir=str(destination/'run'), runtime=str(runtime),
                    first_batch_length_gate=CONTRACT, repair_parent_manifest=str(source/'manifest.json'),
                    repair_parent_manifest_sha256=sha(source/'manifest.json'),
                    preparation_script_sha256=sha(__file__))
    files = sorted(p for p in runtime.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    files += [destination/name for name in ('config.yaml','policy.yaml','selection.json')]
    manifest['inputs'] = {str(p):sha(p) for p in files}
    write(destination/'manifest.json', manifest)
    for path in (runtime/'verl').rglob('*.py'):
        compile(path.read_text(), str(path), 'exec')
    result = subprocess.run([sys.executable, str(runtime/'scripts/run_vopd_6241_pilot_guarded.py'),
                             '--stage','64','--policy',str(destination/'policy.yaml'),'--preflight-only'],
                            cwd=runtime, capture_output=True, text=True)
    preflight = destination/'run/preflight/pilot_guard_preflight.json'
    if result.returncode or not preflight.exists() or json.loads(preflight.read_text())['status'] != 'PASS':
        raise RuntimeError('Repaired isolated preflight failed: '+result.stderr[-1500:])
    check(source)
    check(destination)
    write(destination/'cpu_preparation.json', dict(status='PASS_STATIC_PREPARATION_PENDING_CPU_TESTS_AND_GPU',
          gpu_run_started=False, formal_training_authorized=False, first_batch_length_gate=CONTRACT,
          repaired_sampling_chain=['agent_loop.ignore_eos','vllm_samplingparams.ignore_eos'],
          parent_manifest_sha256=manifest['repair_parent_manifest_sha256']))
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--destination', type=Path, default=DEFAULT_DESTINATION)
    args = parser.parse_args()
    print(prepare(args.source, args.destination))
