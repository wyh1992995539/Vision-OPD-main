"""Lightweight CPU parameter chain tests. Engine/SamplingParams are test doubles."""
import ast
import asyncio
import copy
from contextlib import nullcontext
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace as NS
from uuid import uuid4

import pytest
import yaml

from scripts.audit_formal_candidate import CANDIDATE, ROOT, audit, candidate_checks


def mapped_argv(config):
    launcher = (ROOT/'scripts/run_vopd_2gpu.sh').read_text()
    mapping = launcher.split("<<'PY'\n", 1)[1].split('\nPY\n', 1)[0]
    assignments = subprocess.check_output([sys.executable, '-c', mapping, str(config), str(ROOT)], text=True)
    command = 'python -m verl.trainer.main_ppo' + launcher.split('python -m verl.trainer.main_ppo', 1)[1]
    command = command.split(' 2>&1 | tee', 1)[0]
    # Execute actual shell expansion with a local function replacing Python.
    # No launcher preflight, training module, output directory or GPU is touched.
    script = ('set -eu\n' + assignments + '\nMAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))\n'
              'EXTRA_ARGS=()\npython() { printf "%s\\n" "$@"; }\n' + command)
    args = subprocess.check_output(['bash', '-c', script], text=True).splitlines()
    return dict(arg.lstrip('+').split('=', 1) for arg in args if '=' in arg)


def test_actual_yaml_shell_expansion_to_training_argv():
    args = mapped_argv(CANDIDATE)
    assert args['actor_rollout_ref.actor.defer_optimizer_state_load'] == 'true'
    assert args['actor_rollout_ref.actor.fsdp_config.optimizer_offload'] == 'true'
    assert args['actor_rollout_ref.rollout.ignore_eos'] == 'false'
    assert args['actor_rollout_ref.actor.memory_profile_dir'] == str(ROOT/'artifacts/runs/E-D12-6K-VOPD-001/evidence/memory_stages')
    assert args['trainer.total_training_steps'] == '780'
    assert args['trainer.save_freq'] == '390'
    assert args['data.shuffle'] == 'true'
    assert args['actor_rollout_ref.rollout.response_length'] == '1024'


def test_promoted_formal_config_keeps_deferred_and_normal_eos():
    args = mapped_argv(ROOT/'configs/vopd_6241.yaml')
    assert args['actor_rollout_ref.actor.defer_optimizer_state_load'] == 'true'
    assert args['actor_rollout_ref.actor.memory_profile_dir'] == str(ROOT/'artifacts/runs/E-D12-6K-VOPD-001/evidence/memory_stages')
    assert args['actor_rollout_ref.rollout.ignore_eos'] == 'false'


@pytest.mark.parametrize('bad', ['eos', 'batch', 'lr', 'offload', 'deferred', 'status', 'profile_path', 'pressure'])
def test_candidate_rejects_unapproved_changes(bad):
    candidate = yaml.safe_load(CANDIDATE.read_text())
    base = yaml.safe_load((ROOT/'artifacts/runs/E-D11-6K-GATE-001/formal_promotion_v1'
                           /'previous_vopd_6241.yaml').read_text())
    if bad == 'eos': candidate['rollout']['ignore_eos'] = True
    elif bad == 'batch': candidate['data']['train_batch_size'] = 4
    elif bad == 'lr': candidate['actor']['learning_rate'] = 1e-6
    elif bad == 'offload': candidate['actor']['optimizer_offload'] = False
    elif bad == 'deferred': candidate['actor']['defer_optimizer_state_load'] = False
    elif bad == 'status': candidate['status'] = 'ready_for_formal_training'
    elif bad == 'profile_path': candidate['actor']['memory_profile_dir'] = '/tmp/other'
    else: candidate['diagnostic_generation'] = 'forced_length_not_paper_sampling'
    assert not all(candidate_checks(candidate, base).values())


def method(path, cls, name, scope):
    tree = ast.parse(path.read_text())
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    fn = copy.deepcopy(next(n for n in owner.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name))
    fn.decorator_list = []
    module = ast.parse('from __future__ import annotations\n')
    module.body.append(fn)
    exec(compile(module, str(path), 'exec'), scope)
    return scope[name]


class Config(dict):
    __getattr__ = dict.__getitem__


class Index(list):
    def tolist(self): return list(self)


class Batch:
    meta_info = {'validate': False, 'global_steps': 1}
    non_tensor_batch = {'index': Index([0]), 'agent_name': ['single_turn_agent'],
                        'raw_prompt': [[{'role': 'user', 'content': 'test'}]]}
    def __len__(self): return 1


def server_method():
    return method(ROOT/'verl/workers/rollout/vllm_rollout/vllm_async_server.py', 'vLLMHttpServer', 'generate',
                  dict(SamplingParams=NS, TokensPrompt=lambda **kw: kw, TokenOutput=NS,
                       VISION_SPECIAL_TOKENS=[], _qwen2_5_vl_dedup_image_tokens=lambda ids, p: ids))


def test_candidate_argv_through_actual_worker_single_manager_server():
    args = mapped_argv(CANDIDATE)
    config = Config(ignore_eos=args['actor_rollout_ref.rollout.ignore_eos'] == 'true',
                    temperature=1., top_p=1., calculate_log_probs=True, response_length=1024,
                    max_model_len=9216, enable_rollout_routing_replay=False)
    calls = []
    class Engine:
        async def generate(self, **kwargs):
            calls.append(kwargs)
            # Simulated EOS completion. This tests plumbing, NOT GPU decoding.
            yield NS(outputs=[NS(token_ids=[9]*4, logprobs=[{9: NS(logprob=-.1)}]*4, finish_reason='stop')])
    server = NS(config=config, model_config=NS(processor=None, lora_rank=0), engine=Engine())
    server_fn = server_method()
    async def remote(**kwargs): return await server_fn(server, **copy.deepcopy(kwargs))
    proxy = NS(generate=NS(remote=remote))
    manager_fn = method(ROOT/'verl/experimental/agent_loop/agent_loop.py', 'AsyncLLMServerManager', 'generate', dict(uuid4=uuid4))
    manager = NS(_choose_server=lambda _: proxy)
    async def managed(**kwargs): return await manager_fn(manager, **kwargs)
    single_fn = method(ROOT/'verl/experimental/agent_loop/single_turn_agent_loop.py', 'SingleTurnAgentLoop', 'run',
                       dict(uuid4=uuid4, simple_timer=lambda *a: nullcontext(), AgentLoopOutput=NS))
    async def vision(_): return {}
    async def template(*a, **kw): return [7]*16
    single = NS(process_vision_info=vision, apply_chat_template=template, tool_schemas=[],
                server_manager=NS(generate=managed), _get_response_length=lambda: 1024)
    async def run_one(params, trajectory, **kwargs): return await single_fn(single, params, **kwargs)
    async def trajectories(*a): return [{}]
    worker_fn = method(ROOT/'verl/experimental/agent_loop/agent_loop.py', 'AgentLoopWorker', 'generate_sequences',
                       dict(asyncio=asyncio, get_trajectory_info=trajectories,
                            RolloutTraceConfig=NS(get_instance=lambda: NS(max_samples_per_step_per_worker=None))))
    worker = NS(config=NS(actor_rollout_ref=NS(rollout=config)), _run_agent_loop=run_one, _postprocess=lambda out: out)
    outputs = asyncio.run(worker_fn(worker, Batch()))
    assert calls[0]['sampling_params'].ignore_eos is False
    assert calls[0]['sampling_params'].max_tokens == 1024
    assert outputs[0].response_ids == [9]*4
    assert outputs[0].response_mask == [1]*4  # No forced minimum response length.


@pytest.mark.parametrize('configured,requested,expected', [(False,None,False),(True,False,False),(False,True,True)])
def test_server_default_and_explicit_false_are_preserved(configured, requested, expected):
    class Seen(Exception): pass
    class Engine:
        def generate(self, **kwargs):
            assert kwargs['sampling_params'].ignore_eos is expected
            raise Seen
    server = NS(config=Config(ignore_eos=configured, response_length=1024, max_model_len=9216),
                model_config=NS(processor=None, lora_rank=0), engine=Engine())
    params = {} if requested is None else {'ignore_eos': requested}
    with pytest.raises(Seen): asyncio.run(server_method()(server, [1], params, 'test'))


def test_preparation_binds_sources_but_does_not_authorize_training():
    result = audit()
    assert result['status'] == 'PASS_CPU_CANDIDATE_PREPARATION_PENDING_GPU', result['errors']
    assert result['formal_training_authorized'] is False
    assert result['gpu_validation_completed'] is False


def test_candidate_cannot_be_directly_launched():
    # Stops at guard check, before conda or model/data loading.
    import os
    env = dict(os.environ)
    env.pop('VOPD_GUARD_ACTIVE', None)
    result = subprocess.run(['bash', str(ROOT/'scripts/run_vopd_2gpu.sh'), '--config', str(CANDIDATE), '--run'],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 2
    assert 'Direct --run is blocked' in result.stderr


@pytest.mark.parametrize('release_status', [False, True])
def test_actual_full_data_preflight_keeps_candidate_blocked(tmp_path, release_status):
    from scripts.vopd_training_preflight import validate_config
    config = yaml.safe_load(CANDIDATE.read_text())
    if release_status:
        config['status'] = 'ready_for_formal_training'
    path = tmp_path/'candidate.yaml'
    path.write_text(yaml.safe_dump(config))
    result = validate_config(path, ROOT)
    assert result['checks']['deferred_optimizer_requires_offload']
    assert result['checks']['deferred_candidate_execution_contract']
    assert result['checks']['paper_rollout_sampling_is_frozen']
    assert result['checks']['formal_candidate_promoted'] is False
    failed = {k for k, passed in result['checks'].items() if not passed}
    assert failed == ({'formal_candidate_promoted'} if release_status else
                      {'formal_candidate_promoted', 'config_not_explicitly_blocked'})
