"""CPU integration: actual rollout methods + real SamplingParams, mocked Ray/engine/tokenizer."""
import ast
import asyncio
import copy
from contextlib import nullcontext
import json
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace as NS
from uuid import uuid4

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch
from tensordict import TensorDict
from verl import DataProto

from scripts import pressure_runtime as gate
from scripts.prepare_pressure_repair import DEFAULT_DESTINATION, DEFAULT_SOURCE, prepare, patch_runtime


@pytest.fixture(scope='module')
def runtime():
    path = DEFAULT_DESTINATION/'runtime'
    assert path.exists(), 'CPU preparation required: python scripts/prepare_pressure_repair.py'
    return path


def method(path, cls, name, scope):
    tree = ast.parse(path.read_text())
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    fn = copy.deepcopy(next(n for n in owner.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name))
    fn.decorator_list = []
    module = ast.parse('from __future__ import annotations\n')
    module.body.append(fn)
    exec(compile(module, str(path), 'exec'), scope)
    return scope[name]


def sampling_pipeline(runtime, ignore_eos=True, validate=False, prompt_length=16):
    from vllm.sampling_params import SamplingParams
    config_file = DEFAULT_DESTINATION/'config.yaml'
    launcher = (runtime/'scripts/run_vopd_2gpu.sh').read_text()
    # Execute the real YAML -> shell-variable mapping, without launching the shell/training.
    mapping = launcher.split("<<'PY'\n", 1)[1].split('\nPY\n', 1)[0]
    mapped = subprocess.check_output([sys.executable, '-c', mapping, str(config_file), str(runtime)], text=True)
    values = dict(line.split('=', 1) for line in mapped.splitlines())
    assert shlex.split(values['ROLLOUT_IGNORE_EOS']) == ['true']
    assert 'actor_rollout_ref.rollout.ignore_eos="$ROLLOUT_IGNORE_EOS"' in launcher
    config = OmegaConf.create(dict(ignore_eos=ignore_eos, temperature=1., top_p=1.,
        calculate_log_probs=True, response_length=1024, max_model_len=9216,
        enable_rollout_routing_replay=False, agent={'default_agent_loop':'single_turn_agent'},
        val_kwargs={'top_p':.9,'temperature':.7,'response_length':512}))
    calls = []
    class Engine:
        async def generate(self, **kwargs):
            calls.append(kwargs)
            params = kwargs['sampling_params']
            assert isinstance(params, SamplingParams)
            # This is a mock engine, NOT proof of GPU decoding behavior.
            count = params.max_tokens if params.ignore_eos else min(4, params.max_tokens)
            ids = [9]*count
            yield NS(outputs=[NS(token_ids=ids, logprobs=[{9:NS(logprob=-.1)} for _ in ids],
                                 finish_reason='length' if params.ignore_eos else 'stop')])
    server_fn = method(runtime/'verl/workers/rollout/vllm_rollout/vllm_async_server.py', 'vLLMHttpServer', 'generate',
        dict(SamplingParams=SamplingParams, TokensPrompt=lambda **kw:kw, TokenOutput=NS,
             VISION_SPECIAL_TOKENS=[], _qwen2_5_vl_dedup_image_tokens=lambda ids,p:ids))
    server = NS(config=config, model_config=NS(processor=None,lora_rank=0),engine=Engine())
    async def remote(**kwargs):
        return await server_fn(server, **copy.deepcopy(kwargs))  # Ray serialization boundary
    proxy = NS(generate=NS(remote=remote))
    manager_fn = method(runtime/'verl/experimental/agent_loop/agent_loop.py', 'AsyncLLMServerManager', 'generate',
                        dict(uuid4=uuid4))
    manager = NS(_choose_server=lambda _:proxy)
    async def managed(**kwargs):
        return await manager_fn(manager, **kwargs)
    single_fn = method(runtime/'verl/experimental/agent_loop/single_turn_agent_loop.py', 'SingleTurnAgentLoop','run',
                       dict(uuid4=uuid4, simple_timer=lambda *a:nullcontext(), AgentLoopOutput=NS))
    async def vision(_): return {}
    async def template(*a,**kw): return [7]*prompt_length
    single = NS(process_vision_info=vision,apply_chat_template=template,tool_schemas=[],
                server_manager=NS(generate=managed),_get_response_length=lambda:1024)
    async def run_one(params, trajectory, **kwargs):
        return await single_fn(single, params, **kwargs)
    async def trajectories(step, indices, validate): return [{} for _ in indices]
    worker_fn = method(runtime/'verl/experimental/agent_loop/agent_loop.py','AgentLoopWorker','generate_sequences',
        dict(np=np, asyncio=asyncio, get_trajectory_info=trajectories,
             RolloutTraceConfig=NS(get_instance=lambda:NS(max_samples_per_step_per_worker=None))))
    worker = NS(config=NS(actor_rollout_ref=NS(rollout=config)),_run_agent_loop=run_one,_postprocess=lambda out:out)
    prompts = np.empty(2,dtype=object)
    prompts[:] = [[{'role':'user','content':'test'}]]*2
    batch = DataProto(batch=TensorDict({'prompts':torch.ones(2,1,dtype=torch.long)},batch_size=[2]),
                      non_tensor_batch={'raw_prompt':prompts},meta_info={'validate':validate,'global_steps':1})
    return asyncio.run(worker_fn(worker,batch)),calls


@pytest.mark.parametrize('ignore,validate,expected', [(True,False,1024),(False,False,4),(True,True,512),(False,True,4)])
def test_yaml_through_actual_agent_single_manager_server_to_samplingparams(runtime,ignore,validate,expected):
    outputs,calls = sampling_pipeline(runtime,ignore,validate)
    assert len(calls)==2
    assert all(c['sampling_params'].ignore_eos is ignore for c in calls)
    assert all(len(o.response_ids)==expected and sum(o.response_mask)==expected for o in outputs)
    assert calls[0]['sampling_params'].temperature == (.7 if validate else 1.)


def test_context_cap_is_retained(runtime):
    outputs,calls = sampling_pipeline(runtime,True,prompt_length=9166)
    assert all(c['sampling_params'].max_tokens==50 for c in calls)
    assert len(outputs[0].response_ids)==50  # The early gate must reject this short workload.


def test_old_runtime_reproduces_missing_ignore_eos():
    outputs,calls = sampling_pipeline(DEFAULT_SOURCE/'runtime',True)
    assert all(c['sampling_params'].ignore_eos is False for c in calls)
    assert len(outputs[0].response_ids)==4


@pytest.mark.parametrize('configured,requested,expected',[(True,None,True),(False,None,False),(True,False,False),(False,True,True)])
def test_server_fallback_preserves_explicit_request(runtime,configured,requested,expected):
    from vllm.sampling_params import SamplingParams
    class Seen(Exception): pass
    class Engine:
        def generate(self,**kwargs):
            assert kwargs['sampling_params'].ignore_eos is expected
            raise Seen
    fn=method(runtime/'verl/workers/rollout/vllm_rollout/vllm_async_server.py','vLLMHttpServer','generate',
        dict(SamplingParams=SamplingParams,TokensPrompt=lambda **kw:kw,
             VISION_SPECIAL_TOKENS=[],_qwen2_5_vl_dedup_image_tokens=lambda ids,p:ids))
    server=NS(config=OmegaConf.create(dict(ignore_eos=configured,response_length=1024,max_model_len=9216)),
              model_config=NS(processor=None,lora_rank=0),engine=Engine())
    params={} if requested is None else {'ignore_eos':requested}
    with pytest.raises(Seen): asyncio.run(fn(server,[1],params,'test'))


def response_batch(lengths,width=1024):
    mask=torch.zeros(len(lengths),width,dtype=torch.long)
    for i,n in enumerate(lengths): mask[i,:n]=1
    return DataProto(batch=TensorDict(dict(responses=torch.zeros_like(mask),response_mask=mask),batch_size=[len(lengths)]))


@pytest.fixture
def gate_env(tmp_path,monkeypatch):
    cfg=dict(stage='pressure',mode='observe',output_dir=str(tmp_path),first_batch_length_gate=gate.CONTRACT)
    monkeypatch.setattr(gate,'active',lambda:cfg)
    return tmp_path


@pytest.mark.parametrize('lengths',[[1024]*8,[1000]*8,[1024]*7+[999],[255]*8,[0]*8,[1024]*7])
def test_first_batch_gate_all_rows_actual_tokens(gate_env,lengths):
    okay=len(lengths)==8 and min(lengths)>=1000
    if okay: gate.first_batch_check(NS(global_steps=1),response_batch(lengths))
    else:
        with pytest.raises(RuntimeError,match='PRESSURE_FIRST_BATCH_LENGTH_GATE_FAILED'):
            gate.first_batch_check(NS(global_steps=1),response_batch(lengths))
    receipt=json.loads((gate_env/'evidence/first_batch_length_gate.json').read_text())
    assert gate.validate_receipt(receipt) is okay


@pytest.mark.parametrize('bad', ['nonbinary','missing','shape'])
def test_first_batch_gate_rejects_invalid_masks(gate_env,bad):
    batch=response_batch([1024]*8)
    if bad=='nonbinary': batch.batch['response_mask'][0,0]=2
    if bad=='missing': del batch.batch['response_mask']
    if bad=='shape': batch.batch['response_mask']=torch.ones(8,32)
    with pytest.raises(RuntimeError): gate.first_batch_check(NS(global_steps=1),batch)
    assert json.loads((gate_env/'evidence/first_batch_length_gate.json').read_text())['status']=='FAIL_FIRST_BATCH_LENGTH'


def test_first_batch_does_not_modify_payload_or_overwrite(gate_env):
    batch=response_batch([1024]*8);original=batch.batch.clone()
    gate.first_batch_check(NS(global_steps=1),batch)
    assert torch.equal(batch.batch['response_mask'],original['response_mask'])
    with pytest.raises(FileExistsError): gate.first_batch_check(NS(global_steps=1),batch)
    gate.first_batch_check(NS(global_steps=2),response_batch([4]*8))


def test_actual_trainer_hook_stops_before_balance_or_actor(runtime,gate_env,monkeypatch):
    monkeypatch.setitem(sys.modules,'verl.utils.pressure_runtime',gate)
    tree=ast.parse((runtime/'verl/trainer/ppo/ray_trainer.py').read_text())
    fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='fit')
    nodes=sorted(ast.walk(fn),key=lambda n:getattr(n,'lineno',0))
    call=next(n for n in nodes if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
              and isinstance(n.value.func,ast.Name) and n.value.func.id=='first_batch_check')
    following=[n for n in nodes if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
               and n.func.attr in ('_balance_batch','_compute_old_log_prob','_compute_ref_log_prob','_update_actor')
               and n.lineno>call.lineno]
    assert {n.func.attr for n in following}=={'_balance_batch','_compute_old_log_prob','_compute_ref_log_prob','_update_actor'}
    # Execute the actual injected call with a sentinel downstream update.
    scope=dict(first_batch_check=gate.first_batch_check,self=NS(global_steps=1),batch=response_batch([255]*8))
    module=ast.Module(body=[call,ast.parse('downstream_called=True').body[0]],type_ignores=[])
    with pytest.raises(RuntimeError): exec(compile(module,'actual_first_batch_hook','exec'),scope)
    assert 'downstream_called' not in scope


def test_failure_audited_and_new_preflight_passes(runtime,tmp_path):
    guard=(runtime/'scripts/run_vopd_6241_pilot_guarded.py').read_text()
    assert 'or policy.get("validation_manifest"):' in guard
    r=json.loads((DEFAULT_DESTINATION/'run/preflight/pilot_guard_preflight.json').read_text())
    assert r['status']=='PASS'
    with pytest.raises(FileExistsError): prepare(DEFAULT_SOURCE,DEFAULT_DESTINATION)
    # Exercise repeat-patch rejection only on a disposable copy, never a frozen runtime.
    import shutil
    copied=tmp_path/'runtime'
    shutil.copytree(runtime,copied,ignore=shutil.ignore_patterns('__pycache__'))
    target=copied/'verl/experimental/agent_loop/agent_loop.py'
    before=target.read_bytes()
    with pytest.raises(ValueError,match='anchor changed'): patch_runtime(copied)
    assert target.read_bytes()==before


@pytest.mark.parametrize('problem',['missing','short','changed_contract'])
def test_postflight_cannot_pass_without_first_batch_evidence(tmp_path,monkeypatch,problem):
    import yaml
    from scripts import audit_memory_validation as audit_module
    from scripts.fixed_workload_io import sha
    policy=tmp_path/'policy.yaml';config=tmp_path/'config.yaml';manifest=tmp_path/'manifest.json'
    config.write_text('{}');manifest.write_text('{}')
    policy.write_text(yaml.safe_dump(dict(pilot={'stage_contracts':{'64':{'config':str(config)}}},validation_overrides=[])))
    out=tmp_path/'run';(out/'preflight').mkdir(parents=True);(out/'evidence').mkdir()
    (out/'preflight/validation_launch.json').write_text(json.dumps(dict(manifest_sha256=sha(manifest))))
    (out/'preflight/pilot_live_launch_gate.json').write_text(json.dumps(dict(status='PASS',policy_sha256=sha(policy),config_sha256=sha(config))))
    (out/'preflight/run_invocation.json').write_text(json.dumps(dict(hydra_overrides=[])))
    monkeypatch.setattr(audit_module,'check',lambda _:dict(output_dir=str(out),mode='observe',first_batch_length_gate=gate.CONTRACT))
    def forbidden(*args): raise AssertionError('Generic audit must not run after a failed early gate')
    monkeypatch.setattr(audit_module,'pilot_audit',forbidden)
    if problem!='missing':
        receipt=dict(status='PASS_FIRST_BATCH_LENGTH',step=1,contract=dict(gate.CONTRACT),response_width=1024,response_lengths=[1024]*8)
        if problem=='short': receipt['response_lengths'][0]=255
        else: receipt['contract']['minimum_tokens']=1
        (out/'evidence/first_batch_length_gate.json').write_text(json.dumps(receipt))
    result=audit_module.audit(policy)
    assert result['status']=='FAIL_VALIDATION' and not result['stage_gate_pass'] and result['errors']
