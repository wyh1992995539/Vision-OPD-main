#!/usr/bin/env python3
"""Pilot Student cold inference smoke, NOT training resume or accuracy evaluation.

Run from repository root: python -m scripts.vopd_6241_pilot_reload --run
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request

import pyarrow.parquet as pq
import yaml

from scripts.vopd_day8_reload import (
    directory_manifest, file_snapshot, memory_limit_bytes, port_is_open,
    run_command, sha256_file, stable_key, stop_process_group, utc_now,
    validate_merged_model, verify_predictions, wait_for_server, write_json_atomic,
)

ROOT = Path(__file__).resolve().parents[1]
GIB = 1024**3


def resolve(value):
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def select_samples(rows, count, seed):
    by_id = {}
    for row in rows:
        provenance = row['extra_info']['provenance']
        if provenance.get('split') != 'train':
            raise ValueError('Cold smoke accepts training samples only; no benchmark/eval inputs')
        sid = provenance['sample_id']
        if not sid or sid in by_id:
            raise ValueError('Empty or duplicate sample ID')
        by_id[sid] = row
    if not 0 < count <= len(by_id):
        raise ValueError('Invalid smoke sample count')
    return [(sid, by_id[sid]) for sid in sorted(by_id, key=lambda sid: (stable_key(seed, sid), sid))[:count]]


def student_messages(row):
    # Only student image and user prompt: never bbox_images or ground truth.
    messages, images = row['prompt'], row['images']
    if len(messages) != 1 or messages[0]['role'] != 'user' or len(images) != 1:
        raise ValueError('Expected one user prompt and one student image')
    prompt = messages[0]['content']
    if prompt.count('<image>') != 1:
        raise ValueError('Expected exactly one image placeholder')
    path = Path(images[0]['path'])
    mime = mimetypes.guess_type(str(path))[0] or 'image/png'
    uri = 'data:' + mime + ';base64,' + base64.b64encode(path.read_bytes()).decode('ascii')
    before, after = prompt.split('<image>')
    content = []
    if before:
        content.append({'type': 'text', 'text': before})
    content.append({'type': 'image_url', 'image_url': {'url': uri}})
    if after:
        content.append({'type': 'text', 'text': after})
    return [{'role': 'user', 'content': content}]


def preflight(config):
    source, output = resolve(config['source_dir']), resolve(config['output_dir'])
    if output == source or source in output.parents:
        raise ValueError('Output must be outside original checkpoint')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Refusing to overwrite an existing cold-reload attempt')
    actor, world = source / 'actor', config['world_size']
    for rank in range(world):
        for kind in ('model', 'optim', 'extra_state'):
            path = actor / f'{kind}_world_size_{world}_rank_{rank}.pt'
            if not path.is_file() or not path.stat().st_size:
                raise ValueError(f'Missing checkpoint shard: {path}')
    for path in (source / 'data.pt', actor / 'fsdp_config.json', actor / 'huggingface/config.json'):
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f'Missing checkpoint metadata: {path}')
    if json.loads((actor / 'fsdp_config.json').read_text())['world_size'] != world:
        raise ValueError('Checkpoint world size mismatch')
    marker = source.parent / 'latest_checkpointed_iteration.txt'
    if int(marker.read_text().strip()) != config['global_step'] or source.name != f"global_step_{config['global_step']}":
        raise ValueError('Checkpoint step mismatch')
    data = resolve(config['input_parquet'])
    if sha256_file(data) != config['input_parquet_sha256']:
        raise ValueError('Training parquet hash mismatch')
    rows = pq.read_table(data, columns=['extra_info', 'prompt', 'images']).to_pylist()
    if len(rows) != config['expected_samples']:
        raise ValueError('Training row count mismatch')
    samples = select_samples(rows, config['reload_samples'], config['seed'])
    for _, row in samples:
        student_messages(row)
    if memory_limit_bytes() < config['runtime']['minimum_cpu_memory_gib'] * GIB:
        raise ValueError('Insufficient cgroup/physical memory')
    if shutil.disk_usage(source).free < config['runtime']['minimum_free_disk_gib'] * GIB:
        raise ValueError('Insufficient disk for merged model')
    if port_is_open('127.0.0.1', config['serving']['port']):
        raise ValueError('Serving port already occupied')
    files = sorted(path for path in source.rglob('*') if path.is_file()) + [marker]
    return source, output, samples, files


def runtime_env(**overrides):
    # Some container launchers export empty/zero OMP_NUM_THREADS; vLLM rejects it.
    return dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', **overrides)


def execute(config_path, run, output_override=None, reuse_merged_from=None):
    config = yaml.safe_load(config_path.read_text())
    if output_override:
        config['output_dir'] = str(output_override)
    source, output, samples, files = preflight(config)
    print('PREFLIGHT_PASS sample_ids=' + json.dumps([sid for sid, _ in samples]), flush=True)
    if not run:
        return
    import torch
    if torch.cuda.device_count() != config['world_size']:
        raise ValueError('Expected exactly two visible GPUs')
    output.mkdir(parents=True, exist_ok=False)
    summary = {
        'schema_version': 1, 'experiment_id': config['experiment_id'],
        'phase': 'checkpoint_cold_reload', 'scope': 'student_model_training_samples_functional_inference_only',
        'training_resume_validated': False, 'model_quality_evaluated': False,
        'source_checkpoint': str(source), 'started_at_utc': utc_now(),
        'config_sha256': sha256_file(config_path), 'input_parquet_sha256': config['input_parquet_sha256'],
        'sample_ids': [sid for sid, _ in samples], 'status': 'RUNNING',
    }
    write_json_atomic(output / 'config_snapshot.json', config)
    write_json_atomic(output / 'reload_validation_summary.json', summary)
    server, before = None, None
    try:
        print('HASH_SOURCE_BEFORE (~53 GiB)', flush=True)
        before = file_snapshot(files, include_sha256=True)
        write_json_atomic(output / 'checkpoint_manifest_before.json', {'files': before})
        merged = output / 'merged_hf'
        if reuse_merged_from:
            prior = resolve(reuse_merged_from)
            receipt = json.loads((prior / 'reload_validation_summary.json').read_text())
            manifest = json.loads((prior / 'merged_manifest.json').read_text())
            prior_before = json.loads((prior / 'checkpoint_manifest_before.json').read_text())['files']
            if (receipt.get('source_checkpoint') != str(source)
                    or receipt.get('source_checkpoint_unchanged') is not True
                    or prior_before != before
                    or receipt.get('merged_manifest_sha256') != sha256_file(prior / 'merged_manifest.json')):
                raise ValueError('Cannot bind reusable merged model to unchanged source checkpoint')
            merged = Path(receipt['merged_model'])
            if directory_manifest(merged) != manifest['files']:
                raise ValueError('Reusable merged model contents changed')
            summary['reused_merge_receipt'] = str(prior / 'reload_validation_summary.json')
            print('VERIFIED_MERGE_REUSE', flush=True)
        else:
            print('MERGE_START', flush=True)
            run_command([sys.executable, '-m', 'verl.model_merger', 'merge', '--backend', 'fsdp',
                         '--local_dir', str(source / 'actor'), '--target_dir', str(merged)],
                        output / 'merge.log', env=runtime_env(CUDA_VISIBLE_DEVICES=''))
        errors = validate_merged_model(merged)
        if errors:
            raise ValueError('; '.join(errors))
        write_json_atomic(output / 'merged_manifest.json', {'files': directory_manifest(merged)})
        summary['merged_model'] = str(merged)
        summary['merged_manifest_sha256'] = sha256_file(output / 'merged_manifest.json')
        serving = config['serving']
        if not serving['enforce_eager']:
            raise ValueError('This inference-only profile requires eager mode')
        command = [shutil.which('vllm') or 'vllm', 'serve', str(merged),
                   '--served-model-name', serving['model_name'], '--host', '127.0.0.1',
                   '--port', str(serving['port']), '--trust-remote-code', '--dtype', 'bfloat16',
                   '--max-model-len', str(serving['max_model_len']), '--max-num-seqs', '1',
                   '--tensor-parallel-size', str(serving['tensor_parallel_size']),
                   '--distributed-executor-backend', 'mp', '--gpu-memory-utilization', str(serving['gpu_memory_utilization']),
                   '--enforce-eager', '--limit-mm-per-prompt', '{"image":1,"video":0}',
                   '--chat-template', str(resolve(config['chat_template'])),
                   '--default-chat-template-kwargs', '{"enable_thinking":false}',
                   '--kernel-config', '{"enable_flashinfer_autotune":false}',
                   '--compilation-config', '{"pass_config":{"fuse_allreduce_rms":false}}']
        write_json_atomic(output / 'server_command.json', {'command': command, 'cuda_visible_devices': '0,1', 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'})
        print('SERVER_START TP=2 eager max_model_len=9216', flush=True)
        with (output / 'vllm_server.log').open('w') as stream:
            server = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                      env=runtime_env(CUDA_VISIBLE_DEVICES='0,1'), start_new_session=True)
        base = f"http://127.0.0.1:{serving['port']}/v1"
        ready = wait_for_server(server, base + '/models', serving['startup_timeout_seconds'], output / 'vllm_server.log')
        if serving['model_name'] not in [model['id'] for model in ready['data']]:
            raise ValueError('Unexpected served model ID')
        records = []
        for index, (sid, row) in enumerate(samples, 1):
            payload = {'model': serving['model_name'], 'messages': student_messages(row),
                       'temperature': 0.0, 'top_p': 1.0, 'max_tokens': serving['max_new_tokens'],
                       'seed': config['seed'], 'return_token_ids': True,
                       'chat_template_kwargs': {'enable_thinking': False}}
            request = urllib.request.Request(base + '/chat/completions', data=json.dumps(payload).encode(),
                                             headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(request, timeout=300) as response:
                value = json.load(response)
            choice = value['choices'][0]
            record = {'sample_id': sid, 'raw_prediction': choice['message']['content'],
                      'response_token_count': value['usage']['completion_tokens'],
                      'finish_reason': choice['finish_reason'], 'inference_error': None,
                      'response_token_ids': choice.get('token_ids')}
            if choice['finish_reason'] not in ('stop', 'length'):
                raise ValueError('Unexpected finish reason')
            records.append(record)
            with (output / 'predictions.jsonl').open('a') as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
            print(f"INFERENCE {index}/{len(samples)} {sid} tokens={record['response_token_count']} finish={record['finish_reason']}", flush=True)
        write_json_atomic(output / 'summary.json', {'total': len(records), 'unique_sample_ids': len({r['sample_id'] for r in records}), 'not_accuracy_evaluation': True})
        summary['verification'] = verify_predictions(output, [sid for sid, _ in samples])
        if summary['verification']['status'] != 'PASS':
            raise ValueError(str(summary['verification']))
        summary['status'] = 'PASS'
    except BaseException as exc:
        summary['status'] = 'FAIL'
        summary['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        if server is not None:
            summary['server_exit_code_after_controlled_shutdown'] = stop_process_group(server)
        if before is not None:
            print('HASH_SOURCE_AFTER', flush=True)
            after = file_snapshot(files, include_sha256=True)
            write_json_atomic(output / 'checkpoint_manifest_after.json', {'files': after})
            summary['source_checkpoint_unchanged'] = before == after
            if before != after:
                summary['status'] = 'FAIL'
                summary['error'] = 'Source checkpoint changed'
        summary['completed_at_utc'] = utc_now()
        write_json_atomic(output / 'reload_validation_summary.json', summary)
        print('RELOAD_STATUS=' + summary['status'], flush=True)
    if summary['status'] != 'PASS':
        raise RuntimeError(summary.get('error', 'Cold reload failed'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/vopd_6241_pilot_64_reload.yaml')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--output-dir', type=Path, help='Fresh attempt directory; existing evidence is never overwritten')
    parser.add_argument('--reuse-merged-from', type=Path, help='Reuse only after source and merged SHA256 verification')
    args = parser.parse_args()
    execute(args.config.resolve(), args.run, args.output_dir, args.reuse_merged_from)
