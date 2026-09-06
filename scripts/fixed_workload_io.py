"""Lossless CPU actor-input capture/replay for isolated diagnostics, never formal OPD."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def pack(value):
    # No arbitrary pickle objects: torch.load uses weights_only=True.
    from PIL import Image
    if isinstance(value, Image.Image):
        return {'kind': 'image', 'value': value.tobytes(), 'mode': value.mode, 'size': list(value.size)}
    if isinstance(value, torch.Tensor):
        return {'kind': 'tensor', 'value': value.detach().cpu().clone()}
    if isinstance(value, np.ndarray):
        return {'kind': 'array', 'dtype': value.dtype.str, 'shape': list(value.shape),
                'value': pack(value.tolist())}
    if isinstance(value, np.generic):
        return pack(value.item())
    if isinstance(value, dict):
        if not all(isinstance(k, (str, int)) for k in value):
            raise TypeError('Unsupported mapping key')
        return {'kind': 'dict', 'value': [(k, pack(v)) for k, v in value.items()]}
    if isinstance(value, (list, tuple)):
        return {'kind': 'tuple' if isinstance(value, tuple) else 'list', 'value': [pack(v) for v in value]}
    if value is None or type(value) in (str, int, float, bool, bytes):
        return {'kind': 'scalar', 'value': value}
    raise TypeError(f'Unsupported actor input type: {type(value).__name__}; capture refused')


def unpack(value):
    kind, data = value['kind'], value['value']
    if kind == 'image':
        from PIL import Image
        return Image.frombytes(value['mode'], tuple(value['size']), data)
    if kind in ('scalar', 'tensor'):
        return data
    if kind == 'dict':
        return {k: unpack(v) for k, v in data}
    if kind in ('tuple', 'list'):
        items = [unpack(v) for v in data]
        return tuple(items) if kind == 'tuple' else items
    if kind == 'array':
        decoded = unpack(data)
        if np.dtype(value['dtype']).hasobject:
            array = np.empty(value['shape'], dtype=object)
            # Non-tensor batches are normally 1-D object arrays of multimodal dictionaries.
            if len(value['shape']) != 1:
                raise ValueError('Only 1-D object arrays are supported')
            for i, item in enumerate(decoded):
                array[i] = item
            return array
        return np.asarray(decoded, dtype=value['dtype']).reshape(value['shape'])
    raise ValueError(f'Unknown packed kind: {kind}')


def tensor_hash(tensor):
    t = tensor.detach().cpu().contiguous()
    prefix = json.dumps([str(t.dtype), list(t.shape)]).encode()
    return hashlib.sha256(prefix + t.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def batch_summary(data):
    keys = ('input_ids', 'responses', 'response_mask', 'attention_mask', 'position_ids',
            'teacher_input_ids', 'teacher_attention_mask', 'teacher_position_ids',
            'teacher_response_start_idx', 'old_log_probs', 'self_distillation_mask')
    missing = set(keys) - set(data.batch.keys())
    if missing:
        raise ValueError(f'Missing full actor fields: {sorted(missing)}')
    mask = data.batch['response_mask']
    if not torch.all((mask == 0) | (mask == 1)):
        raise ValueError('Non-binary response mask')
    return dict(rows=len(data), tensor_hashes={k: tensor_hash(data.batch[k]) for k in keys},
                response_lengths=mask.sum(-1).to(torch.int64).tolist(),
                response_width=int(data.batch['responses'].shape[-1]))


def save_batch(data, path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    summary = batch_summary(data)
    payload = dict(batch=pack(dict(data.batch.items())), non_tensor_batch=pack(data.non_tensor_batch),
                   meta_info=pack(data.meta_info), batch_size=list(data.batch.batch_size))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        torch.save(payload, stream)
    return dict(path=str(path.resolve()), sha256=sha(path), **summary)


def load_batch(entry):
    path = Path(entry['path'])
    if sha(path) != entry['sha256']:
        raise ValueError('Captured batch hash mismatch')
    value = torch.load(path, map_location='cpu', weights_only=True)
    from tensordict import TensorDict
    from verl import DataProto
    data = DataProto(batch=TensorDict(unpack(value['batch']), batch_size=value['batch_size']),
                     non_tensor_batch=unpack(value['non_tensor_batch']), meta_info=unpack(value['meta_info']))
    if batch_summary(data) != {k: entry[k] for k in ('rows', 'tensor_hashes', 'response_lengths', 'response_width')}:
        raise ValueError('Captured batch semantic mismatch')
    return data


def active():
    # Copied into <isolated runtime>/verl/utils; no environment propagation dependency.
    path = Path(__file__).resolve().parents[2] / 'validation_active.json'
    return json.loads(path.read_text())


def actor_input(trainer, data):
    cfg = active()
    step = int(trainer.global_steps)
    evidence = Path(cfg['output_dir']) / 'evidence/fixed_workload'
    evidence.mkdir(parents=True, exist_ok=True)
    if cfg['mode'] == 'capture':
        entry = save_batch(data, evidence / f'step{step:04d}.pt')
    elif cfg['mode'] == 'replay':
        launch = json.loads((Path(cfg['output_dir'])/'preflight/validation_launch.json').read_text())
        if sha(cfg['bundle_manifest']) != launch['bundle_sha256']:
            raise ValueError('Sealed bundle changed after launch')
        manifest = json.loads(Path(cfg['bundle_manifest']).read_text())
        entry = manifest['batches'][step - 1]
        if entry['step'] != step:
            raise ValueError('Replay step mismatch')
        replay = load_batch(entry)
        if replay.meta_info['global_steps'] != step:
            raise ValueError('Captured global step mismatch')
        # Mutate caller's DataProto too so training metrics/rollout export describe replayed inputs.
        data.batch, data.non_tensor_batch, data.meta_info = replay.batch, replay.non_tensor_batch, replay.meta_info
    else:
        entry = batch_summary(data)
    receipt = dict(entry, step=step, mode=cfg['mode'])
    with (evidence / f'step{step:04d}.json').open('x') as stream:
        json.dump(receipt, stream, indent=2)
    return data


def microbatch_plan(actor, batches):
    cfg = active()
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    step = int(actor._current_global_steps)
    # Includes exact rank-local order, token/mask hashes, teacher tensors and multimodal fields.
    output = Path(cfg['output_dir']) / 'evidence/fixed_workload'
    output.mkdir(parents=True, exist_ok=True)
    path = output / f'step{step:04d}.rank{rank}.json'
    plans = []
    for data in batches:
        summary = batch_summary(data)
        # Hash all nested non-tensor multimodal inputs using deterministic tensor-aware traversal.
        def fingerprint(value):
            if isinstance(value, torch.Tensor):
                return ['tensor', tensor_hash(value)]
            if isinstance(value, np.ndarray):
                return ['array', value.dtype.str, list(value.shape), fingerprint(value.tolist())]
            if isinstance(value, dict):
                return {str(k): fingerprint(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
            if isinstance(value, (list, tuple)):
                return [fingerprint(v) for v in value]
            if isinstance(value, np.generic):
                return value.item()
            return value
        summary['multimodal_sha256'] = hashlib.sha256(json.dumps(fingerprint(data.non_tensor_batch), sort_keys=True).encode()).hexdigest()
        plans.append(summary)
    with path.open('x') as stream:
        json.dump(dict(step=step, rank=rank, microbatches=plans), stream, indent=2)
