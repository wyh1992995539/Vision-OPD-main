"""Early actual-length gate for the isolated pressure diagnostic (not normal training)."""
import json
from pathlib import Path

import torch

CONTRACT = dict(expected_rows=8, minimum_tokens=1000, maximum_tokens=1024)


def active():
    # Installed as <runtime>/verl/utils/pressure_runtime.py.
    return json.loads((Path(__file__).resolve().parents[2]/'validation_active.json').read_text())


def validate_receipt(receipt):
    lengths = receipt.get('response_lengths', [])
    return (receipt.get('status') == 'PASS_FIRST_BATCH_LENGTH'
            and receipt.get('step') == 1 and receipt.get('contract') == CONTRACT
            and receipt.get('response_width') == CONTRACT['maximum_tokens']
            and len(lengths) == CONTRACT['expected_rows']
            and all(type(n) is int and CONTRACT['minimum_tokens'] <= n <= CONTRACT['maximum_tokens']
                    for n in lengths))


def first_batch_check(trainer, batch):
    cfg = active()
    if cfg['stage'] != 'pressure' or cfg['mode'] != 'observe':
        raise ValueError('Pressure-only hook used outside the pressure diagnostic')
    if cfg.get('first_batch_length_gate') != CONTRACT:
        raise ValueError('Missing or changed first-batch length contract')
    if int(trainer.global_steps) != 1:
        return
    receipt = dict(status='FAIL_FIRST_BATCH_LENGTH', step=1, contract=CONTRACT,
                   checked_before='balance_ref_logprob_actor_update', response_lengths=[],
                   response_width=None, errors=[])
    try:
        mask, responses = batch.batch['response_mask'], batch.batch['responses']
        if mask.ndim != 2 or responses.ndim != 2 or mask.shape != responses.shape:
            raise ValueError('Response/mask shape mismatch')
        if not torch.all((mask == 0) | (mask == 1)):
            raise ValueError('Non-binary response mask')
        receipt['response_lengths'] = mask.sum(-1).to(torch.int64).tolist()
        receipt['response_width'] = int(responses.shape[-1])
        receipt['status'] = 'PASS_FIRST_BATCH_LENGTH'
        if not validate_receipt(receipt):
            raise ValueError('Every first-batch row must have 1000-1024 actual response tokens; expected 8 rows')
    except (KeyError, ValueError, TypeError) as exc:
        receipt['status'] = 'FAIL_FIRST_BATCH_LENGTH'
        receipt['errors'].append(str(exc))
    path = Path(cfg['output_dir'])/'evidence/first_batch_length_gate.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(receipt, stream, indent=2)
    if not validate_receipt(receipt):
        raise RuntimeError('PRESSURE_FIRST_BATCH_LENGTH_GATE_FAILED: '+ '; '.join(receipt['errors']))

