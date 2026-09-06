#!/usr/bin/env python3
"""Exact-target old online A/B retirement. Planning never deletes files."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.cleanup_superseded_pilot16 import metadata, sha, write, utc

GATE = ROOT/'artifacts/runs/E-D11-6K-GATE-001'
AB = GATE/'memory_optimization/ab'
EVIDENCE = GATE/'resource_refreeze_v1'
PLAN = EVIDENCE/'disk_plan.json'
RECEIPT = EVIDENCE/'disk_cleanup_receipt.json'
VARIANTS = ('baseline', 'deferred_v2')
NAMES = ('model_world_size_2_rank_0.pt', 'model_world_size_2_rank_1.pt',
         'optim_world_size_2_rank_0.pt', 'optim_world_size_2_rank_1.pt')


def targets():
    return [AB/v/'run/checkpoints/global_step_8/actor'/name for v in VARIANTS for name in NAMES]


def readers():
    names = {str(p) for p in targets()}
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            fds = list((proc/'fd').iterdir())
        except (OSError, PermissionError):
            continue
        for fd in fds:
            try:
                target = os.readlink(fd)
            except OSError:
                continue
            if target in names:
                raise RuntimeError('Target currently open: '+target)


def protected():
    excluded = set(targets())
    paths = set()
    for v in VARIANTS:
        paths.update((AB/v/'run').rglob('*'))
    for stage in ('pilot/16', 'pilot/64', 'memory_optimization/fixed_validation_v1/capture/run',
                  'memory_optimization/fixed_validation_v1/fixed_baseline/run',
                  'memory_optimization/fixed_validation_v1/fixed_deferred/run',
                  'memory_optimization/fixed_validation_v1/pressure_v2/run'):
        paths.update((GATE/stage/'checkpoints').rglob('*'))
    result = []
    for p in sorted(paths-excluded):
        if p.is_file():
            item = metadata(p)
            if item['size_bytes'] < 1024**2:
                item['sha256'] = sha(p)
            result.append(item)
    return result


def prerequisites():
    pressure = json.loads((GATE/'memory_optimization/fixed_validation_v1/pressure_v2/run/evidence/postflight.json').read_text())
    if pressure['status'] != 'PASS_PRESSURE_DIAGNOSTIC':
        raise RuntimeError('Pressure successor did not pass')
    for v in VARIANTS:
        report = json.loads((AB/v/'run/evidence/postflight.json').read_text())
        # Baseline historical outer FAIL was an audit accounting issue; its
        # successful training remains historical, and is NOT rewritten here.
        if report['training_audit']['training_gate_pass'] is not True:
            raise RuntimeError('Unexpected old A/B training identity')
    readers()


def plan():
    if PLAN.exists():
        raise FileExistsError('Plan already exists; preserve it')
    prerequisites()
    files = [metadata(p) for p in targets()]
    free = shutil.disk_usage(ROOT).free
    reclaimed = sum(f['allocated_bytes'] for f in files)
    write(PLAN, dict(status='PENDING_EXPLICIT_DELETION_APPROVAL', generated_at_utc=utc(), files=files,
        protected_files=protected(), free_bytes_before=free, expected_reclaimed_bytes=reclaimed,
        expected_free_bytes_after=free+reclaimed, minimum_launch_free_bytes=120*1024**3,
        candidate_then_formal_recommended_bytes=193.12*1024**3,
        full_tensor_hashes_verified=False, backup_exists=False,
        recovery='Deleted old online A/B tensor shards cannot be restored; no backup exists.',
        alternatives='No private separate backup mount verified; same-disk moves do not release quota.',
        retained='All small old A/B evidence; all Pilot, fixed A/B and pressure_v2 checkpoints and replay payloads.'))
    print(f'PLANNED_RECLAIM_GIB={reclaimed/1024**3:.2f}')
    print(f'EXPECTED_FREE_GIB={(free+reclaimed)/1024**3:.2f}')


def execute(approved):
    if not approved:
        raise ValueError('Explicit confirmation is required; planning does not authorize deletion')
    if RECEIPT.exists():
        raise FileExistsError('Do not repeat retirement')
    p = json.loads(PLAN.read_text())
    if [f['path'] for f in p['files']] != [str(t) for t in targets()]:
        raise ValueError('Plan exceeds exact allowlist')
    prerequisites()
    if protected() != p['protected_files']:
        raise RuntimeError('Protected files changed since plan')
    verified = []
    for item in p['files']:
        path = Path(item['path'])
        if metadata(path) != item:
            raise RuntimeError('Planned target changed')
        print('HASH '+str(path), flush=True)
        digest = sha(path)
        if metadata(path) != item:
            raise RuntimeError('Target changed while hashing')
        verified.append(dict(**item, sha256=digest))
    readers()
    receipt = dict(status='RUNNING', started_at_utc=utc(), plan_sha256=sha(PLAN), deleted=[],
                   verified=verified, free_before=shutil.disk_usage(ROOT).free,
                   recovery=p['recovery'], authorization='Explicit user approval of the eight old online A/B tensor shards')
    write(RECEIPT, receipt)
    try:
        for item in p['files']:
            path = Path(item['path'])
            if metadata(path) != item:
                raise RuntimeError('Target changed immediately before deletion')
            path.unlink()
            receipt['deleted'].append(item['path'])
            write(RECEIPT, receipt)
        receipt['protected_unchanged'] = protected() == p['protected_files']
        if not receipt['protected_unchanged']:
            raise RuntimeError('Protected files changed')
        receipt['status'] = 'PASS'
    except BaseException as exc:
        receipt.update(status='FAIL', error=repr(exc))
        raise
    finally:
        receipt.update(finished_at_utc=utc(), free_after=shutil.disk_usage(ROOT).free)
        write(RECEIPT, receipt)
    for v in VARIANTS:
        write(AB/v/'checkpoint_retention.json', dict(status='HISTORICAL_EVIDENCE_ONLY_LARGE_SHARDS_RETIRED',
              resumable=False, recoverable=False, receipt=str(RECEIPT), receipt_sha256=sha(RECEIPT)))
    print('CLEANUP='+receipt['status'])
    print(f"FREE_GIB={receipt['free_after']/1024**3:.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--confirm-delete-old-online-ab-shards', action='store_true')
    args = parser.parse_args()
    execute(args.confirm_delete_old_online_ab_shards) if args.execute else plan()
