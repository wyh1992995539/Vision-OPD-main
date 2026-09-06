#!/usr/bin/env python3
"""One-shot, exact-target retention cleanup authorized after Pilot64 cold reload."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat

ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / 'artifacts/runs/E-D11-6K-GATE-001'
ARCHIVE = GATE / 'pilot/16/attempts/attempt_003_pass_before_checkpoint_io_revision'
ACTOR = ARCHIVE / 'checkpoints/global_step_2/actor'
EVIDENCE = GATE / 'superseded_pilot16_cleanup'
NAMES = ('model_world_size_2_rank_0.pt', 'model_world_size_2_rank_1.pt',
         'optim_world_size_2_rank_0.pt', 'optim_world_size_2_rank_1.pt')


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def metadata(path):
    s = path.lstat()
    if not stat.S_ISREG(s.st_mode) or path.resolve() != path.absolute() or s.st_nlink != 1:
        raise ValueError(f'Unsafe file type/link count: {path}')
    return dict(path=str(path), size_bytes=s.st_size, allocated_bytes=s.st_blocks * 512,
                device=s.st_dev, inode=s.st_ino, mtime_ns=s.st_mtime_ns, nlink=s.st_nlink)


def protected_snapshot():
    # All existing evidence and current checkpoints survive, including archived small files.
    targets = {ACTOR / name for name in NAMES}
    paths = set(ARCHIVE.rglob('*'))
    for subtree in (GATE / 'pilot/16/checkpoints', GATE / 'pilot/64'):
        paths.update(subtree.rglob('*'))
    paths.update((GATE / 'pilot/16/evidence').rglob('*'))
    result = []
    for path in sorted(paths):
        if not path.is_file() or path in targets:
            continue
        item = metadata(path)
        if item['size_bytes'] < 32 * 1024**2:
            item['sha256'] = sha(path)
        result.append(item)
    return result


def check_readers():
    target_paths = {str(ACTOR / name) for name in NAMES}
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            descriptors = list((proc / 'fd').iterdir())
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        for fd in descriptors:
            try:
                link = os.readlink(fd)
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
            if link in target_paths:
                raise RuntimeError(f'Target in use by PID {proc.name}: {link}')


def prerequisites():
    for stage in ('16', '64'):
        p = json.loads((GATE / f'pilot/{stage}/evidence/postflight.json').read_text())
        if p['status'] != 'PASS' or not p['stage_gate_pass']:
            raise RuntimeError('Current Pilot did not pass')
    p64 = json.loads((GATE / 'pilot/64/evidence/postflight.json').read_text())
    reload = json.loads(Path(p64['reload_report']).read_text())
    if (reload['status'] != 'PASS' or not reload['source_checkpoint_unchanged']
            or Path(reload['source_checkpoint']) != GATE / 'pilot/64/checkpoints/global_step_8'):
        raise RuntimeError('Pilot64 cold reload is not valid')
    manifest = json.loads((ARCHIVE / 'archive_manifest.json').read_text())
    if manifest['previous_status'] != 'PASS':
        raise RuntimeError('Unexpected archive identity')
    sizes = {r['path']: r['size_bytes'] for r in manifest['files']}
    for name in NAMES:
        if metadata(ACTOR / name)['size_bytes'] != sizes[f'checkpoints/global_step_2/actor/{name}']:
            raise RuntimeError('Archived shard no longer matches archive manifest')
    check_readers()


def plan():
    if EVIDENCE.exists():
        raise RuntimeError('Evidence directory already exists; refusing overwrite')
    prerequisites()
    before = protected_snapshot()
    files = []
    for name in NAMES:
        path = ACTOR / name
        item = metadata(path)
        print('HASH ' + name, flush=True)
        item['sha256'] = sha(path)
        if metadata(path) != {k: v for k, v in item.items() if k != 'sha256'}:
            raise RuntimeError('File changed during hashing')
        files.append(item)
    free = shutil.disk_usage(ACTOR).free
    released = sum(r['allocated_bytes'] for r in files)
    value = dict(schema_version=1, created_at_utc=utc(), status='PLANNED',
                 authorization='User requested generate and execute safe disk cleanup/migration after proposal to retire superseded Pilot16 checkpoint.',
                 action='unlink_exactly_four_superseded_model_and_optimizer_shards',
                 archive=str(ARCHIVE), files=files, protected_files=before,
                 free_bytes_before=free, expected_released_bytes=released,
                 expected_free_bytes_after=free + released,
                 recovery='No backup made; deleted old tensors cannot be recovered from hashes. Historical small-file evidence and all current checkpoints retained.',
                 alternatives='No verified private backup destination available; same-disk moves do not release space.')
    write(EVIDENCE / 'plan.json', value)
    print(f"PLANNED_RELEASE_GIB={released / 1024**3:.3f}", flush=True)


def execute():
    receipt_path = EVIDENCE / 'receipt.json'
    if receipt_path.exists():
        raise RuntimeError('Receipt exists; refusing repeat execution')
    p = json.loads((EVIDENCE / 'plan.json').read_text())
    if [r['path'] for r in p['files']] != [str(ACTOR / name) for name in NAMES]:
        raise RuntimeError('Plan exceeds fixed allowlist')
    prerequisites()
    if protected_snapshot() != p['protected_files']:
        raise RuntimeError('Protected files changed since planning')
    # Complete all hash checks before deleting the first shard.
    for item in p['files']:
        path = Path(item['path'])
        print('VERIFY ' + path.name, flush=True)
        if (metadata(path) != {k: v for k, v in item.items() if k != 'sha256'}
                or sha(path) != item['sha256']):
            raise RuntimeError('Planned target changed')
    check_readers()
    receipt = dict(status='RUNNING', started_at_utc=utc(), plan_sha256=sha(EVIDENCE / 'plan.json'),
                   free_bytes_before=shutil.disk_usage(ACTOR).free, deleted=[], recovery=p['recovery'])
    write(receipt_path, receipt)
    try:
        for item in p['files']:
            path = Path(item['path'])
            if metadata(path) != {k: v for k, v in item.items() if k != 'sha256'}:
                raise RuntimeError('Target metadata changed immediately before unlink')
            path.unlink()
            receipt['deleted'].append(item)
            write(receipt_path, receipt)
            print('DELETED ' + path.name, flush=True)
        receipt['protected_files_unchanged'] = protected_snapshot() == p['protected_files']
        receipt['all_targets_absent'] = all(not (ACTOR / name).exists() for name in NAMES)
        receipt['free_bytes_after'] = shutil.disk_usage(ACTOR).free
        receipt['observed_free_bytes_increase'] = receipt['free_bytes_after'] - receipt['free_bytes_before']
        receipt['formal_disk_floor_met'] = receipt['free_bytes_after'] >= 120 * 1024**3
        receipt['status'] = 'PASS' if receipt['protected_files_unchanged'] and receipt['all_targets_absent'] else 'FAIL'
    except BaseException as exc:
        receipt['status'] = 'FAIL'
        receipt['error'] = repr(exc)
        raise
    finally:
        receipt['completed_at_utc'] = utc()
        write(receipt_path, receipt)
    write(ARCHIVE / 'checkpoint_retention.json', dict(
        status='HISTORICAL_EVIDENCE_ONLY_TENSOR_SHARDS_REMOVED', resumable=False,
        receipt=str(receipt_path), receipt_sha256=sha(receipt_path),
        retained='logs, telemetry, rollout outputs, configs, metadata, tokenizer, extra_state and data.pt',
        original_archive_manifest_preserved=True))
    print('CLEANUP_STATUS=' + receipt['status'], flush=True)
    print(f"FREE_GIB={receipt['free_bytes_after'] / 1024**3:.3f}", flush=True)
    if receipt['status'] != 'PASS':
        raise RuntimeError('Post-cleanup verification failed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    execute() if args.execute else plan()
