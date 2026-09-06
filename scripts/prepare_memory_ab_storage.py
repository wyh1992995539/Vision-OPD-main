#!/usr/bin/env python3
"""Read-only storage inventory; writes a plan, never moves/deletes training files."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess

import yaml

ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / 'artifacts/runs/E-D11-6K-GATE-001'
GIB = 1024**3


def capacity_plan(free, checkpoint, floor, margin=20 * GIB):
    # Sequential runs: before run B, one new checkpoint is already retained.
    minimum = floor + checkpoint
    status = ('BLOCKED_AB_DISK_CAPACITY' if free < minimum else
              'PASS_AB_STORAGE_WITH_LIMITED_MARGIN' if free < minimum + margin else
              'PASS_AB_STORAGE')
    return dict(status=status, checkpoint_estimate_bytes=checkpoint, launch_floor_bytes=floor,
                first_run_disk_gate_pass=free >= floor,
                second_run_free_estimate_bytes=free - checkpoint,
                both_runs_minimum_initial_free_bytes=minimum,
                minimum_additional_bytes=max(0, minimum - free),
                planning_margin_bytes=margin,
                recommended_initial_free_bytes=minimum + margin,
                recommended_additional_bytes=max(0, minimum + margin - free),
                archive_destination_minimum_bytes=checkpoint + margin,
                caveat='Checkpoint sizes and ancillary writes may change; recheck live gates before each launch.')


def allocated(path):
    return int(subprocess.check_output(
        ['du', '-sx', '-B1', str(path)], text=True).split()[0])


def file_inventory(directory):
    result = []
    for path in sorted(directory.rglob('*')):
        info = path.lstat()
        if stat.S_ISDIR(info.st_mode):
            continue
        item = dict(path=str(path), size_bytes=info.st_size,
                    allocated_bytes=info.st_blocks * 512, device=info.st_dev,
                    inode=info.st_ino, mtime_ns=info.st_mtime_ns, nlink=info.st_nlink,
                    regular_file=stat.S_ISREG(info.st_mode))
        if path.is_symlink():
            item['symlink_target'] = os.readlink(path)
        # Full shard hashing belongs to an authorized copy/verify phase.
        if item['regular_file'] and info.st_size < 1024**2:
            item['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        result.append(item)
    return result


def filesystem(path):
    usage = shutil.disk_usage(path)
    mount = json.loads(subprocess.check_output(
        ['findmnt', '-J', '-T', str(path), '-o', 'TARGET,SOURCE,FSTYPE,OPTIONS'], text=True))
    return dict(path=str(path), resolved_path=str(path.resolve()),
                total_bytes=usage.total, used_bytes=usage.used, free_bytes=usage.free,
                mount=mount['filesystems'][0])


def prepare(output, previous=None):
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite evidence: {output}')
    output.resolve().relative_to(GATE.resolve())
    disks = [filesystem(p) for p in (ROOT.parent, Path('/'), Path('/tmp'), Path('/autodl-pub'))]
    workspace = []
    for path in sorted(ROOT.parent.iterdir()):
        if path.is_dir() and not path.is_symlink():
            workspace.append(dict(path=str(path), allocated_bytes=allocated(path)))
    protected = [GATE / 'pilot/16/checkpoints', GATE / 'pilot/64/checkpoints',
                 GATE / 'pilot/64/cold_reload/merged_hf']
    checkpoints = [dict(path=str(p), allocated_bytes=allocated(p), files=file_inventory(p),
                        action='PRESERVE_IN_PLACE', full_content_integrity_verified=False)
                   for p in protected]
    policy_path = ROOT / 'configs/vopd_6241_pilot_abort_policy.yaml'
    policy = yaml.safe_load(policy_path.read_text())
    checkpoint = max(policy['disk']['checkpoint_estimate_bytes'],
                     *(x['allocated_bytes'] for x in checkpoints[:2]))
    budget = capacity_plan(disks[0]['free_bytes'], checkpoint, policy['disk']['prelaunch_required_bytes'])
    previous_entry = None
    if previous is not None:
        previous = previous.resolve()
        historical = json.loads(previous.read_text())
        old_disk = historical['filesystem_inventory'][0]
        if old_disk['resolved_path'] != disks[0]['resolved_path']:
            raise ValueError('Previous audit targets a different training path')
        previous_entry = dict(path=str(previous), sha256=hashlib.sha256(previous.read_bytes()).hexdigest(),
                              total_bytes=old_disk['total_bytes'], free_bytes=old_disk['free_bytes'],
                              capacity_increase_bytes=disks[0]['total_bytes'] - old_disk['total_bytes'],
                              protected_file_metadata_unchanged=(
                                  historical['protected_checkpoints'] == checkpoints),
                              comparison_scope='File metadata and small-file hashes, not full shard content verification')
    enough = budget['minimum_additional_bytes'] == 0
    recommendation = (
        'Measured capacity covers sequential A/B checkpoints; no migration is required for this plan. Recheck live launch gates.'
        if enough else 'Expand existing training filesystem quota; preserve paths and evidence.')
    plan = dict(schema_version=1, generated_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                status=budget['status'], previous_audit=previous_entry,
                migration_required_for_ab_capacity=not enough,
                formal_training_authorized=False,
                scope='Inventory and preparation only; no GPU, copy, move, delete, symlink or policy changes.',
                automatic_source_removal_authorized=False, destination=None,
                filesystem_inventory=disks, workspace_inventory=workspace,
                protected_checkpoints=checkpoints, ab_capacity=budget,
                policy_sha256=hashlib.sha256(policy_path.read_bytes()).hexdigest(),
                preferred_action=recommendation,
                alternate_action='After baseline passes, archive its new checkpoint to a confirmed private durable destination.',
                future_archive_source=str(GATE / 'memory_optimization/ab/baseline/run/checkpoints'),
                migration_steps=[
                    'Confirm private destination path, durable retention and actual quota; never use public/read-only or RAM filesystems.',
                    'Check actual source size, destination free space and different storage/quota; same-disk moves do not release space.',
                    'Wait for baseline guard/postflight/checkpoint completion and no live writers; preserve all logs and original manifests.',
                    'Create source per-file SHA256 inventory; copy into a fresh non-overwriting staging directory on destination.',
                    'Compare file set, sizes and full SHA256 at source and destination; recheck source has not changed.',
                    'Record destination, retention and restoration instructions; verify restore/audit path compatibility.',
                    'Keep source until verification succeeds AND user separately authorizes exact source retirement.',
                    'Recheck 120 GiB source free floor and all other live guards before deferred run.'
                ],
                limitations=['No private migration destination was supplied or verified.',
                             'Large tensor files were inventoried by metadata, not reread for SHA256 in this preparation phase.',
                             'Snapshots are not backup copies or permission to remove protected checkpoints.'])
    output.mkdir(parents=True)
    (output / 'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n')
    rows = '\n'.join(f"| {p['path']} | {p['allocated_bytes']/GIB:.2f} |" for p in checkpoints)
    conclusion = ('当前容量满足两组 A/B 顺序运行的 checkpoint 空间需求，无须为本计划迁移文件。'
                  if enough else '当前容量不足以连续保留两组 A/B checkpoint，需要扩容或确认私有迁移目标。')
    recommendation_zh = ('维持当前路径和保留策略，启动前重新检查实时资源。'
                         if enough else '补足上述容量缺口和规划余量，再重新检查实际可用空间。')
    history = (f"历史记录：`{previous_entry['path']}`；SHA256：`{previous_entry['sha256']}`。\n"
               f"总容量从 {previous_entry['total_bytes']/GIB:.2f} GiB 增至 {disks[0]['total_bytes']/GIB:.2f} GiB；"
               f"受保护文件元数据及小文件哈希一致：`{previous_entry['protected_file_metadata_unchanged']}`。"
               if previous_entry else '本次未指定历史对照记录。')
    report = f'''# A/B 磁盘盘点与迁移准备

状态：`{plan['status']}`。{conclusion}
本次仅生成存储审计；正式训练授权：`false`。

{history}

## 当前空间

训练盘总容量 {disks[0]['total_bytes']/GIB:.2f} GiB，可用 {disks[0]['free_bytes']/GIB:.2f} GiB。
系统盘与 /tmp 属于同一个 overlay，可用 {disks[1]['free_bytes']/GIB:.2f} GiB；不能把两者相加。
/autodl-pub 是只读公共盘，不能作为备份目标；tmpfs 不是持久化磁盘。

| 保留原位的对象 | 已分配 GiB |
| --- | ---: |
{rows}

完整目录占用、文件大小、inode、mtime 和挂载信息见 plan.json。
大张量未做全量内容哈希，这不是备份完成或 checkpoint 完整性认证。

## 两组顺序运行的容量计算

- 每组 checkpoint 按 {checkpoint/GIB:.2f} GiB 估算；启动下限 {budget['launch_floor_bytes']/GIB:.2f} GiB 不变。
- 第二组启动前已保留第一组 checkpoint，因此初始至少需 {budget['both_runs_minimum_initial_free_bytes']/GIB:.2f} GiB 空闲。
- 当前最少还差 {budget['minimum_additional_bytes']/GIB:.2f} GiB；另留 20 GiB 规划余量后，建议增加至少 {budget['recommended_additional_bytes']/GIB:.2f} GiB。
- 这是空间规划，不是放行；日志、临时文件、checkpoint 大小变化仍需现场检查。

## 当前存储结论

{conclusion}
{recommendation_zh}
可用 `df -B1 /root/autodl-tmp` 核实后续实际容量变化。

## 后续需要归档时的参考方案（本次不执行）

先由用户提供私有、持久化目标路径，并核实其空间和独立配额。
目标至少需约 {budget['archive_destination_minimum_bytes']/GIB:.2f} GiB 空闲（含 20 GiB 余量）。
baseline 完成后才会有这个源目录；当前不迁移现有 Pilot-16/Pilot-64 或 merged HF。

迁移顺序：停止写入 → 源文件清单及全量 SHA256 → 复制到全新 staging 目录 →
核对目标文件集合/大小/SHA256、源未变更 → 写归档及恢复凭据 → 经另行授权才移除源副本 → 重新检查启动门槛。
仅复制不释放源盘空间；同盘改目录也不释放空间。不得通过删除唯一副本或降低门槛来放行。
现有审计可能引用原始绝对路径；不能擅自改历史 JSON 或假设软链接一定兼容，迁移前须验证恢复/审计流程。

## 下一步

{recommendation_zh}
存储通过不替代 GPU 显存验证、正式 CPU 门槛冻结和最终 Gate。
'''
    (output / 'report.md').write_text(report)
    (output / 'sha256.txt').write_text(''.join(
        hashlib.sha256((output / name).read_bytes()).hexdigest() + '  ' + str((output / name).resolve()) + '\n'
        for name in ('plan.json', 'report.md')))
    print(json.dumps(dict(status=plan['status'], output=str(output), ab_capacity=budget), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=GATE / 'memory_optimization/storage_preparation')
    parser.add_argument('--previous', type=Path, help='Previous immutable disk audit plan.json')
    args = parser.parse_args()
    prepare(args.output, args.previous)
