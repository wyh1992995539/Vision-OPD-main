import copy
import json

import pytest
import yaml

from scripts import freeze_formal_cpu as cpu
from scripts import retire_old_online_ab as cleanup


def test_cpu_review_uses_all_six_runs_not_only_latest_pressure():
    result = cpu.cpu_review(cpu.Reader())
    assert len(result['runs']) == 6
    assert result['maximum_peak_gib'] == pytest.approx(189.7866668701172)
    assert result['minimum_gib'] == 240
    assert result['abort_threshold_gib'] == 228
    assert result['headroom_to_abort_gib'] == pytest.approx(38.21333312988281)
    assert cpu.verify()


@pytest.mark.parametrize('bad', ['capacity', 'oom', 'empty', 'peak', 'negative'])
def test_cpu_review_rejects_bad_samples(bad):
    class Reader(cpu.Reader):
        def jsonl(self, path):
            samples = super().jsonl(path)
            if bad == 'empty': return []
            if bad == 'capacity': samples[-1]['memory_max_bytes'] = 224*cpu.GIB
            if bad == 'oom': samples[-1]['memory_events']['oom_kill'] += 1
            if bad == 'peak': samples[-1]['memory_current_bytes'] = int(.96*cpu.FLOOR)
            if bad == 'negative': samples[-1]['memory_current_bytes'] = -1
            return samples
    with pytest.raises(ValueError): cpu.cpu_review(Reader())


@pytest.mark.parametrize('bad', ['220', '224', 'algorithm', 'abort_ratio'])
def test_refreeze_allows_only_exact_resource_amendment(bad):
    name = 'vopd_6241_abort_policy.yaml' if bad == 'abort_ratio' else 'vopd_6241.yaml'
    before = yaml.safe_load((cpu.DIRECTORY/'before'/name).read_text())
    after = yaml.safe_load((cpu.ROOT/'configs'/name).read_text())
    if bad in ('220', '224'): after['resources']['prelaunch_cgroup_minimum_bytes'] = int(bad)*cpu.GIB
    elif bad == 'algorithm': after['actor']['learning_rate'] = 1e-6
    else: after['memory']['cgroup_used_ratio_abort'] = .99
    with pytest.raises(ValueError): cpu.only_cpu_change(before, after, name)


@pytest.mark.parametrize('bad', ['sources', 'peak', 'changes'])
def test_freeze_receipt_fails_closed_on_changes(tmp_path, bad):
    report = json.loads(cpu.REPORT.read_text())
    if bad == 'sources': report['sources'] = {}
    elif bad == 'peak': report['review']['maximum_peak_bytes'] = 1
    else: report['changes'] = []
    path = tmp_path/'freeze.json'
    path.write_text(json.dumps(report))
    assert not cpu.verify(path)


@pytest.fixture
def small_cleanup(tmp_path, monkeypatch):
    monkeypatch.setattr(cleanup, 'ROOT', tmp_path)
    monkeypatch.setattr(cleanup, 'GATE', tmp_path/'gate')
    monkeypatch.setattr(cleanup, 'AB', tmp_path/'gate/ab')
    monkeypatch.setattr(cleanup, 'PLAN', tmp_path/'plan.json')
    monkeypatch.setattr(cleanup, 'RECEIPT', tmp_path/'receipt.json')
    monkeypatch.setattr(cleanup, 'prerequisites', lambda: None)
    monkeypatch.setattr(cleanup, 'readers', lambda: None)
    for path in cleanup.targets():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'synthetic test shard')
    keep = cleanup.AB/'baseline/run/checkpoints/global_step_8/data.pt'
    keep.write_bytes(b'keep checkpoint metadata')
    cleanup.plan()
    return keep


def test_cleanup_plan_never_deletes_and_requires_explicit_approval(small_cleanup):
    assert all(p.exists() for p in cleanup.targets())
    with pytest.raises(ValueError): cleanup.execute(False)
    assert all(p.exists() for p in cleanup.targets())
    assert not cleanup.RECEIPT.exists()


def test_cleanup_exact_allowlist_preserves_metadata_and_is_not_repeatable(small_cleanup):
    cleanup.execute(True)
    assert not any(p.exists() for p in cleanup.targets())
    assert small_cleanup.read_bytes() == b'keep checkpoint metadata'
    receipt = json.loads(cleanup.RECEIPT.read_text())
    assert receipt['status'] == 'PASS'
    assert len(receipt['deleted']) == len(receipt['verified']) == 8
    with pytest.raises(FileExistsError): cleanup.execute(True)


@pytest.mark.parametrize('bad', ['target_changed', 'protected_changed', 'extra_target', 'hash_failure'])
def test_cleanup_fails_before_unlink_on_changed_plan_or_files(small_cleanup, bad, monkeypatch):
    if bad == 'target_changed': cleanup.targets()[0].write_bytes(b'changed')
    elif bad == 'protected_changed': small_cleanup.write_bytes(b'changed metadata')
    elif bad == 'extra_target':
        plan = json.loads(cleanup.PLAN.read_text())
        plan['files'][0]['path'] = str(small_cleanup)
        cleanup.PLAN.write_text(json.dumps(plan))
    else:
        original = cleanup.sha
        def fail(path):
            if path == cleanup.targets()[2]: raise OSError('synthetic read failure')
            return original(path)
        monkeypatch.setattr(cleanup, 'sha', fail)
    with pytest.raises((ValueError, RuntimeError, OSError)): cleanup.execute(True)
    assert all(p.exists() for p in cleanup.targets())
