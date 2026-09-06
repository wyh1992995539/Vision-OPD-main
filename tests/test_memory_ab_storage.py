from scripts.prepare_memory_ab_storage import GIB, capacity_plan, file_inventory


def test_two_runs_need_floor_plus_one_retained_checkpoint():
    plan = capacity_plan(122 * GIB, 53 * GIB, 120 * GIB)
    assert plan['first_run_disk_gate_pass'] is True
    assert plan['both_runs_minimum_initial_free_bytes'] == 173 * GIB
    assert plan['minimum_additional_bytes'] == 51 * GIB
    assert plan['recommended_additional_bytes'] == 71 * GIB
    assert plan['second_run_free_estimate_bytes'] == 69 * GIB
    assert plan['status'] == 'BLOCKED_AB_DISK_CAPACITY'


def test_extra_capacity_does_not_create_negative_deficit():
    plan = capacity_plan(220 * GIB, 53 * GIB, 120 * GIB)
    assert plan['minimum_additional_bytes'] == 0
    assert plan['recommended_additional_bytes'] == 0
    assert plan['status'] == 'PASS_AB_STORAGE'


def test_exact_minimum_passes_with_limited_margin():
    plan = capacity_plan(173 * GIB, 53 * GIB, 120 * GIB)
    assert plan['status'] == 'PASS_AB_STORAGE_WITH_LIMITED_MARGIN'
    assert plan['minimum_additional_bytes'] == 0
    assert plan['recommended_additional_bytes'] == 20 * GIB


def test_inventory_preserves_files_and_does_not_hash_symlink_target(tmp_path):
    source = tmp_path / 'small.json'
    source.write_bytes(b'unchanged')
    link = tmp_path / 'link'
    link.symlink_to(source)
    rows = {r['path']: r for r in file_inventory(tmp_path)}
    assert rows[str(source)]['sha256']
    assert rows[str(source)]['size_bytes'] == 9
    assert 'sha256' not in rows[str(link)]
    assert rows[str(link)]['regular_file'] is False
    assert source.read_bytes() == b'unchanged'
    assert link.is_symlink()
