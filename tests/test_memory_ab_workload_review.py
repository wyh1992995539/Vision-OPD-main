import copy

import pytest

from scripts.review_memory_ab_workload import log_lengths, pair_rollouts, review


def rows():
    return [dict(step=1, input=f'prompt {i}', output=f'answer {i}', gts='C') for i in range(8)]


def test_pair_by_unique_input_not_export_row_position():
    a = rows()
    b = list(reversed(copy.deepcopy(a)))
    pairs = pair_rollouts(a, b, 1)
    assert len(pairs) == 8
    assert all(p['output_text_equal'] for p in pairs)
    assert all(p['baseline_line'] != p['deferred_line'] for p in pairs)


@pytest.mark.parametrize('failure', ['duplicate', 'missing', 'input', 'label', 'step'])
def test_unsafe_or_incomplete_joins_are_rejected(failure):
    a, b = rows(), rows()
    if failure == 'duplicate':
        b[1]['input'] = b[0]['input']
    elif failure == 'missing':
        b.pop()
    elif failure == 'input':
        b[0]['input'] = 'wrong'
    elif failure == 'label':
        b[0]['gts'] = 'wrong'
    else:
        b[0]['step'] = 2
    with pytest.raises(ValueError):
        pair_rollouts(a, b, 1)


def test_log_reconciliation_uses_actual_not_non_aborted_fields(tmp_path):
    path = tmp_path / 'train.log'
    path.write_text('step:1 - prompt_length/max:100 - response_length/mean:2.5 - response_length/max:4 - response_length_non_aborted/mean:999\n')
    assert log_lengths(path) == [[1, 100, 2.5, 4]]
    path.write_text(path.read_text() * 2)
    with pytest.raises(ValueError, match='Duplicate'):
        log_lengths(path)


def test_changed_comparison_input_fails_before_analysis(tmp_path):
    import json
    source = tmp_path / 'raw.log'
    source.write_text('changed evidence')
    path = tmp_path / 'comparison.json'
    path.write_text(json.dumps(dict(status='REVIEW_WORKLOAD_DIFFERENCE', runs={
        'baseline': dict(status='PASS_MEMORY_AB_RUN', inputs=[dict(path=str(source), sha256='wrong')])})))
    with pytest.raises(ValueError, match='Changed audit input'):
        review(path)
