import copy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.vopd_6241_pilot_reload import runtime_env, select_samples, student_messages
from scripts.vopd_day8_reload import stable_key, verify_predictions


class PilotReloadTest(unittest.TestCase):
    def test_runtime_overrides_invalid_thread_environment(self):
        from unittest.mock import patch
        with patch.dict('os.environ', {'OMP_NUM_THREADS': '0', 'MKL_NUM_THREADS': ''}):
            env = runtime_env(CUDA_VISIBLE_DEVICES='0,1')
            self.assertEqual(env['OMP_NUM_THREADS'], '4')
            self.assertEqual(env['MKL_NUM_THREADS'], '4')
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '0,1')

    def row(self, sid, split='train'):
        return {'extra_info': {'provenance': {'sample_id': sid, 'split': split}}}

    def test_selection_is_deterministic_under_permutation(self):
        rows = [self.row(str(i)) for i in range(64)]
        selected = [sid for sid, _ in select_samples(rows, 5, 42)]
        self.assertEqual(selected, [sid for sid, _ in select_samples(rows[::-1], 5, 42)])
        self.assertEqual(selected, sorted([str(i) for i in range(64)], key=lambda sid: stable_key(42, sid))[:5])

    def test_rejects_eval_duplicate_and_invalid_counts(self):
        for rows, count in [([self.row('x', 'eval')], 1), ([self.row('x')] * 2, 1), ([self.row('x')], 2)]:
            with self.subTest(rows=rows, count=count), self.assertRaises(ValueError):
                select_samples(rows, count, 42)

    def test_only_student_inputs_are_sent(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'student.png'
            path.write_bytes(b'image-bytes')
            row = {'prompt': [{'role': 'user', 'content': 'before<image>after'}],
                   'images': [{'path': str(path)}], 'bbox_images': [{'path': 'FORBIDDEN'}],
                   'reward_model': {'ground_truth': 'SECRET_ANSWER'}}
            before = copy.deepcopy(row)
            result = student_messages(row)
            self.assertEqual(row, before)
            self.assertNotIn('FORBIDDEN', json.dumps(result))
            self.assertNotIn('SECRET_ANSWER', json.dumps(result))
            self.assertEqual([item['type'] for item in result[0]['content']], ['text', 'image_url', 'text'])

    def test_prediction_order_and_empty_response_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            row = {'sample_id': 'wrong', 'raw_prediction': '', 'response_token_count': 0, 'finish_reason': 'error'}
            (path / 'predictions.jsonl').write_text(json.dumps(row) + '\n')
            (path / 'summary.json').write_text(json.dumps({'total': 1, 'unique_sample_ids': 1}))
            self.assertEqual(verify_predictions(path, ['expected'])['status'], 'FAIL')
