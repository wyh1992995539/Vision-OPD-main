import json
import tempfile
import unittest
from pathlib import Path

from eval.run_day6 import EXPECTED, load_amendment, read_jsonl
from eval.score_day6 import summarize


class Day6FullEvalTest(unittest.TestCase):
    def test_amendment_requires_official_only_vstar_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.yaml"
            path.write_text("status: effective\nexperiment_id: E-D6-001\neffective_reporting_policy:\n  vstar_summary_denominator: 191\n  vstar_official_request_count: 191\n  report_deduplicated_diagnostic_separately: false\n", encoding="utf-8")
            self.assertEqual(load_amendment(path)["effective_reporting_policy"]["vstar_summary_denominator"], 191)

    def test_checkpoint_compacts_duplicate_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            row = {"benchmark": "vstar", "view": "full", "sample_uid": "vstar:1"}
            path.write_text(json.dumps(row) + "\n" + json.dumps({**row, "raw_model_answer": "A"}) + "\n", encoding="utf-8")
            self.assertEqual(len(read_jsonl(path)), 1)

    def test_summary_keeps_vstar_official_denominator(self):
        amendment = {"effective_reporting_policy": {"vstar_summary_denominator": 191, "report_deduplicated_diagnostic_separately": False}}
        scores = []
        for group, count in EXPECTED.items():
            benchmark, view = group.split("/")
            scores.extend({"benchmark": benchmark, "view": view, "official_category": "x", "score_status": "scored", "is_correct": True} for _ in range(count))
        summary = summarize(scores, amendment)
        self.assertEqual(summary["groups"]["vstar/full"]["total"], 191)
        self.assertIsNone(summary["deduplicated_diagnostic"])
