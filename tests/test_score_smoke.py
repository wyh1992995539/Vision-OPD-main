import unittest
from decimal import Decimal

from eval.score_smoke import base_score, parse_single_numeric


def open_item(reference: str, prediction: str) -> dict:
    return {
        "benchmark": "zoombench",
        "view": "full",
        "sample_uid": "z1",
        "question_format": "open_question",
        "official_category": "unavailable_official",
        "reference_answer": reference,
        "raw_model_answer": f"<answer>{prediction}</answer>",
        "error": None,
    }


class ScoreSmokeTest(unittest.TestCase):
    def test_single_numeric_parser_is_strict(self):
        self.assertEqual(parse_single_numeric("1,000.50"), Decimal("1000.50"))
        self.assertEqual(parse_single_numeric("-3"), Decimal("-3"))
        self.assertIsNone(parse_single_numeric("there are 3"))
        self.assertIsNone(parse_single_numeric("3 items"))

    def test_numeric_equal_is_deterministic(self):
        score = base_score(open_item("1", "1.0"))
        self.assertTrue(score["is_correct"])
        self.assertEqual(score["score_source"], "deterministic_numeric_equal")
        self.assertEqual(score["score_status"], "scored")

    def test_numeric_mismatch_never_reaches_judge(self):
        score = base_score(open_item("1", "4"))
        self.assertFalse(score["is_correct"])
        self.assertEqual(score["score_source"], "deterministic_numeric_mismatch")
        self.assertEqual(score["score_status"], "scored")

    def test_semantic_unresolved_reaches_fixed_judge(self):
        score = base_score(open_item("six", "several objects"))
        self.assertFalse(score["is_correct"])
        self.assertEqual(score["score_source"], "fixed_base_4b_judge_required")
        self.assertEqual(score["score_status"], "pending_judge")


if __name__ == "__main__":
    unittest.main()
