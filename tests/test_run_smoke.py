import unittest

from eval.run_smoke import final_answer, parse_mcq, score_records


class RunSmokeTest(unittest.TestCase):
    def test_final_answer_prefers_last_tag_after_thinking(self):
        text = "<think>A maybe</think> reasoning <answer>C</answer>"
        self.assertEqual(final_answer(text), "C")
        self.assertEqual(parse_mcq(text), ("C", "exact_final_option"))

    def test_mcq_parser_rejects_missing_option(self):
        self.assertEqual(parse_mcq("I cannot determine it."), ("", "invalid_or_ambiguous"))

    def test_score_records_keeps_open_question_pending(self):
        base = {
            "benchmark": "zoombench",
            "view": "full",
            "sample_uid": "z1",
            "category": "unavailable_official",
            "reference_answer": "A",
            "error": None,
            "latency_seconds": 1.0,
            "prompt_tokens": 10,
            "completion_tokens": 2,
        }
        mcq = {
            **base,
            "question_format": "multiple_choice",
            "raw_model_answer": "<answer>A</answer>",
        }
        opened = {
            **base,
            "sample_uid": "z2",
            "question_format": "open_question",
            "reference_answer": "word",
            "raw_model_answer": "<answer>word</answer>",
        }
        scores, summary = score_records([mcq, opened])
        self.assertTrue(scores[0]["is_correct"])
        self.assertEqual(scores[1]["score_status"], "pending_judge")
        self.assertEqual(summary["decision_status"], "pending_judge")


if __name__ == "__main__":
    unittest.main()
