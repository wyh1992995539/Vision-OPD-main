import copy
import unittest

from eval.mcq_parser_r4 import classify_mcq_answer
from eval.paper_aligned_common import load_config
from eval.score_paper_aligned import rule_score
from tests.test_paper_aligned_pipeline import prediction


class McqParserR4Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, r3 = load_config("configs/benchmark_eval_paper_basejudge_r3_single_gpu.yaml")
        cls.config = copy.deepcopy(r3)
        cls.config["judge"]["mcq_parser_revision"] = 4

    def test_answer_marker_does_not_become_option_a(self):
        result = classify_mcq_answer("A", "Answer: **D**")
        self.assertEqual(result["status"], "mismatch")
        self.assertEqual(result["predicted_option"], "D")

    def test_explicit_match_formats(self):
        for answer in (
            "<answer>B</answer>",
            "Answer: B",
            "The answer is (B).",
            "Final Answer: **B**",
            "Correct option: **(B) blue**",
            "正确答案是：**B**",
            "Final Answer: $\\boxed{B}$",
            "reasoning complete\n\nB",
            "B",
            "(B).",
        ):
            with self.subTest(answer=answer):
                self.assertEqual(classify_mcq_answer("B", answer)["status"], "match")

    def test_arbitrary_capital_and_conflict_are_ambiguous(self):
        self.assertEqual(classify_mcq_answer("A", "I cannot tell.")["status"], "ambiguous")
        self.assertEqual(
            classify_mcq_answer("A", "Answer: A. Final Answer: C.")["status"],
            "ambiguous",
        )

    def test_explicit_mismatch_is_incorrect_without_judge(self):
        item = prediction("mmstar", "Answer: **D**")
        item["reference_answer"] = "A"
        score = rule_score(item, self.config, lambda reference, answer: False)
        self.assertEqual(score["rule_source"], "mcq_option_mismatch_r4")
        self.assertEqual(score["mcq_predicted_option"], "D")
        self.assertFalse(score["judge_required"])
        self.assertFalse(score["final_is_correct"])

    def test_unparseable_answer_routes_to_frozen_judge(self):
        score = rule_score(
            prediction("vstar", "I cannot determine the answer."),
            self.config,
            lambda reference, answer: False,
        )
        self.assertEqual(score["mcq_parse_status"], "ambiguous")
        self.assertEqual(score["rule_source"], "llm_judge_required")
        self.assertTrue(score["judge_required"])
        self.assertEqual(score["score_status"], "pending_judge")

    def test_r3_behavior_is_unchanged_without_revision_flag(self):
        _, r3 = load_config("configs/benchmark_eval_paper_basejudge_r3_single_gpu.yaml")
        item = prediction("mmstar", "Answer: **D**")
        item["reference_answer"] = "A"
        score = rule_score(item, r3, lambda reference, answer: False)
        self.assertEqual(score["rule_source"], "first_letter")
        self.assertTrue(score["final_is_correct"])


if __name__ == "__main__":
    unittest.main()
