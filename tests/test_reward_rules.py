import copy
import unittest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "verl/utils/reward_score"))
from vision_opd_mcq_grpo import (
    DATA_SOURCE,
    REWARD_ROUTE,
    compute_score,
    extract_final_option,
)


def extra_info(**updates):
    value = {
        "reward_route": REWARD_ROUTE,
        "valid_options": ["A", "B", "C", "D"],
    }
    value.update(updates)
    return value


class VisionOpdMcqGrpoRewardTest(unittest.TestCase):
    def score(self, response, gold="D", **updates):
        return compute_score(
            data_source=DATA_SOURCE,
            solution_str=response,
            ground_truth=gold,
            extra_info=extra_info(**updates),
        )

    def test_supported_correct_answer_variants(self):
        responses = [
            "D",
            "d",
            "(D)",
            "[D]",
            "D.",
            "**D**",
            "$D$",
            r"\boxed{D}",
            "Answer: D",
            "answer is d",
            "Final answer: D.",
            "The answer is option D.",
            "Selected option: (D)",
            "答案：D",
            "最终答案为 D。",
            "<answer>D</answer>",
            "Reasoning mentions A and B.\nD",
        ]
        for response in responses:
            with self.subTest(response=response):
                result = self.score(response)
                self.assertEqual(result["score"], 1.0)
                self.assertEqual(result["prediction"], "D")
                self.assertTrue(result["parse_valid"])

    def test_explicit_wrong_answer_is_valid_but_scores_zero(self):
        result = self.score("Final answer: C")
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["prediction"], "C")
        self.assertTrue(result["parse_valid"])
        self.assertEqual(result["parse_status"], "parsed")

    def test_all_option_and_gold_pairs_score_exact_match_only(self):
        formats = (
            "{}",
            "Answer: {}",
            "Final answer: {}.",
            "答案：{}",
            "<answer>{}</answer>",
            r"\boxed{{{}}}",
        )
        for option in "ABCD":
            for gold in "ABCD":
                for template in formats:
                    response = template.format(option)
                    with self.subTest(response=response, gold=gold):
                        result = self.score(response, gold=gold)
                        self.assertEqual(result["prediction"], option)
                        self.assertEqual(result["score"], float(option == gold))

    def test_ambiguous_malformed_or_unmarked_outputs_score_zero(self):
        cases = {
            "": "empty_response",
            "I am not sure.": "no_explicit_final_answer",
            "A is wrong. B is plausible.": "no_explicit_final_answer",
            "Answer: A or D": "conflicting_final_answers",
            "Answer: A/D": "conflicting_final_answers",
            "Answer: A. Final answer: D.": "conflicting_final_answers",
            "<answer>A</answer><answer>D</answer>": "conflicting_final_answers",
            "<answer>D": "malformed_answer_tag",
            "<answer>Option D</answer>": "malformed_answer_tag",
            "Answer: E": "option_out_of_range",
            "Option A: red; Option B: blue; Option C: green; Option D: black": (
                "conflicting_final_answers"
            ),
            "The quoted answer says Answer: A. Final answer: D.": (
                "conflicting_final_answers"
            ),
        }
        for response, expected_status in cases.items():
            with self.subTest(response=response):
                result = self.score(response)
                self.assertEqual(result["score"], 0.0)
                self.assertFalse(result["parse_valid"])
                self.assertEqual(result["parse_status"], expected_status)

    def test_negated_option_is_not_treated_as_the_final_answer(self):
        result = self.score("The correct answer is not A. Final answer: D.")
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["prediction"], "D")
        self.assertTrue(result["parse_valid"])

    def test_answer_prefix_regression_does_not_extract_a(self):
        result = self.score("Answer: D")
        self.assertEqual(result["prediction"], "D")
        self.assertNotEqual(result["prediction"], "A")
        self.assertEqual(result["score"], 1.0)

    def test_prediction_extraction_does_not_depend_on_gold(self):
        response = "Final answer: C"
        result_c = self.score(response, gold="C")
        result_d = self.score(response, gold="D")
        self.assertEqual(result_c["prediction"], "C")
        self.assertEqual(result_d["prediction"], "C")
        self.assertEqual(result_c["score"], 1.0)
        self.assertEqual(result_d["score"], 0.0)

    def test_truncation_is_logged_without_overriding_an_explicit_answer(self):
        explicit = self.score("Answer: D", truncated=True)
        missing = self.score("The response was cut off", truncated=True)
        self.assertEqual(explicit["score"], 1.0)
        self.assertTrue(explicit["truncated"])
        self.assertEqual(missing["score"], 0.0)
        self.assertTrue(missing["truncated"])

    def test_compute_score_does_not_mutate_extra_info(self):
        info = extra_info(truncated=False)
        before = copy.deepcopy(info)
        compute_score(DATA_SOURCE, "D", "D", info)
        self.assertEqual(info, before)

    def test_dataset_contract_errors_raise(self):
        with self.assertRaisesRegex(ValueError, "data_source"):
            compute_score("wrong", "D", "D", extra_info())
        with self.assertRaisesRegex(ValueError, "reward_route"):
            compute_score(DATA_SOURCE, "D", "D", extra_info(reward_route="wrong"))
        with self.assertRaisesRegex(ValueError, "ground_truth"):
            compute_score(DATA_SOURCE, "D", "E", extra_info())
        with self.assertRaisesRegex(ValueError, "valid_options"):
            compute_score(
                DATA_SOURCE,
                "D",
                "D",
                extra_info(valid_options=["A", "B", "C"]),
            )
        with self.assertRaisesRegex(ValueError, "truncated"):
            compute_score(DATA_SOURCE, "D", "D", extra_info(truncated="false"))
        with self.assertRaisesRegex(ValueError, "extra_info"):
            compute_score(DATA_SOURCE, "D", "D", None)

    def test_non_string_response_is_an_invalid_model_output(self):
        result = self.score(None)
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["parse_status"], "empty_response")

    def test_parser_returns_one_candidate_for_repeated_same_answer(self):
        parsed = extract_final_option("Answer: D. Final answer: D.")
        self.assertTrue(parsed["parse_valid"])
        self.assertEqual(parsed["prediction"], "D")
        self.assertEqual(parsed["candidate_options"], ["D"])


if __name__ == "__main__":
    unittest.main()
