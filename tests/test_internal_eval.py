import tempfile
import unittest
from pathlib import Path

from eval.internal_eval import (
    build_prediction_record,
    load_jsonl,
    parse_multiple_choice,
    score_prediction,
    summarize_predictions,
    write_jsonl_atomic,
)
from eval.run_internal_eval import extract_eval_sample


class MultipleChoiceParsingTest(unittest.TestCase):
    def test_accepts_supported_answer_forms(self):
        cases = {
            "D": "D",
            "D.": "D",
            "(D)": "D",
            "Answer: D": "D",
            "The answer is D.": "D",
            "<answer>D</answer>": "D",
            "D. The padlock is gold.": "D",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                parsed = parse_multiple_choice(raw)
                self.assertEqual(parsed["parse_status"], "parsed")
                self.assertEqual(parsed["parsed_choice"], expected)

    def test_removes_thinking_before_parsing_final_answer(self):
        parsed = parse_multiple_choice(
            "<think>I considered A, B, and C.</think>\nThe answer is D."
        )
        self.assertEqual(parsed["parsed_choice"], "D")
        self.assertEqual(parsed["choice_candidates"], ["D"])

    def test_prefers_bare_final_line_over_option_letters_in_explanation(self):
        parsed = parse_multiple_choice(
            "Options:\n- A. pole\n- B. flagpole\n- C. cell tower\n- D. antenna\n\nC"
        )
        self.assertEqual(parsed["parse_status"], "parsed")
        self.assertEqual(parsed["parsed_choice"], "C")

    def test_accepts_markdown_explicit_answer_after_option_discussion(self):
        parsed = parse_multiple_choice(
            "A. Tella's\nB. Tella\nC. Tello\nD. Tella\n\nAnswer: **B**"
        )
        self.assertEqual(parsed["parse_status"], "parsed")
        self.assertEqual(parsed["parsed_choice"], "B")

    def test_rejects_empty_missing_and_ambiguous_outputs(self):
        cases = {
            "": "invalid_empty",
            "I cannot determine the answer.": "invalid_no_choice",
            "A or B": "invalid_ambiguous",
            "Option A is plausible, but D is correct.": "invalid_ambiguous",
        }
        for raw, expected_status in cases.items():
            with self.subTest(raw=raw):
                parsed = parse_multiple_choice(raw)
                self.assertIsNone(parsed["parsed_choice"])
                self.assertEqual(parsed["parse_status"], expected_status)

    def test_scores_parsed_invalid_and_unsupported_predictions(self):
        self.assertEqual(score_prediction("Answer: B", "B")["score_status"], "correct")
        self.assertEqual(score_prediction("Answer: C", "B")["score_status"], "incorrect")
        self.assertEqual(score_prediction("A or B", "B")["score_status"], "invalid_prediction")
        unsupported = score_prediction("Answer: B", "B", "short_answer")
        self.assertEqual(unsupported["score_status"], "unsupported")
        self.assertFalse(unsupported["is_correct"])

    def test_rejects_non_letter_ground_truth(self):
        with self.assertRaisesRegex(ValueError, "ground_truth"):
            score_prediction("Answer: A", "red")


class PredictionSummaryTest(unittest.TestCase):
    def make_record(self, sample_id: str, raw: str, answer: str = "A"):
        return build_prediction_record(
            sample_id=sample_id,
            ground_truth=answer,
            raw_prediction=raw,
            metadata={"response_token_count": len(raw)},
        )

    def test_summary_keeps_invalid_prediction_in_denominator(self):
        records = [
            self.make_record("one", "A", "A"),
            self.make_record("two", "B", "A"),
            self.make_record("three", "A or B", "A"),
        ]
        summary = summarize_predictions(records, expected_count=3)
        self.assertEqual(summary["correct"], 1)
        self.assertEqual(summary["incorrect"], 1)
        self.assertEqual(summary["invalid_prediction"], 1)
        self.assertEqual(summary["unsupported"], 0)
        self.assertAlmostEqual(summary["accuracy"], 1 / 3)

    def test_rejects_duplicate_ids_and_wrong_expected_count(self):
        record = self.make_record("duplicate", "A")
        with self.assertRaisesRegex(ValueError, "duplicate sample_id"):
            summarize_predictions([record, record])
        with self.assertRaisesRegex(ValueError, "expected 2"):
            summarize_predictions([record], expected_count=2)

    def test_jsonl_atomic_round_trip(self):
        records = [self.make_record("one", "A")]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            write_jsonl_atomic(path, records)
            self.assertEqual(load_jsonl(path), records)


class EvalRowContractTest(unittest.TestCase):
    def make_row(self):
        return {
            "prompt": [{"role": "user", "content": "<image>\nQuestion?\nA. x\nB. y\nC. z\nD. w"}],
            "images": [{"path": "/data/images/sample.png"}],
            "bbox_images": [{"path": "/data/teacher_images/sample.png"}],
            "reward_model": {"style": "none", "ground_truth": "C"},
            "extra_info": {
                "provenance": {
                    "sample_id": "sample-1",
                    "source_id": "source-1",
                    "split": "eval",
                    "question_type": "multiple_choice",
                }
            },
        }

    def test_extracts_only_student_image_contract(self):
        sample = extract_eval_sample(self.make_row())
        self.assertEqual(sample["sample_id"], "sample-1")
        self.assertEqual(sample["ground_truth"], "C")
        self.assertEqual(sample["image_path"].as_posix(), "/data/images/sample.png")
        self.assertNotIn("bbox_images", sample)
        self.assertNotIn("reward_model", sample)
        self.assertNotIn("<image>", sample["prompt_text"])

    def test_rejects_training_rows(self):
        row = self.make_row()
        row["extra_info"]["provenance"]["split"] = "train"
        with self.assertRaisesRegex(ValueError, "only accepts eval"):
            extract_eval_sample(row)


if __name__ == "__main__":
    unittest.main()
