import base64
import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path

from PIL import Image

from eval.paper_aligned_common import (
    extract_answer_official,
    first_letter_match,
    inference_messages,
    inference_request_kwargs,
    image_data_uri,
    judge_complete,
    judge_request_kwargs,
    load_config,
    normalize_judge_decision,
    prediction_complete,
    read_jsonl_map,
    validate_config,
    write_jsonl_map,
)
from eval.score_paper_aligned import merge_judge, rule_score
from eval.run_paper_aligned_eval import output_directory


def prediction(benchmark: str = "mmstar", answer: str = "B") -> dict:
    return {
        "schema_version": 1,
        "benchmark": benchmark,
        "view": "full",
        "sample_uid": f"{benchmark}:1",
        "source_id": "1",
        "question_format": "multiple_choice",
        "official_category": "category",
        "official_l2_category": "l2",
        "prompt": "Which option?",
        "reference_answer": "B",
        "raw_model_answer": answer,
        "finish_reason": "stop",
        "prompt_tokens": 10,
        "completion_tokens": 2,
        "latency_seconds": 0.1,
        "error": None,
    }


class PaperAlignedProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, cls.config = load_config("configs/benchmark_eval_paper_basejudge_r3_single_gpu.yaml")

    def test_frozen_config_has_exact_request_contract(self):
        validate_config(self.config)
        self.assertIsNone(self.config["prompt_and_image"]["system_prompt"])
        self.assertFalse(self.config["generation"]["enable_thinking"])
        self.assertEqual(self.config["generation"]["temperature"], 0)
        self.assertEqual(self.config["generation"]["max_tokens"], 1024)
        self.assertEqual(self.config["reporting"]["expected_total_visual_requests"], 2536)
        self.assertEqual(self.config["benchmarks"]["vstar"]["primary_summary_denominator"], 191)

    def test_inference_request_has_only_user_message_and_no_forbidden_sampling(self):
        messages = inference_messages("data:image/png;base64,AA==", "<image> Question?")
        request = inference_request_kwargs(
            model_id="target",
            messages=messages,
            config=self.config,
        )
        self.assertEqual([item["role"] for item in request["messages"]], ["user"])
        self.assertEqual(request["temperature"], 0)
        self.assertEqual(request["max_tokens"], 1024)
        self.assertEqual(
            request["extra_body"],
            {"chat_template_kwargs": {"enable_thinking": False}},
        )
        forbidden = set(self.config["generation"]["forbidden_request_parameters"])
        self.assertFalse(forbidden.intersection(request))

    def test_judge_request_has_only_user_message_and_frozen_parameters(self):
        request = judge_request_kwargs(
            model_id="vision-opd-base-judge",
            prompt="Judge this",
            config=self.config,
        )
        self.assertEqual(request["messages"], [{"role": "user", "content": "Judge this"}])
        self.assertEqual(request["temperature"], 0)
        self.assertEqual(request["max_tokens"], 2048)
        self.assertFalse(
            request["extra_body"]["chat_template_kwargs"]["enable_thinking"]
        )

    def test_official_answer_and_first_letter_behavior(self):
        self.assertEqual(extract_answer_official("<answer>B</answer> tail"), "B")
        self.assertEqual(extract_answer_official("Answer: B"), "Answer: B")
        self.assertTrue(first_letter_match("B", "The answer is (B)."))
        self.assertFalse(first_letter_match("B", "I cannot tell."))

    def test_mathruler_then_first_letter_then_judge(self):
        always_false = lambda reference, answer: False
        mmstar = rule_score(prediction("mmstar", "The answer is (B)."), self.config, always_false)
        self.assertEqual(mmstar["rule_source"], "first_letter")
        self.assertTrue(mmstar["final_is_correct"])
        zoom = rule_score(prediction("zoombench", "B"), self.config, always_false)
        self.assertEqual(zoom["rule_source"], "llm_judge_required")
        self.assertTrue(zoom["judge_required"])
        self.assertEqual(zoom["score_status"], "pending_judge")

    def test_inference_failure_is_counted_wrong_without_judge(self):
        item = prediction()
        item["raw_model_answer"] = ""
        item["error"] = "APIError"
        score = rule_score(item, self.config, lambda reference, answer: True)
        self.assertEqual(score["rule_source"], "inference_failure")
        self.assertFalse(score["judge_required"])
        self.assertFalse(score["final_is_correct"])

    def test_finalized_judge_failure_merges_as_incorrect(self):
        score = rule_score(prediction("zoombench", "unknown"), self.config, lambda a, b: False)
        failed = {
            "benchmark": "zoombench",
            "view": "full",
            "sample_uid": "zoombench:1",
            "raw_judge_output": "Maybe",
            "normalized_decision": None,
            "error": "invalid Judge output",
            "finalized": True,
        }
        self.assertTrue(judge_complete(failed))
        merged = merge_judge(score, failed)
        self.assertEqual(merged["score_status"], "scored")
        self.assertEqual(merged["judge_source"], "judge_failure")
        self.assertFalse(merged["final_is_correct"])

    def test_resume_compaction_prefers_successful_record(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            failed = prediction()
            failed["raw_model_answer"] = ""
            failed["error"] = "failed"
            successful = prediction()
            path.write_text(
                json.dumps(failed) + "\n" + "{broken\n" + json.dumps(successful) + "\n",
                encoding="utf-8",
            )
            records, stats = read_jsonl_map(path, complete=prediction_complete)
            self.assertEqual(len(records), 1)
            self.assertEqual(next(iter(records.values()))["raw_model_answer"], "B")
            self.assertEqual(stats["duplicate_keys"], 1)
            self.assertEqual(stats["malformed_lines"], 1)
            write_jsonl_map(path, records)
            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 1)

    def test_judge_normalization_is_exact(self):
        self.assertEqual(normalize_judge_decision(" Yes\n"), "Yes")
        self.assertEqual(normalize_judge_decision("no"), "No")
        self.assertIsNone(normalize_judge_decision("Yes, correct"))


    def test_r3_contract_and_output_directory_are_isolated(self):
        _, r3 = load_config("configs/benchmark_eval_paper_basejudge_r3_single_gpu.yaml")
        self.assertEqual(r3["protocol"]["protocol_revision"], 3)
        self.assertEqual(r3["serving"]["tensor_parallel_size"], 1)
        self.assertEqual(r3["serving"]["gpu_memory_utilization"], 0.75)
        self.assertEqual(
            r3["serving"]["additional_config"]["gdn_prefill_backend"], "triton"
        )
        self.assertEqual(r3["budget"]["gpu_count"], 1)
        self.assertEqual(r3["budget"]["instance_cost_per_wall_hour"], 5.98)
        output = output_directory(r3, "base", None, 4)
        self.assertTrue(str(output).endswith("/smoke_r3/base"))

    def test_r3_vstar_always_encodes_rgb_png(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.jpg"
            Image.new("RGB", (2, 2), color=(10, 20, 30)).save(path, format="JPEG")
            legacy = image_data_uri(path, "vstar", vstar_always_rgb_png=False)
            r3 = image_data_uri(path, "vstar", vstar_always_rgb_png=True)
            self.assertTrue(legacy.startswith("data:image/jpeg;base64,"))
            self.assertTrue(r3.startswith("data:image/png;base64,"))
            payload = base64.b64decode(r3.split(",", 1)[1])
            with Image.open(BytesIO(payload)) as decoded:
                self.assertEqual(decoded.mode, "RGB")
                self.assertEqual(decoded.format, "PNG")


if __name__ == "__main__":
    unittest.main()
