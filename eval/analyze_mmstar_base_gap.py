#!/usr/bin/env python3
"""Build a read-only, auditable diagnosis of the local Base MMStar gap."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent.parent
R3_ROOT = ROOT / "artifacts/runs/E-PAPER-BASEJUDGE-001/base"
R4_ROOT = ROOT / "artifacts/runs/E-PAPER-BASEJUDGE-R4-001/base"
TOKEN_GATE = ROOT / "artifacts/runs/E-MMSTAR-BASE-MAXTOK-DIAG-001/gate_receipt.json"
R3_CONFIG = ROOT / "configs/benchmark_eval_paper_basejudge_r3_single_gpu.yaml"
DEFAULT_OUTPUT = ROOT / "artifacts/reports/mmstar_base_gap_root_cause.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("benchmark") != "mmstar":
                continue
            key = str(item["sample_uid"])
            if key in records:
                raise ValueError(f"duplicate key in {path}: {key}")
            records[key] = item
    return records


def pct(correct: int, total: int) -> float:
    return round(100.0 * correct / total, 6) if total else 0.0


def run_git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, check=True, text=True, capture_output=True
    )
    return result.stdout


def official_snapshot() -> dict[str, Any]:
    commit = run_git("rev-parse", "official/main").strip()
    run_eval = run_git("show", "official/main:eval/run_eval.sh")
    infer = run_git("show", "official/main:eval/infer.py")
    judge = run_git("show", "official/main:eval/judge_qwenlm.py")
    cal_acc = run_git("show", "official/main:eval/cal_acc.py")
    max_tokens = re.search(r'MAX_TOKENS="\$\{MAX_TOKENS:-(\d+)\}"', run_eval)
    infer_default = re.search(r'--max_tokens[^\n]*default=(\d+)', infer)
    return {
        "commit": commit,
        "files": {
            "eval/run_eval.sh": hashlib.sha256(run_eval.encode()).hexdigest(),
            "eval/infer.py": hashlib.sha256(infer.encode()).hexdigest(),
            "eval/judge_qwenlm.py": hashlib.sha256(judge.encode()).hexdigest(),
            "eval/cal_acc.py": hashlib.sha256(cal_acc.encode()).hexdigest(),
        },
        "run_eval_default_max_tokens": int(max_tokens.group(1)) if max_tokens else None,
        "infer_cli_default_max_tokens": int(infer_default.group(1)) if infer_default else None,
        "first_letter_fallback_present": 're.search(r"([A-Z])", text)' in judge,
        "strict_yes_scoring_present": 'str(item.get("judge", "")).strip().lower() == "yes"' in cal_acc,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output

    paths = {
        "r3_predictions": R3_ROOT / "predictions.jsonl",
        "r3_scores": R3_ROOT / "scores.jsonl",
        "r3_judges": R3_ROOT / "judge_results.jsonl",
        "r4_predictions": R4_ROOT / "predictions.jsonl",
        "r4_scores": R4_ROOT / "scores.jsonl",
        "token_gate": TOKEN_GATE,
        "r3_config": R3_CONFIG,
    }
    initial_hashes = {name: sha256(path) for name, path in paths.items()}

    predictions = read_jsonl(paths["r3_predictions"])
    r3_scores = read_jsonl(paths["r3_scores"])
    r4_scores = read_jsonl(paths["r4_scores"])
    r3_judges = read_jsonl(paths["r3_judges"])
    if set(predictions) != set(r3_scores) or set(predictions) != set(r4_scores):
        raise ValueError("R3 prediction/score and R4 score key sets differ")
    if len(predictions) != 1500:
        raise ValueError(f"expected 1500 MMStar records, found {len(predictions)}")
    if sha256(paths["r3_predictions"]) != sha256(paths["r4_predictions"]):
        raise ValueError("R4 predictions are not byte-identical to frozen R3 predictions")

    transitions: Counter[str] = Counter()
    route_cross: Counter[str] = Counter()
    mismatch_r3_sources: Counter[str] = Counter()
    mismatch_r3_judges: Counter[str] = Counter()
    finish_cross: Counter[str] = Counter()
    categories: dict[str, Counter[str]] = defaultdict(Counter)
    examples: list[dict[str, Any]] = []

    for key in sorted(predictions):
        pred = predictions[key]
        old = r3_scores[key]
        new = r4_scores[key]
        old_correct = bool(old["final_is_correct"])
        new_correct = bool(new["final_is_correct"])
        transition = f"r3_{'correct' if old_correct else 'wrong'}__r4_{'correct' if new_correct else 'wrong'}"
        transitions[transition] += 1

        route = str(new.get("mcq_parse_status") or new.get("rule_source"))
        route_cross[f"{route}|r3={old.get('rule_source')}|r3_correct={old_correct}|r4_correct={new_correct}"] += 1

        finish = str(pred.get("finish_reason"))
        finish_cross[f"{finish}|r4_correct={new_correct}|route={route}"] += 1

        category = str(pred.get("official_category") or "unknown")
        cat = categories[category]
        cat["total"] += 1
        cat["correct"] += int(new_correct)
        cat["finish_length"] += int(finish == "length")
        cat["judge_required"] += int(bool(new.get("judge_required")))

        if route == "mismatch":
            mismatch_r3_sources[str(old.get("rule_source"))] += 1
            mismatch_r3_sources[f"{old.get('rule_source')}|correct={old_correct}"] += 1
            decision = str(old.get("judge_normalized_decision"))
            mismatch_r3_judges[f"{decision}|correct={old_correct}"] += 1
            if old_correct and len(examples) < 8:
                examples.append(
                    {
                        "sample_uid": key,
                        "reference": pred.get("reference_answer"),
                        "r4_predicted_option": new.get("mcq_predicted_option"),
                        "r3_rule_source": old.get("rule_source"),
                        "r3_judge_decision": old.get("judge_normalized_decision"),
                        "answer_excerpt": str(pred.get("raw_model_answer") or "")[-240:],
                    }
                )

    token_gate = json.loads(TOKEN_GATE.read_text(encoding="utf-8"))
    config = yaml.safe_load(R3_CONFIG.read_text(encoding="utf-8"))
    official = official_snapshot()
    r3_correct = sum(bool(item["final_is_correct"]) for item in r3_scores.values())
    r4_correct = sum(bool(item["final_is_correct"]) for item in r4_scores.values())
    paper_correct_equivalent = round(0.7853 * 1500)

    category_report = {}
    for category, values in sorted(categories.items()):
        total = values["total"]
        stop_total = total - values["finish_length"]
        stop_correct = sum(
            bool(r4_scores[key]["final_is_correct"])
            for key, pred in predictions.items()
            if pred.get("official_category") == category and pred.get("finish_reason") != "length"
        )
        category_report[category] = {
            "total": total,
            "correct": values["correct"],
            "accuracy_percent": pct(values["correct"], total),
            "finish_length": values["finish_length"],
            "finish_length_percent": pct(values["finish_length"], total),
            "stop_only_accuracy_percent": pct(stop_correct, stop_total),
            "judge_required": values["judge_required"],
        }

    result = {
        "schema_version": 1,
        "scope": "local Base MMStar paper-gap diagnosis; source artifacts are read-only",
        "paper_reference": {
            "reported_accuracy_percent": 78.53,
            "correct_equivalent_rounded": paper_correct_equivalent,
            "note": "The paper reports a percentage, not a released per-sample score file.",
        },
        "local_results": {
            "r3_upstream_like_parser_local_judge": {
                "correct": r3_correct,
                "total": 1500,
                "accuracy_percent": pct(r3_correct, 1500),
                "delta_from_paper_pp": round(pct(r3_correct, 1500) - 78.53, 6),
            },
            "r4_corrected_parser_local_judge": {
                "correct": r4_correct,
                "total": 1500,
                "accuracy_percent": pct(r4_correct, 1500),
                "delta_from_paper_pp": round(pct(r4_correct, 1500) - 78.53, 6),
            },
        },
        "r3_to_r4": {
            "transitions": dict(sorted(transitions.items())),
            "route_cross": dict(sorted(route_cross.items())),
            "explicit_mismatch_r3_sources": dict(sorted(mismatch_r3_sources.items())),
            "explicit_mismatch_r3_judge_decisions": dict(sorted(mismatch_r3_judges.items())),
            "representative_r3_false_positive_examples": examples,
        },
        "finish_reason_cross": dict(sorted(finish_cross.items())),
        "category_diagnostics_r4": category_report,
        "max_token_diagnostic": {
            "baseline_correct": token_gate["baseline"]["r4_correct"],
            "hybrid_2048_4096_correct": token_gate["stages"]["residual_4096"]["counterfactual_full_correct"],
            "hybrid_2048_4096_accuracy_percent": round(
                100 * token_gate["stages"]["residual_4096"]["counterfactual_full_accuracy"], 6
            ),
            "remaining_length_at_4096": token_gate["stages"]["residual_4096"]["residual_length_count"],
            "diagnostic_gate_pass": token_gate["stages"]["residual_4096"]["gate_pass"],
            "decision": token_gate["decision"],
        },
        "protocol_differences": {
            "local_r3_eval_max_tokens": int(config["generation"]["max_tokens"]),
            "official_repository": official,
            "paper_eval_max_tokens_disclosed": False,
            "paper_judge": str(config["judge"]["paper_model"]),
            "local_judge": str(config["judge"]["model"]["name"]),
            "local_runtime_checkpoint_revision": str(config["model_under_test"]["base_huggingface_revision"]),
            "paper_runtime_checkpoint_revision_disclosed": False,
        },
        "findings": [
            {
                "rank": 1,
                "cause": "The corrected R4 parser is not the released upstream parser used by the public evaluation code.",
                "confidence": "confirmed",
                "effect": "R4 removes false positives caused by the upstream first-uppercase fallback; its 68.00% is an internal corrected score, not a like-for-like Table 2 reproduction.",
            },
            {
                "rank": 2,
                "cause": "The local frozen Qwen3.5-4B Judge substitutes for the paper GPT-OSS-120B Judge.",
                "confidence": "confirmed protocol mismatch; exact score effect unmeasured",
                "effect": "Judge decisions are not directly comparable, especially under the released parser where many samples reach the Judge.",
            },
            {
                "rank": 3,
                "cause": "The local evaluation cap is 1024 while the released run_eval.sh default is 32768.",
                "confidence": "confirmed public-command mismatch; actual paper-run override unknown",
                "effect": "Long math/science answers truncate more often. The controlled 2048/4096 R4 diagnostic improves 68.00% to 71.47%, so truncation is material but insufficient by itself.",
            },
            {
                "rank": 4,
                "cause": "Exact Base checkpoint and serving runtime identity cannot be established from the paper.",
                "confidence": "unresolved",
                "effect": "The local revision and hashes are frozen, but the paper does not publish an exact revision, weight hash, serving backend, or all decoding/runtime details.",
            },
            {
                "rank": 5,
                "cause": "Dataset conversion or prompt construction error.",
                "confidence": "low likelihood based on current evidence",
                "effect": "Pinned 1500-sample data, raw question text, answer labels, and one-image user prompts match the released preparation/inference structure; paper hashes are unavailable for byte-level proof.",
            },
        ],
        "recommended_interpretation": {
            "correct_internal_metric": "Keep R4 as the corrected local benchmark.",
            "paper_reproduction_metric": "Run and label a separate upstream-faithful track; do not overwrite R4.",
            "next_high_value_test": "Use the released parser/default generation contract and the paper GPT-OSS-120B Judge (or explicitly label any substitute), then report upstream-like and R4 scores side by side.",
            "gpu_needed_now": False,
        },
        "source_sha256_before": initial_hashes,
    }

    final_hashes = {name: sha256(path) for name, path in paths.items()}
    result["source_unchanged_during_analysis"] = final_hashes == initial_hashes
    result["source_sha256_after"] = final_hashes
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    digest = sha256(output)
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{digest}  {output.name}\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), "sha256": digest, "source_unchanged": final_hashes == initial_hashes}, indent=2))


if __name__ == "__main__":
    main()
