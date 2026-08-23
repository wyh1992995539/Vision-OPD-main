"""Deterministic internal multiple-choice scoring for the frozen project eval set."""

from __future__ import annotations

import json
import math
import os
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


CHOICES = frozenset("ABCD")
SUPPORTED_QUESTION_TYPE = "multiple_choice"
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_ANSWER_BLOCK_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
_EXPLICIT_ANSWER_RE = re.compile(
    r"(?:final\s+)?answer\s*(?:is|:|=)\s*"
    r"(?:\*\*|__)?\(?([A-D])\)?(?:\*\*|__)?"
    r"(?=\s|[.,;:!?]|$)",
    re.IGNORECASE,
)
_FINAL_CHOICE_LINE_RE = re.compile(
    r"^\s*(?:\*\*|__)?\(?([A-D])\)?(?:\*\*|__)?[.)]?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_FINAL_LABELED_CHOICE_RE = re.compile(
    r"^\s*(?:\*\*|__)?\(?([A-D])\)?[.)]\s+.+?(?:\*\*|__)?\s*$",
    re.IGNORECASE,
)
_CONCLUSION_CUE_RE = re.compile(
    r"\b(?:therefore|thus|correct answer|best answer|most accurate answer|"
    r"most accurate classification)\b",
    re.IGNORECASE,
)
_STANDALONE_CHOICE_RE = re.compile(r"(?<![A-Za-z0-9])([A-D])(?![A-Za-z0-9])")


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return unicodedata.normalize("NFKC", str(value)).strip()


def remove_thinking_trace(raw_prediction: Any) -> str:
    """Remove completed thinking blocks before deterministic answer parsing."""

    text = _normalize_text(raw_prediction)
    if not text:
        return ""
    return _THINK_BLOCK_RE.sub(" ", text).strip()


def parse_multiple_choice(raw_prediction: Any) -> dict[str, Any]:
    """Extract one unambiguous A-D choice from the visible final response.

    Repeated mentions of the same option are accepted. Mentions of different
    options are rejected rather than guessed.
    """

    visible_text = remove_thinking_trace(raw_prediction)
    if not visible_text:
        return {
            "parsed_choice": None,
            "parse_status": "invalid_empty",
            "visible_text": visible_text,
            "choice_candidates": [],
        }

    answer_blocks = _ANSWER_BLOCK_RE.findall(visible_text)
    if answer_blocks:
        candidate_text = "\n".join(answer_blocks)
        candidates = [
            match.upper() for match in _STANDALONE_CHOICE_RE.findall(candidate_text)
        ]
    else:
        # Prefer an explicit final-answer declaration over option letters quoted
        # while explaining the alternatives (for example, "Answer: **B**").
        candidates = [match.upper() for match in _EXPLICIT_ANSWER_RE.findall(visible_text)]
        if not candidates:
            # Accept a labeled final line such as "D. silver" only when the
            # immediately preceding conclusion contains an explicit cue.  This
            # avoids treating the last row of an ordinary A-D option list as the
            # answer.
            nonempty_lines = [line for line in visible_text.splitlines() if line.strip()]
            labeled_final = (
                _FINAL_LABELED_CHOICE_RE.fullmatch(nonempty_lines[-1])
                if nonempty_lines
                else None
            )
            preceding_tail = "\n".join(nonempty_lines[:-1])[-800:]
            if labeled_final and _CONCLUSION_CUE_RE.search(preceding_tail):
                candidates = [labeled_final.group(1).upper()]

        if not candidates:
            # Models commonly finish a chain of explanation with a bare choice on
            # its own line.  Use the last such line and ignore option labels above.
            final_lines = [
                match.upper() for match in _FINAL_CHOICE_LINE_RE.findall(visible_text)
            ]
            if final_lines:
                candidates = [final_lines[-1]]
            else:
                candidates = [
                    match.upper()
                    for match in _STANDALONE_CHOICE_RE.findall(visible_text)
                ]
    unique_candidates = sorted(set(candidates))

    if not unique_candidates:
        status = "invalid_no_choice"
        parsed_choice = None
    elif len(unique_candidates) > 1:
        status = "invalid_ambiguous"
        parsed_choice = None
    else:
        status = "parsed"
        parsed_choice = unique_candidates[0]

    return {
        "parsed_choice": parsed_choice,
        "parse_status": status,
        "visible_text": visible_text,
        "choice_candidates": unique_candidates,
    }


def score_prediction(
    raw_prediction: Any,
    ground_truth: Any,
    question_type: str = SUPPORTED_QUESTION_TYPE,
) -> dict[str, Any]:
    """Score one prediction without an LLM judge or heuristic ground-truth rewrite."""

    normalized_ground_truth = _normalize_text(ground_truth).upper()
    if normalized_ground_truth not in CHOICES:
        raise ValueError(f"ground_truth must be one of A/B/C/D, got {ground_truth!r}")

    normalized_question_type = _normalize_text(question_type) or "unknown"
    if normalized_question_type != SUPPORTED_QUESTION_TYPE:
        return {
            "ground_truth": normalized_ground_truth,
            "parsed_prediction": None,
            "parse_status": "unsupported_question_type",
            "score_status": "unsupported",
            "is_correct": False,
            "visible_prediction": remove_thinking_trace(raw_prediction),
            "choice_candidates": [],
        }

    parsed = parse_multiple_choice(raw_prediction)
    parsed_choice = parsed["parsed_choice"]
    if parsed_choice is None:
        score_status = "invalid_prediction"
        is_correct = False
    else:
        is_correct = parsed_choice == normalized_ground_truth
        score_status = "correct" if is_correct else "incorrect"

    return {
        "ground_truth": normalized_ground_truth,
        "parsed_prediction": parsed_choice,
        "parse_status": parsed["parse_status"],
        "score_status": score_status,
        "is_correct": is_correct,
        "visible_prediction": parsed["visible_text"],
        "choice_candidates": parsed["choice_candidates"],
    }


def build_prediction_record(
    *,
    sample_id: str,
    ground_truth: Any,
    raw_prediction: Any,
    question_type: str = SUPPORTED_QUESTION_TYPE,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_sample_id = _normalize_text(sample_id)
    if not normalized_sample_id:
        raise ValueError("sample_id must be non-empty")

    score = score_prediction(raw_prediction, ground_truth, question_type)
    record = {
        "sample_id": normalized_sample_id,
        "question_type": _normalize_text(question_type) or "unknown",
        "raw_prediction": "" if raw_prediction is None else str(raw_prediction),
        **score,
    }
    if metadata:
        overlap = set(record).intersection(metadata)
        if overlap:
            raise ValueError(f"metadata cannot overwrite scoring fields: {sorted(overlap)}")
        record.update(metadata)
    return record


def _safe_accuracy(correct: int, total: int) -> float:
    return correct / total if total else 0.0


def summarize_predictions(
    records: Iterable[dict[str, Any]], expected_count: int | None = None
) -> dict[str, Any]:
    materialized = list(records)
    if expected_count is not None and len(materialized) != expected_count:
        raise ValueError(f"expected {expected_count} predictions, found {len(materialized)}")

    seen_ids: set[str] = set()
    status_counts: dict[str, int] = defaultdict(int)
    label_stats: dict[str, dict[str, int]] = {
        choice: {"total": 0, "correct": 0} for choice in sorted(CHOICES)
    }
    response_lengths: list[int] = []

    for index, record in enumerate(materialized):
        sample_id = _normalize_text(record.get("sample_id"))
        if not sample_id:
            raise ValueError(f"prediction at index {index} has no sample_id")
        if sample_id in seen_ids:
            raise ValueError(f"duplicate sample_id: {sample_id}")
        seen_ids.add(sample_id)

        status = _normalize_text(record.get("score_status"))
        if status not in {"correct", "incorrect", "invalid_prediction", "unsupported"}:
            raise ValueError(f"{sample_id}: invalid score_status {status!r}")
        status_counts[status] += 1

        ground_truth = _normalize_text(record.get("ground_truth")).upper()
        if ground_truth not in CHOICES:
            raise ValueError(f"{sample_id}: invalid ground_truth {ground_truth!r}")
        label_stats[ground_truth]["total"] += 1
        if status == "correct":
            label_stats[ground_truth]["correct"] += 1

        length = record.get("response_token_count")
        if isinstance(length, int) and length >= 0:
            response_lengths.append(length)

    total = len(materialized)
    correct = status_counts["correct"]
    unsupported = status_counts["unsupported"]
    supported_total = total - unsupported
    by_ground_truth = {
        label: {
            **stats,
            "accuracy": _safe_accuracy(stats["correct"], stats["total"]),
        }
        for label, stats in label_stats.items()
    }
    response_summary: dict[str, float | int | None] = {
        "count": len(response_lengths),
        "min": min(response_lengths) if response_lengths else None,
        "max": max(response_lengths) if response_lengths else None,
        "mean": (
            math.fsum(response_lengths) / len(response_lengths) if response_lengths else None
        ),
    }

    return {
        "schema_version": 1,
        "total": total,
        "unique_sample_ids": len(seen_ids),
        "correct": correct,
        "incorrect": status_counts["incorrect"],
        "invalid_prediction": status_counts["invalid_prediction"],
        "unsupported": unsupported,
        "accuracy": _safe_accuracy(correct, total),
        "supported_accuracy": _safe_accuracy(correct, supported_total),
        "by_ground_truth": by_ground_truth,
        "response_token_count": response_summary,
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(value)
    return records


def write_jsonl_atomic(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
