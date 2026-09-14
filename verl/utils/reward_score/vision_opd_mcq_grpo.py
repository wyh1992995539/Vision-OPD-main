"""Deterministic multiple-choice reward for the Vision-OPD GRPO branch.

This module deliberately implements outcome-only rule scoring.  It does not
call a learned reward model or the benchmark Base Judge.  Dataset contract
errors raise immediately; ambiguous or malformed model responses receive a
zero reward with an auditable parse reason.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any


DATA_SOURCE = "vision_opd_mcq_grpo_v1"
REWARD_ROUTE = "vision_opd_mcq_v1"
DEFAULT_VALID_OPTIONS = ("A", "B", "C", "D")

_ANSWER_TAG = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
_ANSWER_TAG_TOKEN = re.compile(r"</?answer\b", re.IGNORECASE)
_OPTION_WRAPPED = re.compile(
    r"""
    ^\s*
    (?:\${1,2}\s*)?
    (?:\\boxed\s*\{\s*)?
    [\(\[]?\s*
    [\*_]*([A-Z])[\*_]*
    \s*[\)\]]?
    \s*\}?
    \s*(?:\${1,2})?
    \s*[\.\!\?。！？]?
    \s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)
_MARKER = r"""
    (?:
        \b(?:
            final\s+answer|correct\s+answer|the\s+answer|answer
            |final\s+option|correct\s+option|selected\s+option|option
        )\b
        |
        (?:最终答案|正确答案|答案|最终选项|正确选项|选择)
    )
"""
_MARKED_OPTION = re.compile(
    rf"""
    {_MARKER}
    \s*(?:(?:is|是|为)\s*)?
    (?::|：|-)?\s*
    (?:option\s*)?
    (?:\${{1,2}}\s*)?
    (?:\\boxed\s*\{{\s*)?
    [\(\[]?\s*([A-Z])
    (?=\s*(?:[\)\]\}}]|\$|\*|_)*(?:$|[\s\.\,;:!\?，。；：！？/]))
    """,
    re.IGNORECASE | re.VERBOSE,
)
_IMMEDIATE_ALTERNATIVE = re.compile(
    r"^\s*(?:/|、|,\s*|\b(?:or|and)\b|或|和|还是)\s*[\(\[]?\s*([A-Z])"
    r"(?=$|[\s\)\]\.,;:!?，。；：！？])",
    re.IGNORECASE,
)


def _normalize_valid_options(value: Any) -> tuple[str, ...]:
    if value is None:
        return DEFAULT_VALID_OPTIONS
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise ValueError("extra_info.valid_options must be an iterable of option labels")
    options = tuple(str(item).strip().upper() for item in value)
    if options != DEFAULT_VALID_OPTIONS:
        raise ValueError(
            f"extra_info.valid_options must be exactly {list(DEFAULT_VALID_OPTIONS)}, got {list(options)}"
        )
    return options


def _normalize_gold(value: Any, valid_options: tuple[str, ...]) -> str:
    if not isinstance(value, str):
        raise ValueError("ground_truth must be a string option label")
    gold = value.strip().upper()
    if len(gold) != 1 or gold not in valid_options:
        raise ValueError(f"ground_truth must be one of {list(valid_options)}, got {value!r}")
    return gold


def _option_from_complete_fragment(fragment: str) -> str | None:
    match = _OPTION_WRAPPED.fullmatch(fragment)
    return match.group(1).upper() if match else None


def _unique_in_order(values: list[str]) -> list[str]:
    unique: list[str] = []
    for value in values:
        normalized = value.upper()
        if normalized not in unique:
            unique.append(normalized)
    return unique


def extract_final_option(
    solution_str: Any, *, valid_options: Iterable[str] = DEFAULT_VALID_OPTIONS
) -> dict[str, Any]:
    """Conservatively extract one explicit final option from a model response."""

    options = _normalize_valid_options(valid_options)
    text = solution_str.strip() if isinstance(solution_str, str) else ""
    if not text:
        return {
            "prediction": None,
            "parse_valid": False,
            "status": "empty_response",
            "evidence": "none",
            "candidate_options": [],
        }

    candidates: list[str] = []
    evidence: list[str] = []
    malformed_tag = False
    tagged_bodies = _ANSWER_TAG.findall(text)
    tag_tokens = _ANSWER_TAG_TOKEN.findall(text)
    if tag_tokens:
        if len(tag_tokens) != 2 * len(tagged_bodies):
            malformed_tag = True
        for body in tagged_bodies:
            candidate = _option_from_complete_fragment(body)
            if candidate is None:
                malformed_tag = True
            else:
                candidates.append(candidate)
                evidence.append("answer_tag")

    for match in _MARKED_OPTION.finditer(text):
        candidates.append(match.group(1).upper())
        evidence.append("explicit_answer_marker")
        alternative = _IMMEDIATE_ALTERNATIVE.match(text[match.end() :])
        if alternative:
            candidates.append(alternative.group(1).upper())
            evidence.append("immediate_alternative")

    if malformed_tag:
        return {
            "prediction": None,
            "parse_valid": False,
            "status": "malformed_answer_tag",
            "evidence": "answer_tag",
            "candidate_options": _unique_in_order(candidates),
        }

    if not candidates:
        bare = _option_from_complete_fragment(text)
        if bare is not None:
            candidates.append(bare)
            evidence.append("bare_option")
        else:
            nonempty_lines = [line for line in text.splitlines() if line.strip()]
            if nonempty_lines:
                trailing = _option_from_complete_fragment(nonempty_lines[-1])
                if trailing is not None:
                    candidates.append(trailing)
                    evidence.append("trailing_option_line")

    unique = _unique_in_order(candidates)
    evidence_value = "+".join(dict.fromkeys(evidence)) if evidence else "none"
    if not unique:
        return {
            "prediction": None,
            "parse_valid": False,
            "status": "no_explicit_final_answer",
            "evidence": evidence_value,
            "candidate_options": [],
        }
    if len(unique) > 1:
        return {
            "prediction": None,
            "parse_valid": False,
            "status": "conflicting_final_answers",
            "evidence": evidence_value,
            "candidate_options": unique,
        }

    prediction = unique[0]
    if prediction not in options:
        return {
            "prediction": None,
            "parse_valid": False,
            "status": "option_out_of_range",
            "evidence": evidence_value,
            "candidate_options": unique,
        }
    return {
        "prediction": prediction,
        "parse_valid": True,
        "status": "parsed",
        "evidence": evidence_value,
        "candidate_options": unique,
    }


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Mapping[str, Any] | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Return a binary correctness reward plus parse diagnostics."""

    if data_source != DATA_SOURCE:
        raise ValueError(f"data_source must be {DATA_SOURCE!r}, got {data_source!r}")
    if not isinstance(extra_info, Mapping):
        raise ValueError("extra_info must be a mapping")
    if extra_info.get("reward_route") != REWARD_ROUTE:
        raise ValueError(
            f"extra_info.reward_route must be {REWARD_ROUTE!r}, "
            f"got {extra_info.get('reward_route')!r}"
        )
    valid_options = _normalize_valid_options(extra_info.get("valid_options"))
    gold = _normalize_gold(ground_truth, valid_options)
    truncated = extra_info.get("truncated", False)
    if not isinstance(truncated, bool):
        raise ValueError("extra_info.truncated must be a boolean when present")

    parsed = extract_final_option(solution_str, valid_options=valid_options)
    prediction = parsed["prediction"]
    correct = bool(parsed["parse_valid"] and prediction == gold)
    return {
        "score": float(correct),
        "acc": float(correct),
        "prediction": prediction or "",
        "ground_truth": gold,
        "parse_valid": bool(parsed["parse_valid"]),
        "parse_status": parsed["status"],
        "parse_evidence": parsed["evidence"],
        "candidate_options": parsed["candidate_options"],
        "truncated": truncated,
        "reward_route": REWARD_ROUTE,
        "reward_source": "deterministic_mcq_rule",
    }


__all__ = [
    "DATA_SOURCE",
    "DEFAULT_VALID_OPTIONS",
    "REWARD_ROUTE",
    "compute_score",
    "extract_final_option",
]
