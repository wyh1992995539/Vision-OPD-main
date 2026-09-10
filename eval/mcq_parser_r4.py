#!/usr/bin/env python3
"""Conservative R4 multiple-choice parser.

R3 searched for any capital letter in the answer. That made the A in
"Answer: D" a valid prediction. R4 only accepts explicit answer markers or
an answer that consists of a bare option. Conflicting or unparseable answers
remain ambiguous so the frozen Judge can decide them.
"""

from __future__ import annotations

import re
from typing import Any

from eval.paper_aligned_common import extract_mcq_option


_OPTION = r"[A-F]"
_OPTION_OPEN = r"(?:\${0,2}\s*)?(?:\\boxed\s*\{\s*)?[\s\*\_\(\[]*"
_OPTION_END = r"(?=$|[\s\*\_\)\]\}\$\.,:;!\?\-])"

_ANSWER_TAG = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
_TAG_OPTION = re.compile(
    rf"^\s*{_OPTION_OPEN}({_OPTION}){_OPTION_END}", re.IGNORECASE
)
_EXPLICIT_MARKER = re.compile(
    rf"(?:\b(?:final\s+answer|correct\s+answer|the\s+answer|answer|"
    rf"final\s+option|correct\s+option|selected\s+option)\b|"
    rf"(?:最终答案|正确答案|答案|最终选项|正确选项|选择))"
    rf"\s*(?:(?:is|是|为)\s*)?(?::|：)?\s*(?:option\s*)?"
    rf"{_OPTION_OPEN}({_OPTION}){_OPTION_END}",
    re.IGNORECASE,
)
_BARE_OPTION = re.compile(
    rf"^\s*{_OPTION_OPEN}({_OPTION})"
    rf"(?:[\*\_\)\]\}}\$]*)\s*(?:[\.,:;!\?\-].*)?$",
    re.IGNORECASE | re.DOTALL,
)
_TRAILING_OPTION = re.compile(
    rf"(?:^|\n)\s*{_OPTION_OPEN}({_OPTION})"
    rf"(?:[\*\_\)\]\}}\$]*)\s*$",
    re.IGNORECASE,
)


def _unique_in_order(values: list[str]) -> list[str]:
    unique: list[str] = []
    for value in values:
        normalized = value.upper()
        if normalized not in unique:
            unique.append(normalized)
    return unique


def classify_mcq_answer(reference: Any, raw_answer: Any) -> dict[str, Any]:
    """Return a tri-state R4 decision: match, mismatch, or ambiguous.

    A mismatch is safe to count incorrect without a Judge because both the
    reference and prediction contain one explicit option and they differ.
    An ambiguous answer must use the same frozen Judge used by R3.
    """

    expected = extract_mcq_option(reference)
    text = str(raw_answer or "").strip()
    candidates: list[str] = []
    evidence = "none"

    tagged = _ANSWER_TAG.findall(text)
    if tagged:
        evidence = "answer_tag"
        for body in tagged:
            match = _TAG_OPTION.match(body)
            if match:
                candidates.append(match.group(1))
    else:
        marked = [match.group(1) for match in _EXPLICIT_MARKER.finditer(text)]
        if marked:
            evidence = "explicit_answer_marker"
            candidates.extend(marked)
        else:
            bare = _BARE_OPTION.match(text)
            if bare:
                evidence = "bare_option"
                candidates.append(bare.group(1))
            else:
                trailing = _TRAILING_OPTION.search(text)
                if trailing:
                    evidence = "trailing_option_line"
                    candidates.append(trailing.group(1))

    unique = _unique_in_order(candidates)
    predicted = unique[0] if len(unique) == 1 else None
    if not expected or predicted is None:
        status = "ambiguous"
    elif predicted == expected:
        status = "match"
    else:
        status = "mismatch"

    return {
        "status": status,
        "expected_option": expected or None,
        "predicted_option": predicted,
        "candidate_options": unique,
        "evidence": evidence,
    }
