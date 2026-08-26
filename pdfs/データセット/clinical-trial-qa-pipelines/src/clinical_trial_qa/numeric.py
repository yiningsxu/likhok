"""Deterministic, unit-safe reduction of validated numeric evidence."""

from __future__ import annotations

from dataclasses import dataclass, replace
import re
from typing import Iterable

from .models import DocumentStatus, EvidenceSpan, QuestionSpec, Reducer


_NUMERIC_BODY = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_NUMBER = re.compile(
    rf"^\s*(?P<comparator><=|>=|<|>)?\s*(?P<number>{_NUMERIC_BODY})\s*(?P<percent>%?)\s*$"
)
_QUOTE_NUMBER = re.compile(
    rf"(?<![A-Za-z0-9_.,])(?P<token>(?:<=|>=|<|>)?\s*{_NUMERIC_BODY}\s*%?)(?![A-Za-z0-9_.,])"
)
_UNIT_SUFFIX = re.compile(
    r"^\s*(?P<unit>(?:[A-Za-zµμ]+|[x×]?\s*10\s*\^?\s*\d+)\s*/\s*[A-Za-zµμ]+)(?![A-Za-zµμ])"
)


@dataclass(frozen=True)
class NumericDecision:
    status: DocumentStatus
    answer: float | None
    unit: str | None
    candidates: tuple[EvidenceSpan, ...]
    evidence: tuple[EvidenceSpan, ...] = ()


def parse_numeric(raw: str) -> float | None:
    """Parse a single model-extracted numeric token without changing its unit."""
    match = _NUMBER.fullmatch(raw)
    if match is None:
        return None
    return float(match.group("number").replace(",", ""))


def is_complete_numeric_token(quote: str, raw: str) -> bool:
    """Return whether ``raw`` canonically equals a complete numeric token in ``quote``."""
    canonical_raw = _canonical_numeric_token(raw)
    return canonical_raw is not None and any(
        _canonical_numeric_token(match.group("token")) == canonical_raw for match in _QUOTE_NUMBER.finditer(quote)
    )


def reduce_candidates(
    spec: QuestionSpec,
    candidates: Iterable[EvidenceSpan],
    related_mention_present: bool,
) -> NumericDecision:
    """Recompute a numeric answer from every validated candidate, never a model answer."""
    normalized_candidates = tuple(_normalize_candidate(candidate) for candidate in candidates)
    usable = tuple(candidate for candidate in normalized_candidates if candidate.normalized_value is not None)
    if not usable:
        status = DocumentStatus.INDETERMINATE if related_mention_present else DocumentStatus.NOT_DOCUMENTED
        return NumericDecision(status=status, answer=None, unit=None, candidates=normalized_candidates)

    grounded_units = tuple(_grounded_unit(candidate) for candidate in usable)
    if any(not valid for valid, _unit in grounded_units):
        return NumericDecision(
            status=DocumentStatus.INDETERMINATE,
            answer=None,
            unit=None,
            candidates=normalized_candidates,
        )
    comparable_units = {_normalized_unit(unit) for _valid, unit in grounded_units}
    if len(comparable_units) > 1:
        return NumericDecision(
            status=DocumentStatus.INDETERMINATE,
            answer=None,
            unit=None,
            candidates=normalized_candidates,
        )

    if spec.reducer is Reducer.MIN:
        chosen_value = min(candidate.normalized_value for candidate in usable)
    elif spec.reducer is Reducer.MAX:
        chosen_value = max(candidate.normalized_value for candidate in usable)
    else:
        return NumericDecision(
            status=DocumentStatus.INDETERMINATE,
            answer=None,
            unit=None,
            candidates=normalized_candidates,
        )

    answer = chosen_value
    if spec.output_floor_or_cap is not None:
        answer = min(answer, spec.output_floor_or_cap)

    evidence = tuple(candidate for candidate in usable if candidate.normalized_value == chosen_value)
    unit = next((unit for _valid, unit in grounded_units if unit is not None), None)
    return NumericDecision(
        status=DocumentStatus.VALUE_AVAILABLE,
        answer=answer,
        unit=unit,
        candidates=normalized_candidates,
        evidence=evidence,
    )


def _normalize_candidate(candidate: EvidenceSpan) -> EvidenceSpan:
    raw_value = candidate.raw_value
    if raw_value is None or not is_complete_numeric_token(candidate.quote, raw_value):
        return replace(candidate, normalized_value=None)
    return replace(candidate, normalized_value=parse_numeric(raw_value))


def _grounded_unit(candidate: EvidenceSpan) -> tuple[bool, str | None]:
    """Validate model unit metadata against the unit visible beside the exact value."""
    raw_value = candidate.raw_value
    if raw_value is None:
        return False, None
    canonical_raw = _canonical_numeric_token(raw_value)
    if canonical_raw is None:
        return False, None

    observed: list[str | None] = []
    for match in _QUOTE_NUMBER.finditer(candidate.quote):
        if _canonical_numeric_token(match.group("token")) != canonical_raw:
            continue
        if canonical_raw[2]:
            observed.append("%")
            continue
        suffix = _UNIT_SUFFIX.match(candidate.quote[match.end() :])
        observed.append(suffix.group("unit") if suffix is not None else None)

    observed_keys = {_normalized_unit(unit) for unit in observed}
    if len(observed_keys) != 1:
        return False, None
    visible_unit = observed[0]
    model_unit = _normalized_unit(candidate.unit)
    visible_key = _normalized_unit(visible_unit)
    if visible_key is None:
        return model_unit is None, None
    return model_unit == visible_key, visible_unit


def _normalized_unit(unit: str | None) -> str | None:
    if unit is None or not unit.strip():
        return None
    return "".join(unit.casefold().replace("μ", "u").replace("µ", "u").split())


def _canonical_numeric_token(value: str) -> tuple[str, str, bool] | None:
    match = _NUMBER.fullmatch(value)
    if match is None:
        return None
    return match.group("comparator") or "", match.group("number"), bool(match.group("percent"))
