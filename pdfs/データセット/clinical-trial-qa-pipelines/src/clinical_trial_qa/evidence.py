"""Exact-match evidence validation shared by every QA pipeline."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

from .models import EvidenceSpan


@dataclass(frozen=True)
class EvidenceLedger:
    """An immutable allow-list of evidence spans validated against a note."""

    spans: tuple[EvidenceSpan, ...] = ()

    @classmethod
    def from_spans(cls, spans: Iterable[EvidenceSpan]) -> "EvidenceLedger":
        return cls(tuple(spans))

    def contains(self, span: EvidenceSpan) -> bool:
        return span in self.spans


class EvidenceValidator:
    """Validate a model quotation and rebuild its offsets from the source text."""

    def validate(self, note_text: str, span: EvidenceSpan) -> EvidenceSpan | None:
        if not span.quote:
            return None

        starts = _all_starts(note_text, span.quote)
        if not starts:
            return None

        start = _select_start(starts, span.start_char, len(note_text))
        return replace(span, start_char=start, end_char=start + len(span.quote))


def _all_starts(text: str, quote: str) -> tuple[int, ...]:
    starts: list[int] = []
    start = text.find(quote)
    while start != -1:
        starts.append(start)
        start = text.find(quote, start + 1)
    return tuple(starts)


def _select_start(starts: tuple[int, ...], supplied_start: int | None, note_length: int) -> int:
    if supplied_start is None or not 0 <= supplied_start < note_length:
        return starts[0]
    return min(starts, key=lambda start: (abs(start - supplied_start), start))
