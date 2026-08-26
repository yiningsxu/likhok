"""Recall-first section splitting and multi-label note routing."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

from .llm import LLMClient
from .models import NoteCase, QuestionSpec
from .prompts import build_section_label_request


_HEADING = re.compile(r"^[A-Za-z][A-Za-z0-9 /&()_-]{1,}:\s*$")
_LABEL_KEYWORDS = {
    "laboratory": ("lab", "laboratory", "cbc", "chemistry", "hematology", "microbiology"),
    "history": ("history", "past medical", "pmh", "hpi"),
    "neurology": ("neurology", "neurologic", "neuro", "stroke", "tia"),
    "cardiology": ("cardiology", "cardiac", "heart", "afib", "atrial fibrillation"),
    "psychiatry": ("psychiatry", "psychiatric", "mental health", "bipolar", "depression"),
    "diagnosis": ("diagnosis", "assessment", "problem list", "impression"),
    "procedure": ("procedure", "operative", "surgery", "ablation", "catheterization"),
    "bleeding": ("bleeding", "hemorrhage", "hematoma", "gi bleed"),
    "medication": ("medication", "medications", "meds", "prescription"),
    "capacity": ("capacity", "decision making", "incapacitated"),
}


@dataclass(frozen=True)
class Section:
    section_id: str
    text: str
    start_char: int
    end_char: int
    labels: tuple[str, ...] = ()
    deterministic_labels: tuple[str, ...] = ()
    client_labels: tuple[str, ...] = ()


@dataclass(frozen=True)
class RoutedContext:
    text: str
    sections: tuple[Section, ...]
    used_full_text_fallback: bool = False


class SectionSplitter:
    """Split plain-text notes at heading lines while retaining full-note offsets."""

    def split(self, text: str) -> tuple[Section, ...]:
        if not text:
            return ()
        heading_offsets = [offset for offset, line in _lines_with_offsets(text) if _HEADING.match(line.strip())]
        if not heading_offsets:
            return (self._make_section(text, 0, len(text), 1),)
        starts = ([0] if text[:heading_offsets[0]].strip() else []) + heading_offsets
        sections = tuple(
            self._make_section(text, start, starts[index + 1] if index + 1 < len(starts) else len(text), index + 1)
            for index, start in enumerate(starts)
        )
        return tuple(section for section in sections if section.text.strip())

    def _make_section(self, full_text: str, start: int, end: int, index: int) -> Section:
        section_text = full_text[start:end]
        labels = _heuristic_labels(section_text)
        heading = section_text.splitlines()[0].rstrip(":").strip().casefold() if section_text else "section"
        section_id = f"{_slug(heading) or 'section'}-{index}"
        return Section(section_id, section_text, start, end, labels, labels)


class RecallFirstRouter:
    """Prefer matching labelled sections, but retain the whole note on every miss."""

    def __init__(self, top_k: int = 3, label_client: LLMClient | None = None, splitter: SectionSplitter | None = None) -> None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        self.top_k = top_k
        self.label_client = label_client
        self.splitter = splitter or SectionSplitter()

    def route(self, case: NoteCase, spec: QuestionSpec) -> RoutedContext:
        sections = self.splitter.split(case.text)
        labelled = tuple(self._add_optional_labels(section) for section in sections)
        desired = set(spec.labels)
        ranked = [(self._score(section, desired, spec.aliases), section) for section in labelled]
        selected = [section for score, section in ranked if score > 0]
        selected.sort(key=lambda section: (-self._score(section, desired, spec.aliases), section.start_char))
        selected = selected[:self.top_k]
        if not selected:
            return RoutedContext(case.text, sections, used_full_text_fallback=True)
        return RoutedContext("\n\n".join(section.text for section in selected), tuple(selected))

    def _add_optional_labels(self, section: Section) -> Section:
        if self.label_client is None:
            return section
        try:
            response = self.label_client.generate(build_section_label_request(section.text, _LABEL_KEYWORDS))
            extra_labels = response.data.get("labels")
        except Exception:
            return section
        if not isinstance(extra_labels, list) or not all(isinstance(label, str) and label.strip() for label in extra_labels):
            return section
        return Section(
            section.section_id,
            section.text,
            section.start_char,
            section.end_char,
            tuple(sorted(set(section.labels) | {label.strip().casefold() for label in extra_labels})),
            section.deterministic_labels,
            tuple(sorted(set(section.client_labels) | {label.strip().casefold() for label in extra_labels})),
        )

    @staticmethod
    def _score(section: Section, desired: set[str], aliases: Iterable[str]) -> int:
        label_score = len(set(section.deterministic_labels) & desired)
        body = section.text.casefold()
        alias_score = sum(1 for alias in aliases if _has_positive_alias(body, alias.casefold()))
        return label_score + alias_score


def _lines_with_offsets(text: str) -> Iterable[tuple[int, str]]:
    offset = 0
    for line in text.splitlines(keepends=True):
        yield offset, line
        offset += len(line)
    if text and not text.endswith(("\n", "\r")) and not list(text.splitlines(keepends=True)):
        yield 0, text


def _heuristic_labels(text: str) -> tuple[str, ...]:
    normalized = text.casefold()
    labels = [label for label, keywords in _LABEL_KEYWORDS.items() if any(keyword in normalized for keyword in keywords)]
    return tuple(labels or ["other"])


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")


def _has_positive_alias(text: str, alias: str) -> bool:
    """Avoid treating explicit no-value statements as positive routing evidence."""
    for match in re.finditer(re.escape(alias), text):
        prefix = text[max(0, match.start() - 16):match.start()]
        if re.search(r"\b(?:no|without|denies|denied)\s+$", prefix):
            continue
        return True
    return False
