"""Immutable domain objects shared by every pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class DocumentStatus(str, Enum):
    YES = "yes"
    NO = "no"
    NOT_DOCUMENTED = "not_documented"
    INDETERMINATE = "indeterminate"
    VALUE_AVAILABLE = "value_available"


class QuestionType(str, Enum):
    BOOLEAN = "yes"
    NUMERIC = "numeric"


class Reducer(str, Enum):
    MIN = "min"
    MAX = "max"


@dataclass(frozen=True)
class EvidenceSpan:
    quote: str
    start_char: int | None = None
    end_char: int | None = None
    section_id: str | None = None
    source_scope: str | None = None
    raw_value: str | None = None
    normalized_value: float | None = None
    unit: str | None = None
    time_text: str | None = None


@dataclass(frozen=True)
class QuestionSpec:
    criterion: str
    question_type: QuestionType
    question: str
    aliases: tuple[str, ...] = ()
    expected_unit: str | None = None
    labels: tuple[str, ...] = ("other",)
    time_condition: str | None = None
    subject_condition: str | None = None
    reducer: Reducer | None = None
    output_floor_or_cap: float | None = None


@dataclass(frozen=True)
class QuestionItem:
    spec: QuestionSpec
    answer: str | float | None = None
    not_specified: bool = False
    question: str | None = None

    @property
    def criterion(self) -> str:
        return self.spec.criterion

    @property
    def question_type(self) -> QuestionType:
        return self.spec.question_type

    @property
    def prompt(self) -> str:
        return self.question or self.spec.question


@dataclass(frozen=True)
class NoteCase:
    note_id: str
    hadm_id: str
    text: str
    questions: tuple[QuestionItem, ...]


@dataclass(frozen=True)
class ModelDraft:
    status: str | DocumentStatus = DocumentStatus.INDETERMINATE
    answer: str | float | None = None
    unit: str | None = None
    evidence: tuple[EvidenceSpan, ...] = ()
    candidate_values: tuple[EvidenceSpan, ...] = ()
    inference: str | None = None
    confidence: float | None = None
    provenance: tuple[str, ...] = ()
    validation_errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class QuestionResult:
    note_id: str
    criterion: str
    question_type: QuestionType
    document_status: DocumentStatus
    answer: str | float | None = None
    unit: str | None = None
    evidence: tuple[EvidenceSpan, ...] = ()
    candidate_values: tuple[EvidenceSpan, ...] = ()
    inference: str | None = None
    confidence: float | None = None
    provenance: tuple[str, ...] = ()
    validation_errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready result without a route to the clinical note text."""
        return _json_primitives(asdict(self))


def _json_primitives(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _json_primitives(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_primitives(item) for item in value]
    return value
