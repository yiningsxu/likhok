"""Shared pipeline interface and note-safe per-question isolation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from ..llm import LLMClient
from ..models import DocumentStatus, NoteCase, QuestionItem, QuestionResult
from ..postprocess import postprocess_draft
from ..prompts import build_answer_request
from ..verification import model_draft_from_data


class Pipeline(ABC):
    """Run a pipeline against every question without cross-question failure propagation."""

    def run_case(self, case: NoteCase) -> list[QuestionResult]:
        results: list[QuestionResult] = []
        for item in case.questions:
            try:
                result = self._answer_question(case, item)
            except Exception as exc:
                result = failure_result(case, item, f"pipeline_question_failed:{exc.__class__.__name__}")
            results.append(result)
        return results

    @abstractmethod
    def _answer_question(self, case: NoteCase, item: QuestionItem) -> QuestionResult:
        """Return one result or raise for ``run_case`` to isolate safely."""


def primary_result(
    client: LLMClient,
    case: NoteCase,
    item: QuestionItem,
    context: Any,
    source_scope: str,
) -> QuestionResult:
    """Make one structured primary call and ground its output."""
    try:
        request = build_answer_request(case, item.spec, context)
        response = client.generate(request)
        data = request.decode(response.data)
        draft = model_draft_from_data(data, provenance=(f"primary:{source_scope}",))
    except Exception:
        return failure_result(case, item, "primary_invalid_response")
    return postprocess_draft(case, item, draft, source_scope, context)


def verified_once(
    verifier: Any,
    case: NoteCase,
    item: QuestionItem,
    candidate: QuestionResult,
) -> QuestionResult:
    """Cross the verifier boundary exactly once and redact any thrown message."""
    try:
        return verifier.verify(case, item, candidate)
    except Exception as exc:
        return sanitized_failure_result(
            case,
            item,
            f"verifier_invalid_response:{exc.__class__.__name__}",
            "verifier:failed",
        )


def failure_result(case: NoteCase, item: QuestionItem, error: str) -> QuestionResult:
    return QuestionResult(
        note_id=case.note_id,
        criterion=item.criterion,
        question_type=item.question_type,
        document_status=DocumentStatus.INDETERMINATE,
        validation_errors=(error,),
    )


def sanitized_failure_result(
    case: NoteCase,
    item: QuestionItem,
    error: str,
    provenance: str,
) -> QuestionResult:
    """Return an indeterminate result containing no candidate-owned text."""
    return QuestionResult(
        note_id=case.note_id,
        criterion=item.criterion,
        question_type=item.question_type,
        document_status=DocumentStatus.INDETERMINATE,
        confidence=0.0,
        provenance=(provenance,),
        validation_errors=(error,),
    )


def independent_clients(clients: Sequence[LLMClient], pipeline_name: str) -> tuple[LLMClient, ...]:
    """Require multiple client instances without repeated stateful identities."""
    client_tuple = tuple(clients)
    if len(client_tuple) < 2:
        raise ValueError(f"{pipeline_name} requires at least two primary clients")
    if len({id(client) for client in client_tuple}) != len(client_tuple):
        raise ValueError(f"{pipeline_name} requires distinct primary client instances")
    return client_tuple
