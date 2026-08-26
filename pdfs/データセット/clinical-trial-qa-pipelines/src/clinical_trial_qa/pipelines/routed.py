"""Routed P4, P5, and P6 implementations with mandatory full-note audit."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from ..aggregation import EvidenceGatedAggregator
from ..llm import LLMClient
from ..models import NoteCase, QuestionItem, QuestionResult
from ..sections import RecallFirstRouter, RoutedContext
from ..verification import RolePanel
from .base import Pipeline, independent_clients, primary_result, sanitized_failure_result, verified_once


class P4Pipeline(Pipeline):
    def __init__(self, client: LLMClient, router: Any | None = None, verifier: Any | None = None) -> None:
        if verifier is None:
            raise ValueError("P4 requires a full-document verifier")
        self.client = client
        self.router = router or RecallFirstRouter()
        self.verifier = verifier

    def _answer_question(self, case: NoteCase, item: QuestionItem) -> QuestionResult:
        context, route_errors = _routed_context(self.router, case, item)
        try:
            candidate = _with_errors(primary_result(self.client, case, item, context, "routed"), route_errors)
        except Exception as exc:
            candidate = _candidate_failure(case, item, exc)
        return verified_once(self.verifier, case, item, candidate)


class P5Pipeline(Pipeline):
    def __init__(
        self,
        clients: Sequence[LLMClient],
        router: Any | None = None,
        verifier: Any | None = None,
        aggregator: EvidenceGatedAggregator | None = None,
    ) -> None:
        client_tuple = independent_clients(clients, "P5")
        if verifier is None:
            raise ValueError("P5 requires a full-document verifier")
        self.clients = client_tuple
        self.router = router or RecallFirstRouter()
        self.verifier = verifier
        self.aggregator = aggregator or EvidenceGatedAggregator()

    def _answer_question(self, case: NoteCase, item: QuestionItem) -> QuestionResult:
        context, route_errors = _routed_context(self.router, case, item)
        try:
            proposals = tuple(primary_result(client, case, item, context, "routed") for client in self.clients)
            candidate = _with_errors(self.aggregator.aggregate(case, item, proposals), route_errors)
        except Exception as exc:
            candidate = _candidate_failure(case, item, exc)
        return verified_once(self.verifier, case, item, candidate)


class P6Pipeline(Pipeline):
    def __init__(
        self,
        role_client: LLMClient | None,
        router: Any | None = None,
        verifier: Any | None = None,
        *,
        max_rounds: int = 2,
        aggregator: EvidenceGatedAggregator | None = None,
    ) -> None:
        if role_client is None:
            raise ValueError("P6 requires one role client")
        if verifier is None:
            raise ValueError("P6 requires a full-document verifier")
        self.role_client = role_client
        self.router = router or RecallFirstRouter()
        self.verifier = verifier
        self.panel = RolePanel(max_rounds=max_rounds, aggregator=aggregator, source_scope="routed")

    def _answer_question(self, case: NoteCase, item: QuestionItem) -> QuestionResult:
        context, route_errors = _routed_context(self.router, case, item)
        try:
            candidate = _with_errors(self.panel.answer(case, item, context, self.role_client), route_errors)
        except Exception as exc:
            candidate = _candidate_failure(case, item, exc)
        return verified_once(self.verifier, case, item, candidate)


def _routed_context(
    router: Any,
    case: NoteCase,
    item: QuestionItem,
) -> tuple[RoutedContext, tuple[str, ...]]:
    try:
        return router.route(case, item.spec), ()
    except Exception:
        return RoutedContext(case.text, (), used_full_text_fallback=True), ("router_failed_full_text_fallback",)


def _with_errors(result: QuestionResult, errors: tuple[str, ...]) -> QuestionResult:
    if not errors:
        return result
    return replace(result, validation_errors=tuple(dict.fromkeys((*result.validation_errors, *errors))))


def _candidate_failure(case: NoteCase, item: QuestionItem, exc: Exception) -> QuestionResult:
    return sanitized_failure_result(
        case,
        item,
        f"candidate_build_failed:{exc.__class__.__name__}",
        "candidate:failed",
    )
