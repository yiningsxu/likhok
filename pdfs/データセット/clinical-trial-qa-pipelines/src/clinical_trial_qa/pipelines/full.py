"""Full-document P1, P2, and P3 implementations."""

from __future__ import annotations

from collections.abc import Sequence

from ..aggregation import EvidenceGatedAggregator
from ..llm import LLMClient
from ..models import NoteCase, QuestionItem, QuestionResult
from ..verification import RolePanel
from .base import Pipeline, independent_clients, primary_result


class P1Pipeline(Pipeline):
    def __init__(self, client: LLMClient) -> None:
        self.client = client

    def _answer_question(self, case: NoteCase, item: QuestionItem) -> QuestionResult:
        return primary_result(self.client, case, item, case.text, "full")


class P2Pipeline(Pipeline):
    def __init__(
        self,
        clients: Sequence[LLMClient],
        aggregator: EvidenceGatedAggregator | None = None,
    ) -> None:
        self.clients = independent_clients(clients, "P2")
        self.aggregator = aggregator or EvidenceGatedAggregator()

    def _answer_question(self, case: NoteCase, item: QuestionItem) -> QuestionResult:
        proposals = tuple(primary_result(client, case, item, case.text, "full") for client in self.clients)
        return self.aggregator.aggregate(case, item, proposals)


class P3Pipeline(Pipeline):
    def __init__(
        self,
        role_client: LLMClient | None,
        *,
        max_rounds: int = 2,
        aggregator: EvidenceGatedAggregator | None = None,
    ) -> None:
        if role_client is None:
            raise ValueError("P3 requires one role client")
        self.role_client = role_client
        self.panel = RolePanel(max_rounds=max_rounds, aggregator=aggregator, source_scope="full")

    def _answer_question(self, case: NoteCase, item: QuestionItem) -> QuestionResult:
        return self.panel.answer(case, item, case.text, self.role_client)
