"""Evidence-gated adjudication for independent question proposals."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from typing import Iterable

from .llm import LLMClient
from .models import (
    DocumentStatus,
    EvidenceSpan,
    ModelDraft,
    NoteCase,
    QuestionItem,
    QuestionResult,
    QuestionType,
)
from .postprocess import postprocess_draft
from .prompts import build_aggregation_request


class EvidenceGatedAggregator:
    """Combine only postprocessed proposals and their existing evidence."""

    def __init__(self, client: LLMClient | None = None) -> None:
        self.client = client

    def aggregate(
        self,
        case: NoteCase,
        item: QuestionItem,
        proposals: Iterable[QuestionResult],
    ) -> QuestionResult:
        proposal_tuple = tuple(proposals)
        if not proposal_tuple:
            return _abstain(case, item, "aggregator_no_proposals")

        selected = proposal_tuple
        selection_confidence: float | None = None
        selection_error: str | None = None
        if self.client is not None:
            payload = _proposal_payload(proposal_tuple)
            request = build_aggregation_request(item.spec, payload)
            try:
                response = self.client.generate(request)
                data = request.decode(response.data)
                selected_ids = data["selected_proposal_ids"]
                allowed = {entry["proposal_id"] for entry in payload}
                if not set(selected_ids).issubset(allowed):
                    selection_error = "aggregator_selected_unknown_proposal"
                elif not selected_ids and item.question_type is not QuestionType.NUMERIC:
                    return _abstain(case, item, "aggregator_abstained")
                else:
                    by_id = {entry["proposal_id"]: proposal for entry, proposal in zip(payload, proposal_tuple)}
                    if item.question_type is not QuestionType.NUMERIC:
                        selected = tuple(by_id[identifier] for identifier in selected_ids)
                    selection_confidence = data["confidence"]
            except ValueError as exc:
                selection_error = (
                    "aggregator_selected_unknown_proposal"
                    if "outside the request proposals" in str(exc)
                    else "aggregator_invalid_response"
                )
            except Exception:
                selection_error = "aggregator_invalid_response"

        result = self._deterministic(case, item, selected)
        errors = (*result.validation_errors, *((selection_error,) if selection_error else ()))
        return replace(
            result,
            confidence=selection_confidence if selection_confidence is not None else result.confidence,
            validation_errors=_unique(errors),
        )

    @staticmethod
    def _deterministic(
        case: NoteCase,
        item: QuestionItem,
        proposals: tuple[QuestionResult, ...],
    ) -> QuestionResult:
        if item.question_type is QuestionType.NUMERIC:
            candidates = _unique_spans(
                span for proposal in proposals for span in (proposal.candidate_values or proposal.evidence)
            )
            evidence = _unique_spans(span for proposal in proposals for span in proposal.evidence)
            draft = ModelDraft(
                status=DocumentStatus.VALUE_AVAILABLE,
                evidence=evidence,
                candidate_values=candidates,
                confidence=_mean_confidence(proposals),
                provenance=_provenance(proposals, "aggregated"),
                validation_errors=_proposal_errors(proposals),
            )
            return postprocess_draft(case, item, draft, "full")

        eligible = tuple(
            proposal
            for proposal in proposals
            if proposal.document_status in {DocumentStatus.YES, DocumentStatus.NO, DocumentStatus.NOT_DOCUMENTED}
        )
        counts = Counter(proposal.document_status for proposal in eligible)
        if not counts:
            status = DocumentStatus.INDETERMINATE
            winners: tuple[QuestionResult, ...] = ()
        else:
            highest = max(counts.values())
            winning_statuses = tuple(status for status, count in counts.items() if count == highest)
            status = winning_statuses[0] if len(winning_statuses) == 1 else DocumentStatus.INDETERMINATE
            winners = tuple(proposal for proposal in eligible if proposal.document_status is status)
        evidence = _unique_spans(span for proposal in winners for span in proposal.evidence)
        draft = ModelDraft(
            status=status,
            evidence=evidence,
            confidence=_mean_confidence(winners),
            provenance=_provenance(proposals, "aggregated"),
            validation_errors=_proposal_errors(proposals),
        )
        return postprocess_draft(case, item, draft, "full")


def _proposal_payload(proposals: tuple[QuestionResult, ...]) -> tuple[dict[str, object], ...]:
    evidence_ids: dict[EvidenceSpan, str] = {}
    payload: list[dict[str, object]] = []
    for proposal_index, proposal in enumerate(proposals, 1):
        evidence = []
        for span in (*proposal.evidence, *proposal.candidate_values):
            identifier = evidence_ids.setdefault(span, f"evidence-{len(evidence_ids) + 1}")
            evidence.append(
                {
                    "evidence_id": identifier,
                    "quote": span.quote,
                    "raw_value": span.raw_value,
                    "normalized_value": span.normalized_value,
                    "unit": span.unit,
                }
            )
        payload.append(
            {
                "proposal_id": f"proposal-{proposal_index}",
                "document_status": proposal.document_status.value,
                "answer": proposal.answer,
                "confidence": proposal.confidence,
                "evidence": evidence,
            }
        )
    return tuple(payload)


def _abstain(case: NoteCase, item: QuestionItem, error: str) -> QuestionResult:
    return QuestionResult(
        note_id=case.note_id,
        criterion=item.criterion,
        question_type=item.question_type,
        document_status=DocumentStatus.INDETERMINATE,
        validation_errors=(error,),
    )


def _unique_spans(spans: Iterable[EvidenceSpan]) -> tuple[EvidenceSpan, ...]:
    return tuple(dict.fromkeys(spans))


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _proposal_errors(proposals: Iterable[QuestionResult]) -> tuple[str, ...]:
    return _unique(error for proposal in proposals for error in proposal.validation_errors)


def _provenance(proposals: Iterable[QuestionResult], marker: str) -> tuple[str, ...]:
    return _unique((*[entry for proposal in proposals for entry in proposal.provenance], marker))


def _mean_confidence(proposals: Iterable[QuestionResult]) -> float | None:
    values = tuple(proposal.confidence for proposal in proposals if proposal.confidence is not None)
    return sum(values) / len(values) if values else None
