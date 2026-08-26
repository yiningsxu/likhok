"""Bounded error-role adjudication and full-document verification."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from .aggregation import EvidenceGatedAggregator
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
from .prompts import build_role_request, build_role_review_request, build_verification_request


_BOOLEAN_ROLES = ("assertion/negation/experiencer", "temporality", "evidence fidelity")
_NUMERIC_ROLES = ("numeric completeness", "temporality", "evidence fidelity")


class RolePanel:
    """Generate three independent error-role proposals and at most two reviews."""

    def __init__(
        self,
        max_rounds: int = 2,
        aggregator: EvidenceGatedAggregator | None = None,
        *,
        source_scope: str | None = None,
    ) -> None:
        if not 0 <= max_rounds <= 2:
            raise ValueError("max_rounds must be between 0 and 2")
        if source_scope not in {None, "full", "routed"}:
            raise ValueError("source_scope must be full or routed")
        self.max_rounds = max_rounds
        self.aggregator = aggregator or EvidenceGatedAggregator()
        self.source_scope = source_scope

    def answer(
        self,
        case: NoteCase,
        item: QuestionItem,
        context: Any,
        client: LLMClient,
    ) -> QuestionResult:
        roles = _NUMERIC_ROLES if item.question_type is QuestionType.NUMERIC else _BOOLEAN_ROLES
        scope = self.source_scope or _context_scope(case, context)
        proposals: list[QuestionResult] = []
        for role in roles:
            request = build_role_request(case, item.spec, role, context)
            proposals.append(_generate_proposal(client, request, case, item, scope, context, f"role:{role}"))

        current = self.aggregator.aggregate(case, item, proposals)
        for round_index in range(self.max_rounds):
            role = roles[round_index % len(roles)]
            request = build_role_review_request(case, item.spec, role, context, current)
            proposals.append(
                _generate_proposal(
                    client,
                    request,
                    case,
                    item,
                    scope,
                    context,
                    f"role_review:{round_index + 1}:{role}",
                )
            )
            current = self.aggregator.aggregate(case, item, proposals)
        return current


class FullDocumentVerifier:
    """Audit one candidate once and permit revisions only through its evidence IDs."""

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    def verify(
        self,
        case: NoteCase,
        item: QuestionItem,
        candidate: QuestionResult,
    ) -> QuestionResult:
        request = build_verification_request(case, item.spec, candidate)
        try:
            response = self.client.generate(request)
            data = request.decode(response.data)
        except ValueError as exc:
            code = (
                "verifier_revision_not_grounded"
                if "outside the candidate ledger" in str(exc)
                else "verifier_invalid_response"
            )
            return _abstain(case, item, candidate, code)
        except Exception:
            return _abstain(case, item, candidate, "verifier_invalid_response")

        grounded_candidate = postprocess_draft(case, item, _draft_from_result(candidate), "full")
        if data["approved"]:
            return replace(
                grounded_candidate,
                provenance=_unique((*grounded_candidate.provenance, "verifier:approved")),
            )

        if item.question_type is QuestionType.NUMERIC:
            return _abstain(case, item, grounded_candidate, "verifier_revision_not_grounded")

        revision = data["result"]
        if _contains_free_text_spans(revision):
            return _abstain(case, item, grounded_candidate, "verifier_revision_not_grounded")
        ledger = _candidate_ledger(candidate)
        selected_ids = revision.get("selected_evidence_ids", [])
        if not isinstance(selected_ids, list) or not all(identifier in ledger for identifier in selected_ids):
            return _abstain(case, item, grounded_candidate, "verifier_revision_not_grounded")
        status = revision.get("document_status")
        if not isinstance(status, str):
            return _abstain(case, item, grounded_candidate, "verifier_invalid_response")
        selected = tuple(ledger[identifier] for identifier in selected_ids)
        draft = ModelDraft(
            status=status,
            answer=revision.get("answer"),
            unit=revision.get("unit") if isinstance(revision.get("unit"), (str, type(None))) else None,
            evidence=selected,
            candidate_values=selected if item.question_type is QuestionType.NUMERIC else (),
            inference=revision.get("inference") if isinstance(revision.get("inference"), (str, type(None))) else None,
            confidence=_confidence(revision.get("confidence")),
            provenance=_unique((*candidate.provenance, "verifier:revised")),
            validation_errors=candidate.validation_errors,
        )
        return postprocess_draft(case, item, draft, "full")


def model_draft_from_data(data: Mapping[str, Any], *, provenance: tuple[str, ...] = ()) -> ModelDraft:
    """Convert decoded answer JSON into immutable untrusted model data."""
    return ModelDraft(
        status=data["document_status"],
        answer=data["answer"],
        unit=data["unit"],
        evidence=tuple(_span_from_data(value) for value in data["evidence"]),
        candidate_values=tuple(_span_from_data(value) for value in data["candidate_values"]),
        inference=data["inference"],
        confidence=_confidence(data["confidence"]),
        provenance=provenance,
    )


def _generate_proposal(
    client: LLMClient,
    request: Any,
    case: NoteCase,
    item: QuestionItem,
    source_scope: str,
    source_context: Any,
    provenance: str,
) -> QuestionResult:
    try:
        response = client.generate(request)
        data = request.decode(response.data)
        draft = model_draft_from_data(data, provenance=(provenance,))
    except Exception:
        draft = ModelDraft(
            validation_errors=("role_invalid_response",),
            provenance=(provenance,),
        )
    return postprocess_draft(case, item, draft, source_scope, source_context)


def _span_from_data(data: Mapping[str, Any]) -> EvidenceSpan:
    fields = {
        "quote",
        "start_char",
        "end_char",
        "section_id",
        "source_scope",
        "raw_value",
        "normalized_value",
        "unit",
        "time_text",
    }
    return EvidenceSpan(**{name: data[name] for name in fields if name in data})


def _draft_from_result(result: QuestionResult) -> ModelDraft:
    return ModelDraft(
        status=result.document_status,
        answer=result.answer,
        unit=result.unit,
        evidence=result.evidence,
        candidate_values=result.candidate_values,
        inference=result.inference,
        confidence=result.confidence,
        provenance=result.provenance,
        validation_errors=result.validation_errors,
    )


def _candidate_ledger(candidate: QuestionResult) -> dict[str, EvidenceSpan]:
    spans = tuple(dict.fromkeys((*candidate.evidence, *candidate.candidate_values)))
    return {f"evidence-{index}": span for index, span in enumerate(spans, 1)}


def _contains_free_text_spans(revision: Mapping[str, Any]) -> bool:
    return any(isinstance(revision.get(field), list) and bool(revision[field]) for field in ("evidence", "candidate_values"))


def _context_scope(case: NoteCase, context: Any) -> str:
    if isinstance(context, str):
        return "full" if context == case.text else "routed"
    return "routed"


def _confidence(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if 0 <= value <= 1 else None


def _unique(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _abstain(
    case: NoteCase,
    item: QuestionItem,
    candidate: QuestionResult,
    error: str,
) -> QuestionResult:
    return QuestionResult(
        note_id=case.note_id,
        criterion=item.criterion,
        question_type=item.question_type,
        document_status=DocumentStatus.INDETERMINATE,
        candidate_values=candidate.candidate_values if item.question_type is QuestionType.NUMERIC else (),
        provenance=_unique((*candidate.provenance, "verifier:abstained")),
        validation_errors=_unique((*candidate.validation_errors, error)),
    )
