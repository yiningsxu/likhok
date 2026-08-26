"""Evidence-gated conversion of model drafts into shared result objects."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .evidence import EvidenceLedger, EvidenceValidator
from .models import (
    DocumentStatus,
    EvidenceSpan,
    ModelDraft,
    NoteCase,
    QuestionItem,
    QuestionResult,
    QuestionType,
)
from .numeric import is_complete_numeric_token, reduce_candidates


def postprocess_draft(
    case: NoteCase,
    item: QuestionItem,
    draft: ModelDraft,
    source_scope: str,
    source_context: Any | None = None,
) -> QuestionResult:
    """Accept only exact source quotations and recompute numeric outcomes."""
    validator = EvidenceValidator()
    errors = list(draft.validation_errors)
    evidence = _validate_spans(
        case.text,
        draft.evidence,
        source_scope,
        source_context,
        validator,
        errors,
        "invalid_evidence_quote",
    )
    ledger = EvidenceLedger.from_spans(evidence)

    if item.question_type is QuestionType.NUMERIC:
        candidate_source = draft.candidate_values or draft.evidence
        candidates = _validate_numeric_candidates(
            case.text,
            candidate_source,
            source_scope,
            source_context,
            validator,
            errors,
        )
        decision = reduce_candidates(item.spec, candidates, _has_related_mention(case.text, item))
        ledger = EvidenceLedger.from_spans((*ledger.spans, *decision.candidates))
        return _result(
            case,
            item,
            document_status=decision.status,
            answer=decision.answer,
            unit=decision.unit,
            evidence=tuple(span for span in decision.evidence if ledger.contains(span)),
            candidate_values=decision.candidates,
            draft=draft,
            errors=errors,
        )

    status = _as_document_status(draft.status)
    if status in {DocumentStatus.YES, DocumentStatus.NO} and not evidence:
        errors.append("definitive_boolean_without_valid_evidence")
        status = DocumentStatus.INDETERMINATE
    elif status not in {DocumentStatus.YES, DocumentStatus.NO, DocumentStatus.NOT_DOCUMENTED, DocumentStatus.INDETERMINATE}:
        errors.append("invalid_boolean_status")
        status = DocumentStatus.INDETERMINATE
    answer = status.value if status in {DocumentStatus.YES, DocumentStatus.NO} else None
    return _result(
        case,
        item,
        document_status=status,
        answer=answer,
        unit=None,
        evidence=tuple(span for span in evidence if ledger.contains(span)),
        candidate_values=(),
        draft=draft,
        errors=errors,
    )


def _validate_spans(
    text: str,
    spans: tuple[EvidenceSpan, ...],
    source_scope: str,
    source_context: Any | None,
    validator: EvidenceValidator,
    errors: list[str],
    error_code: str,
) -> tuple[EvidenceSpan, ...]:
    validated: list[EvidenceSpan] = []
    for span in spans:
        fixed = _validate_source_span(text, span, source_scope, source_context, validator)
        if fixed is None:
            errors.append(error_code)
            continue
        validated.append(replace(fixed, source_scope=source_scope))
    return tuple(validated)


def _validate_numeric_candidates(
    text: str,
    spans: tuple[EvidenceSpan, ...],
    source_scope: str,
    source_context: Any | None,
    validator: EvidenceValidator,
    errors: list[str],
) -> tuple[EvidenceSpan, ...]:
    candidates = _validate_spans(
        text,
        spans,
        source_scope,
        source_context,
        validator,
        errors,
        "invalid_numeric_candidate_quote",
    )
    valid: list[EvidenceSpan] = []
    for candidate in candidates:
        if candidate.raw_value is None or not is_complete_numeric_token(candidate.quote, candidate.raw_value):
            errors.append("numeric_raw_value_not_in_quote")
            continue
        valid.append(candidate)
    return tuple(valid)


def _validate_source_span(
    note_text: str,
    span: EvidenceSpan,
    source_scope: str,
    source_context: Any | None,
    validator: EvidenceValidator,
) -> EvidenceSpan | None:
    if source_scope != "routed" or source_context is None:
        return validator.validate(note_text, span)
    if getattr(source_context, "used_full_text_fallback", False):
        return validator.validate(note_text, span)

    routed_text = getattr(source_context, "text", None)
    sections = getattr(source_context, "sections", None)
    if not isinstance(routed_text, str) or not isinstance(sections, tuple):
        return None
    if routed_text != "\n\n".join(section.text for section in sections):
        return None

    routed_span = validator.validate(routed_text, span)
    if routed_span is None or routed_span.start_char is None or routed_span.end_char is None:
        return None
    routed_cursor = 0
    for section in sections:
        routed_end = routed_cursor + len(section.text)
        if routed_cursor <= routed_span.start_char and routed_span.end_char <= routed_end:
            full_start = section.start_char + routed_span.start_char - routed_cursor
            full_end = full_start + len(routed_span.quote)
            if full_end <= section.end_char and note_text[full_start:full_end] == routed_span.quote:
                return replace(
                    routed_span,
                    start_char=full_start,
                    end_char=full_end,
                    section_id=section.section_id,
                )
            return None
        routed_cursor = routed_end + 2
    return None


def _has_related_mention(text: str, item: QuestionItem) -> bool:
    folded_text = text.casefold()
    return any(alias.casefold() in folded_text for alias in item.spec.aliases)


def _as_document_status(value: str | DocumentStatus) -> DocumentStatus | None:
    if isinstance(value, DocumentStatus):
        return value
    normalized = value.strip().casefold().replace(" ", "_").replace("-", "_")
    aliases = {"na": "not_documented", "n/a": "not_documented"}
    try:
        return DocumentStatus(aliases.get(normalized, normalized))
    except ValueError:
        return None


def _result(
    case: NoteCase,
    item: QuestionItem,
    *,
    document_status: DocumentStatus,
    answer: str | float | None,
    unit: str | None,
    evidence: tuple[EvidenceSpan, ...],
    candidate_values: tuple[EvidenceSpan, ...],
    draft: ModelDraft,
    errors: list[str],
) -> QuestionResult:
    return QuestionResult(
        note_id=case.note_id,
        criterion=item.criterion,
        question_type=item.question_type,
        document_status=document_status,
        answer=answer,
        unit=unit,
        evidence=evidence,
        candidate_values=candidate_values,
        inference=draft.inference,
        confidence=draft.confidence,
        provenance=draft.provenance,
        validation_errors=tuple(errors),
    )
