from clinical_trial_qa.aggregation import EvidenceGatedAggregator
from clinical_trial_qa.llm import ScriptedLLMClient
from clinical_trial_qa.models import (
    DocumentStatus,
    EvidenceSpan,
    NoteCase,
    QuestionItem,
    QuestionResult,
    QuestionType,
)
from clinical_trial_qa.questions import get_question_spec


def _case(
    text: str = "Bipolar disorder documented. No bipolar disorder documented. PLT 143 K/uL; PLT 91 K/uL",
) -> NoteCase:
    return NoteCase("note-1", "hadm-1", text, ())


def _item(criterion: str = "bipolar") -> QuestionItem:
    return QuestionItem(get_question_spec(criterion))


def _boolean(status: DocumentStatus, quote: str | None = None) -> QuestionResult:
    evidence = (EvidenceSpan(quote=quote),) if quote else ()
    return QuestionResult(
        note_id="note-1",
        criterion="bipolar",
        question_type=QuestionType.BOOLEAN,
        document_status=status,
        answer=status.value if status in {DocumentStatus.YES, DocumentStatus.NO} else None,
        evidence=evidence,
    )


def _numeric(quote: str, raw_value: str) -> QuestionResult:
    span = EvidenceSpan(quote=quote, raw_value=raw_value, unit="K/uL")
    return QuestionResult(
        note_id="note-1",
        criterion="PLT",
        question_type=QuestionType.NUMERIC,
        document_status=DocumentStatus.VALUE_AVAILABLE,
        answer=float(raw_value),
        unit="K/uL",
        evidence=(span,),
        candidate_values=(span,),
    )


def test_equal_boolean_votes_abstain():
    """Would fail if a tie were resolved as an unsupported positive or negative answer."""
    result = EvidenceGatedAggregator().aggregate(
        _case(),
        _item(),
        (
            _boolean(DocumentStatus.YES, "Bipolar disorder documented."),
            _boolean(DocumentStatus.NO, "No bipolar disorder documented."),
        ),
    )

    assert result.document_status is DocumentStatus.INDETERMINATE
    assert result.answer is None


def test_boolean_majority_uses_only_grounded_evidence_from_winning_proposals():
    """Would fail if a majority answer could retain a quote absent from the full note."""
    result = EvidenceGatedAggregator().aggregate(
        _case("Bipolar disorder documented. No bipolar disorder documented."),
        _item(),
        (
            _boolean(DocumentStatus.YES, "Bipolar disorder documented."),
            _boolean(DocumentStatus.YES, "invented quote"),
            _boolean(DocumentStatus.NO, "No bipolar disorder documented."),
        ),
    )

    assert result.document_status is DocumentStatus.YES
    assert tuple(span.quote for span in result.evidence) == ("Bipolar disorder documented.",)


def test_numeric_ensemble_reduces_union_of_valid_candidates():
    """Would fail if numeric aggregation trusted one proposal rather than the full candidate union."""
    result = EvidenceGatedAggregator().aggregate(
        _case(), _item("PLT"), (_numeric("PLT 143 K/uL", "143"), _numeric("PLT 91 K/uL", "91"))
    )

    assert result.answer == 91.0
    assert {candidate.normalized_value for candidate in result.candidate_values} == {91.0, 143.0}


def test_llm_selection_cannot_narrow_the_numeric_candidate_union():
    """Would fail if adjudicator preference could hide a valid numeric candidate."""
    client = ScriptedLLMClient([{"selected_proposal_ids": ["proposal-1"], "confidence": 0.8}])

    result = EvidenceGatedAggregator(client).aggregate(
        _case(), _item("PLT"), (_numeric("PLT 143 K/uL", "143"), _numeric("PLT 91 K/uL", "91"))
    )

    assert result.answer == 91.0
    assert {candidate.normalized_value for candidate in result.candidate_values} == {91.0, 143.0}


def test_llm_abstention_cannot_suppress_a_deterministic_numeric_reduction():
    """Would fail if an empty adjudicator selection overrode validated numeric evidence."""
    client = ScriptedLLMClient([{"selected_proposal_ids": [], "confidence": 0.2}])

    result = EvidenceGatedAggregator(client).aggregate(
        _case(), _item("PLT"), (_numeric("PLT 143 K/uL", "143"), _numeric("PLT 91 K/uL", "91"))
    )

    assert result.document_status is DocumentStatus.VALUE_AVAILABLE
    assert result.answer == 91.0


def test_llm_aggregator_can_select_only_a_provided_proposal():
    """Would fail if a valid opaque proposal selection were ignored."""
    client = ScriptedLLMClient([{"selected_proposal_ids": ["proposal-2"], "confidence": 0.7}])
    result = EvidenceGatedAggregator(client).aggregate(
        _case(),
        _item(),
        (
            _boolean(DocumentStatus.YES, "Bipolar disorder documented."),
            _boolean(DocumentStatus.NO, "No bipolar disorder documented."),
        ),
    )

    assert result.document_status is DocumentStatus.NO
    assert result.confidence == 0.7


def test_llm_aggregator_cannot_select_evidence_outside_ledger():
    """Would fail if the aggregator could fabricate a proposal identifier."""
    client = ScriptedLLMClient([{"selected_proposal_ids": ["unknown"], "confidence": 1.0}])
    result = EvidenceGatedAggregator(client).aggregate(
        _case(),
        _item(),
        (
            _boolean(DocumentStatus.YES, "Bipolar disorder documented."),
            _boolean(DocumentStatus.NO, "No bipolar disorder documented."),
        ),
    )

    assert result.document_status is DocumentStatus.INDETERMINATE
    assert "aggregator_selected_unknown_proposal" in result.validation_errors


def test_llm_aggregator_rejects_boolean_confidence():
    """Would fail because JSON true is an int subclass unless confidence rejects it explicitly."""
    client = ScriptedLLMClient([{"selected_proposal_ids": ["proposal-1"], "confidence": True}])

    result = EvidenceGatedAggregator(client).aggregate(
        _case(), _item(), (_boolean(DocumentStatus.YES, "Bipolar disorder documented."),)
    )

    assert result.confidence is None
    assert "aggregator_invalid_response" in result.validation_errors


def test_empty_proposal_set_abstains_with_a_safe_error():
    """Would fail if aggregation crashed or fabricated an answer without proposals."""
    result = EvidenceGatedAggregator().aggregate(_case(), _item(), ())

    assert result.document_status is DocumentStatus.INDETERMINATE
    assert result.validation_errors == ("aggregator_no_proposals",)
