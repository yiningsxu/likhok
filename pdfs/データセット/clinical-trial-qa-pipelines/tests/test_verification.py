from clinical_trial_qa.llm import ScriptedLLMClient
from clinical_trial_qa.models import (
    DocumentStatus,
    EvidenceSpan,
    NoteCase,
    QuestionItem,
    QuestionResult,
    QuestionType,
)
from clinical_trial_qa.prompts import build_role_review_request
from clinical_trial_qa.questions import get_question_spec
from clinical_trial_qa.verification import FullDocumentVerifier, RolePanel


def _case(text: str = "Bipolar disorder documented. Separate stroke history.") -> NoteCase:
    return NoteCase("note-1", "hadm-1", text, ())


def _item(criterion: str = "bipolar") -> QuestionItem:
    return QuestionItem(get_question_spec(criterion))


def _draft(status: str = "not_documented", quote: str | None = None) -> dict[str, object]:
    evidence = [{"quote": quote, "start_char": None, "end_char": None}] if quote else []
    return {
        "document_status": status,
        "answer": status if status in {"yes", "no"} else None,
        "unit": None,
        "evidence": evidence,
        "candidate_values": [],
        "inference": None,
        "confidence": 0.5,
    }


def _candidate() -> QuestionResult:
    return QuestionResult(
        note_id="note-1",
        criterion="bipolar",
        question_type=QuestionType.BOOLEAN,
        document_status=DocumentStatus.YES,
        answer="yes",
        evidence=(EvidenceSpan("Bipolar disorder documented.", 0, 30, source_scope="routed"),),
    )


def _assert_value_error(callable_object) -> None:
    try:
        callable_object()
    except ValueError:
        return
    raise AssertionError("ValueError was not raised")


def test_role_review_request_has_a_distinct_task_name():
    """Would fail if initial role proposals and bounded review rounds were indistinguishable."""
    request = build_role_review_request(_case(), _item().spec, "evidence fidelity", "context", _candidate())

    assert request.task == "role_review"
    assert "Current result to review" in request.messages[-1][1]


def test_role_panel_never_exceeds_two_review_rounds():
    """Would fail if bounded debate made one review call per role or looped until agreement."""
    client = ScriptedLLMClient([_draft()] * 5)

    RolePanel(max_rounds=2).answer(_case(), _item(), _case().text, client)

    assert sum(call.task == "role" for call in client.calls) == 3
    assert sum(call.task == "role_review" for call in client.calls) == 2


def test_role_panel_allows_zero_rounds_and_rejects_out_of_range_values():
    """Would fail if the role budget were not the closed interval from zero through two."""
    client = ScriptedLLMClient([_draft()] * 3)

    RolePanel(max_rounds=0).answer(_case(), _item(), _case().text, client)

    assert sum(call.task == "role_review" for call in client.calls) == 0
    _assert_value_error(lambda: RolePanel(max_rounds=-1))
    _assert_value_error(lambda: RolePanel(max_rounds=3))


def test_role_proposals_are_postprocessed_before_adjudication():
    """Would fail if three agreeing roles could promote hallucinated positive evidence."""
    client = ScriptedLLMClient([_draft("yes", "invented quote")] * 3)

    result = RolePanel(max_rounds=0).answer(_case(), _item(), _case().text, client)

    assert result.document_status is DocumentStatus.INDETERMINATE
    assert "invalid_evidence_quote" in result.validation_errors


def test_verifier_approves_only_after_full_note_postprocessing():
    """Would fail if approval bypassed a final full-note evidence and offset check."""
    verifier = FullDocumentVerifier(ScriptedLLMClient([{"approved": True, "result": {}}]))

    checked = verifier.verify(_case(), _item(), _candidate())

    assert checked.document_status is DocumentStatus.YES
    assert checked.evidence[0].start_char == 0
    assert checked.evidence[0].source_scope == "full"


def test_verifier_rejects_revision_with_invented_quote():
    """Would fail if a verifier revision could add a quote outside the candidate ledger."""
    response = {
        "approved": False,
        "result": {
            **_draft("yes", "not in note"),
            "selected_evidence_ids": [],
        },
    }
    verifier = FullDocumentVerifier(ScriptedLLMClient([response]))

    checked = verifier.verify(_case(), _item(), _candidate())

    assert checked.document_status is DocumentStatus.INDETERMINATE
    assert "verifier_revision_not_grounded" in checked.validation_errors


def test_verifier_rejects_new_quote_even_when_it_exists_in_the_full_note():
    """Would fail if full-note presence alone let a verifier escape its evidence ledger."""
    response = {
        "approved": False,
        "result": {
            **_draft("yes", "Separate stroke history."),
            "selected_evidence_ids": [],
        },
    }
    verifier = FullDocumentVerifier(ScriptedLLMClient([response]))

    checked = verifier.verify(_case(), _item(), _candidate())

    assert checked.document_status is DocumentStatus.INDETERMINATE
    assert "verifier_revision_not_grounded" in checked.validation_errors


def test_verifier_revision_can_select_an_existing_evidence_id_only():
    """Would fail if an opaque in-ledger selection could not survive the full-note gate."""
    response = {
        "approved": False,
        "result": {
            "document_status": "yes",
            "answer": "yes",
            "unit": None,
            "selected_evidence_ids": ["evidence-1"],
            "inference": None,
            "confidence": 0.9,
        },
    }
    verifier = FullDocumentVerifier(ScriptedLLMClient([response]))

    checked = verifier.verify(_case(), _item(), _candidate())

    assert checked.document_status is DocumentStatus.YES
    assert tuple(span.quote for span in checked.evidence) == ("Bipolar disorder documented.",)


def test_numeric_verifier_revision_cannot_narrow_candidates_and_abstention_preserves_them():
    """Would fail if selected evidence IDs could hide a validated numeric candidate."""
    case = _case("PLT 143 K/uL and later PLT 91 K/uL")
    item = _item("PLT")
    high = EvidenceSpan("PLT 143 K/uL", 0, 13, source_scope="routed", raw_value="143", unit="K/uL")
    low = EvidenceSpan("PLT 91 K/uL", 24, 36, source_scope="routed", raw_value="91", unit="K/uL")
    candidate = QuestionResult(
        note_id=case.note_id,
        criterion=item.criterion,
        question_type=QuestionType.NUMERIC,
        document_status=DocumentStatus.VALUE_AVAILABLE,
        answer=91.0,
        unit="K/uL",
        evidence=(low,),
        candidate_values=(high, low),
    )
    response = {
        "approved": False,
        "result": {
            "document_status": "value_available",
            "answer": 91,
            "unit": "K/uL",
            "selected_evidence_ids": ["evidence-1"],
            "inference": None,
            "confidence": 0.9,
        },
    }

    checked = FullDocumentVerifier(ScriptedLLMClient([response])).verify(case, item, candidate)

    assert checked.document_status is DocumentStatus.INDETERMINATE
    assert checked.answer is None
    assert {span.normalized_value for span in checked.candidate_values} == {91.0, 143.0}
    assert "verifier_revision_not_grounded" in checked.validation_errors


def test_verifier_unknown_evidence_selection_abstains():
    """Would fail if an unknown evidence ID were treated as an empty but valid revision."""
    response = {
        "approved": False,
        "result": {
            "document_status": "yes",
            "answer": "yes",
            "unit": None,
            "selected_evidence_ids": ["unknown"],
            "inference": None,
            "confidence": 1.0,
        },
    }
    verifier = FullDocumentVerifier(ScriptedLLMClient([response]))

    checked = verifier.verify(_case(), _item(), _candidate())

    assert checked.document_status is DocumentStatus.INDETERMINATE
    assert "verifier_revision_not_grounded" in checked.validation_errors


def test_verifier_revision_rejects_boolean_confidence():
    """Would fail if JSON true passed the verifier's numeric confidence contract."""
    response = {
        "approved": False,
        "result": {
            "document_status": "yes",
            "answer": "yes",
            "unit": None,
            "selected_evidence_ids": ["evidence-1"],
            "inference": None,
            "confidence": True,
        },
    }

    checked = FullDocumentVerifier(ScriptedLLMClient([response])).verify(_case(), _item(), _candidate())

    assert checked.document_status is DocumentStatus.INDETERMINATE
    assert checked.confidence is None
    assert "verifier_invalid_response" in checked.validation_errors
