from clinical_trial_qa.models import DocumentStatus, EvidenceSpan, ModelDraft, NoteCase, QuestionItem
from clinical_trial_qa.postprocess import postprocess_draft
from clinical_trial_qa.questions import get_question_spec


def case(text: str) -> NoteCase:
    return NoteCase(note_id="note-1", hadm_id="hadm-1", text=text, questions=())


def bipolar_item() -> QuestionItem:
    return QuestionItem(spec=get_question_spec("bipolar"))


def plt_item() -> QuestionItem:
    return QuestionItem(spec=get_question_spec("PLT"))


def lvef_item() -> QuestionItem:
    return QuestionItem(spec=get_question_spec("lvef"))


def numeric_draft(model_answer: float, quotes: list[tuple[str, str]], unit: str = "K/uL") -> ModelDraft:
    candidates = tuple(EvidenceSpan(quote=quote, raw_value=raw_value, unit=unit) for quote, raw_value in quotes)
    return ModelDraft(status="value_available", answer=model_answer, evidence=candidates, candidate_values=candidates)


def test_boolean_yes_without_valid_quote_becomes_indeterminate():
    """Would fail if a positive clinical assertion survived a hallucinated quote."""
    result = postprocess_draft(
        case("No psychiatric history."),
        bipolar_item(),
        ModelDraft(status="yes", evidence=(EvidenceSpan(quote="Bipolar disorder"),)),
        "full",
    )

    assert result.document_status is DocumentStatus.INDETERMINATE
    assert result.answer is None


def test_boolean_no_without_valid_quote_becomes_indeterminate_with_fixed_error():
    """Would fail if a definitive negative answer could survive without exact evidence."""
    result = postprocess_draft(
        case("No psychiatric history."),
        bipolar_item(),
        ModelDraft(status="no"),
        "full",
    )

    assert result.document_status is DocumentStatus.INDETERMINATE
    assert result.answer is None
    assert result.validation_errors == ("definitive_boolean_without_valid_evidence",)


def test_numeric_result_is_recomputed_not_trusted_from_model():
    """Would fail if the postprocessor returned the model's numeric answer."""
    draft = numeric_draft(model_answer=999, quotes=[("PLT 143 K/uL", "143"), ("PLT 91 K/uL", "91")])

    result = postprocess_draft(case("PLT 143 K/uL; PLT 91 K/uL"), plt_item(), draft, "full")

    assert result.document_status is DocumentStatus.VALUE_AVAILABLE
    assert result.answer == 91.0
    assert len(result.candidate_values) == 2


def test_numeric_extremum_retains_its_normalized_supporting_evidence():
    """Would fail if normalization changed span identity after the evidence ledger was built."""
    draft = numeric_draft(model_answer=143, quotes=[("PLT 143 K/uL", "143"), ("PLT 91 K/uL", "91")])

    result = postprocess_draft(case("PLT 143 K/uL; PLT 91 K/uL"), plt_item(), draft, "full")

    assert result.answer == 91.0
    assert {span.normalized_value for span in result.candidate_values} == {91.0, 143.0}
    assert tuple(span.quote for span in result.evidence) == ("PLT 91 K/uL",)
    assert result.evidence[0].normalized_value == 91.0


def test_postprocessor_rebuilds_offsets_and_records_the_input_scope():
    """Would fail if final evidence retained unverified offsets or lacked its provenance scope."""
    result = postprocess_draft(
        case("History: bipolar disorder."),
        bipolar_item(),
        ModelDraft(status="yes", evidence=(EvidenceSpan(quote="bipolar disorder", start_char=0, end_char=1),)),
        "routed",
    )

    evidence = result.evidence[0]
    assert result.document_status is DocumentStatus.YES
    assert evidence.start_char == 9
    assert evidence.end_char == 25
    assert evidence.source_scope == "routed"


def test_rejected_numeric_quote_cannot_supply_a_value():
    """Would fail if a numeric candidate could bypass exact evidence validation."""
    result = postprocess_draft(
        case("PLT pending"),
        plt_item(),
        numeric_draft(model_answer=91, quotes=[("PLT 91", "91")]),
        "full",
    )

    assert result.document_status is DocumentStatus.INDETERMINATE
    assert result.answer is None
    assert result.candidate_values == ()


def test_partial_numeric_raw_value_cannot_bypass_postprocessing_evidence_gate():
    """Would fail if a digit inside a larger lab value became the final answer."""
    result = postprocess_draft(
        case("PLT 143 K/uL"),
        plt_item(),
        numeric_draft(model_answer=1, quotes=[("PLT 143 K/uL", "1")]),
        "full",
    )

    assert result.document_status is DocumentStatus.INDETERMINATE
    assert result.answer is None
    assert result.candidate_values == ()


def test_postprocessor_accepts_complete_comparator_comma_and_percentage_tokens():
    """Would fail if the safety gate rejected valid complete numeric notation."""
    result = postprocess_draft(
        case("PLT <= 1.5 K/uL; PLT 1,234.50 K/uL"),
        plt_item(),
        numeric_draft(
            model_answer=999,
            quotes=[("PLT <= 1.5 K/uL", "<=1.5"), ("PLT 1,234.50 K/uL", "1,234.50")],
        ),
        "full",
    )
    percentage = postprocess_draft(
        case("EF 60%"),
        lvef_item(),
        numeric_draft(model_answer=999, quotes=[("EF 60%", "60%")], unit="%"),
        "full",
    )

    assert result.document_status is DocumentStatus.VALUE_AVAILABLE
    assert result.answer == 1.5
    assert len(result.candidate_values) == 2
    assert percentage.document_status is DocumentStatus.VALUE_AVAILABLE
    assert percentage.answer == 55.0


def test_spaced_comparator_omission_cannot_bypass_postprocessing_evidence_gate():
    """Would fail if dropping a spaced comparator changed the supported measurement."""
    result = postprocess_draft(
        case("PLT <= 1.5 K/uL"),
        plt_item(),
        numeric_draft(model_answer=1.5, quotes=[("PLT <= 1.5 K/uL", "1.5")]),
        "full",
    )

    assert result.document_status is DocumentStatus.INDETERMINATE
    assert result.answer is None
    assert result.candidate_values == ()


def test_percentage_omission_and_alphanumeric_fragment_are_rejected_by_postprocessing():
    """Would fail if raw values could omit a percent suffix or come from an analyte name."""
    missing_percentage = postprocess_draft(
        case("PLT 60%"),
        plt_item(),
        numeric_draft(model_answer=60, quotes=[("PLT 60%", "60")]),
        "full",
    )
    alphanumeric_fragment = postprocess_draft(
        case("PLT HbA1c 7.2%"),
        plt_item(),
        numeric_draft(model_answer=1, quotes=[("HbA1c 7.2%", "1")]),
        "full",
    )

    assert missing_percentage.document_status is DocumentStatus.INDETERMINATE
    assert missing_percentage.answer is None
    assert missing_percentage.candidate_values == ()
    assert alphanumeric_fragment.document_status is DocumentStatus.INDETERMINATE
    assert alphanumeric_fragment.answer is None
    assert alphanumeric_fragment.candidate_values == ()
