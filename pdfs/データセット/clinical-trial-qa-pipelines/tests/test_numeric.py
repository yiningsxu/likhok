from clinical_trial_qa.models import DocumentStatus, EvidenceSpan
from clinical_trial_qa.numeric import parse_numeric, reduce_candidates
from clinical_trial_qa.questions import get_question_spec


def candidate(quote: str, raw_value: str, unit: str | None) -> EvidenceSpan:
    return EvidenceSpan(quote=quote, raw_value=raw_value, unit=unit)


def test_plt_minimum_uses_all_validated_candidates():
    """Would fail if the reducer used only the model's preferred platelet value."""
    candidates = (
        candidate("PLT 143 K/uL", "143", "K/uL"),
        candidate("PLT 91 K/uL", "91", "K/uL"),
    )

    decision = reduce_candidates(get_question_spec("PLT"), candidates, related_mention_present=True)

    assert decision.status is DocumentStatus.VALUE_AVAILABLE
    assert decision.answer == 91.0
    assert len(decision.candidates) == 2


def test_lvef_at_or_above_55_is_legacy_clipped_but_raw_is_retained():
    """Would fail if dataset compatibility clipping overwrote the extracted value."""
    decision = reduce_candidates(get_question_spec("lvef"), (candidate("EF 60%", "60%", "%"),), True)

    assert decision.answer == 55.0
    assert decision.candidates[0].normalized_value == 60.0


def test_incompatible_units_abstain_without_dropping_candidates():
    """Would fail if mixed units were silently compared or discarded."""
    candidates = (
        candidate("Creat 1 mg/dL", "1", "mg/dL"),
        candidate("Creat 90 umol/L", "90", "umol/L"),
    )

    decision = reduce_candidates(get_question_spec("CREAT"), candidates, related_mention_present=True)

    assert decision.status is DocumentStatus.INDETERMINATE
    assert decision.answer is None
    assert len(decision.candidates) == 2


def test_visible_quote_units_defeat_model_spoofing_and_preserve_raw_spans():
    """Would fail if equal model units hid incompatible units visible in exact quotes."""
    candidates = (
        candidate("Creat 1 mg/dL", "1", "mg/dL"),
        candidate("Creat 90 umol/L", "90", "mg/dL"),
    )

    decision = reduce_candidates(get_question_spec("CREAT"), candidates, related_mention_present=True)

    assert decision.status is DocumentStatus.INDETERMINATE
    assert decision.answer is None
    assert tuple(span.quote for span in decision.candidates) == ("Creat 1 mg/dL", "Creat 90 umol/L")
    assert tuple(span.unit for span in decision.candidates) == ("mg/dL", "mg/dL")


def test_visible_quote_unit_rejects_model_contradiction_or_omission():
    """Would fail if a visible unit could be contradicted or omitted by model metadata."""
    contradiction = reduce_candidates(
        get_question_spec("CREAT"),
        (candidate("Creat 1 mg/dL", "1", "umol/L"),),
        related_mention_present=True,
    )
    omission = reduce_candidates(
        get_question_spec("CREAT"),
        (candidate("Creat 1 mg/dL", "1", None),),
        related_mention_present=True,
    )

    assert contradiction.status is DocumentStatus.INDETERMINATE
    assert contradiction.answer is None
    assert omission.status is DocumentStatus.INDETERMINATE
    assert omission.answer is None


def test_numeric_parser_accepts_inequality_commas_and_percentages():
    """Would fail if common lab-value notation were treated as nonnumeric."""
    assert parse_numeric(">=1,234.50%") == 1234.5
    assert parse_numeric("not a value") is None


def test_related_mention_without_a_parseable_value_is_indeterminate():
    """Would fail if a mentioned test with no usable value became not documented."""
    decision = reduce_candidates(get_question_spec("PLT"), (candidate("PLT pending", "pending", None),), True)

    assert decision.status is DocumentStatus.INDETERMINATE
    assert decision.answer is None
    assert decision.candidates[0].normalized_value is None


def test_partial_numeric_substrings_are_not_candidate_values():
    """Would fail if a digit prefix inside a different measurement were accepted."""
    partial_integer = reduce_candidates(
        get_question_spec("PLT"),
        (candidate("PLT 143 K/uL", "1", "K/uL"),),
        related_mention_present=True,
    )
    partial_decimal = reduce_candidates(
        get_question_spec("PLT"),
        (candidate("PLT 1.43 K/uL", "1.4", "K/uL"),),
        related_mention_present=True,
    )
    partial_comparator = reduce_candidates(
        get_question_spec("PLT"),
        (candidate("PLT <=1.5 K/uL", "1.5", "K/uL"),),
        related_mention_present=True,
    )
    spaced_comparator_omission = reduce_candidates(
        get_question_spec("PLT"),
        (candidate("PLT <= 1.5 K/uL", "1.5", "K/uL"),),
        related_mention_present=True,
    )

    assert partial_integer.status is DocumentStatus.INDETERMINATE
    assert partial_integer.candidates[0].normalized_value is None
    assert partial_decimal.status is DocumentStatus.INDETERMINATE
    assert partial_decimal.candidates[0].normalized_value is None
    assert partial_comparator.status is DocumentStatus.INDETERMINATE
    assert partial_comparator.candidates[0].normalized_value is None
    assert spaced_comparator_omission.status is DocumentStatus.INDETERMINATE
    assert spaced_comparator_omission.candidates[0].normalized_value is None


def test_complete_numeric_tokens_support_comparator_commas_and_percentages():
    """Would fail if legitimate standalone numeric syntax were rejected with boundary checks."""
    comparator = reduce_candidates(
        get_question_spec("PLT"),
        (candidate("PLT <= 1.5 K/uL", "<=1.5", "K/uL"),),
        related_mention_present=True,
    )
    comma_separated = reduce_candidates(
        get_question_spec("PLT"),
        (candidate("PLT 1,234.50 K/uL", "1,234.50", "K/uL"),),
        related_mention_present=True,
    )
    percentage = reduce_candidates(
        get_question_spec("lvef"), (candidate("EF 60%", "60%", "%"),), True)

    assert comparator.answer == 1.5
    assert comma_separated.answer == 1234.5
    assert percentage.answer == 55.0


def test_missing_percentage_or_alphanumeric_fragment_is_not_a_complete_numeric_token():
    """Would fail if token extraction accepted omitted suffixes or digits inside analyte names."""
    missing_percentage = reduce_candidates(
        get_question_spec("lvef"), (candidate("EF 60%", "60", "%"),), True
    )
    alphanumeric_fragment = reduce_candidates(
        get_question_spec("PLT"), (candidate("HbA1c 7.2%", "1", None),), True
    )

    assert missing_percentage.status is DocumentStatus.INDETERMINATE
    assert missing_percentage.candidates[0].normalized_value is None
    assert alphanumeric_fragment.status is DocumentStatus.INDETERMINATE
    assert alphanumeric_fragment.candidates[0].normalized_value is None
