from clinical_trial_qa.evidence import EvidenceLedger, EvidenceValidator
from clinical_trial_qa.models import EvidenceSpan


def test_exact_quote_recomputes_wrong_offsets():
    """Would fail if model-provided offsets were trusted instead of repaired."""
    text = "Labs: PLT 143 K/uL. Later PLT 91 K/uL."
    span = EvidenceSpan(quote="PLT 91 K/uL", start_char=0, end_char=3)

    fixed = EvidenceValidator().validate(text, span)

    assert fixed is not None
    assert text[fixed.start_char:fixed.end_char] == fixed.quote
    assert fixed.start_char == 26
    assert fixed.end_char == 37


def test_nonexistent_quote_is_rejected():
    """Would fail if a hallucinated quotation could enter downstream results."""
    span = EvidenceSpan(quote="PLT 19")

    assert EvidenceValidator().validate("PLT 91", span) is None


def test_duplicate_quote_uses_closest_valid_supplied_offset():
    """Would fail if duplicate quotes always selected their first occurrence."""
    text = "PLT 91 K/uL; later PLT 91 K/uL"
    second_start = text.rindex("PLT 91 K/uL")

    fixed = EvidenceValidator().validate(text, EvidenceSpan(quote="PLT 91 K/uL", start_char=second_start + 1))

    assert fixed is not None
    assert fixed.start_char == second_start


def test_ledger_accepts_only_the_validated_span_it_was_given():
    """Would fail if a verifier could substitute a novel citation into a ledger."""
    text = "PLT 91 K/uL"
    validated = EvidenceValidator().validate(text, EvidenceSpan(quote="PLT 91 K/uL"))
    assert validated is not None
    ledger = EvidenceLedger.from_spans((validated,))

    assert ledger.contains(validated)
    assert not ledger.contains(EvidenceSpan(quote="PLT 19", start_char=0, end_char=6))
