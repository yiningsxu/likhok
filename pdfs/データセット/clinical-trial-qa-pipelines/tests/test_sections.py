from clinical_trial_qa.llm import ScriptedLLMClient
from clinical_trial_qa.models import NoteCase
from clinical_trial_qa.questions import get_question_spec
from clinical_trial_qa.sections import RecallFirstRouter, Section, SectionSplitter


def make_case(text: str) -> NoteCase:
    return NoteCase(note_id="note-1", hadm_id="hadm-1", text=text, questions=())


def lab_case() -> NoteCase:
    return make_case("HISTORY:\nNo stroke.\n\nLABS:\nPLT 91 K/uL")


def multilabel_case() -> NoteCase:
    return make_case("HISTORY:\nPrior stroke.\n\nNEUROLOGY:\nRecent stroke this admission.\n\nLABS:\nPLT 91 K/uL")


def test_lab_question_routes_lab_section_and_preserves_full_offsets():
    """Would fail if routed spans did not retain document-relative offsets."""
    text = "HISTORY:\nNo stroke.\n\nLABS:\nPLT 91 K/uL"

    routed = RecallFirstRouter(top_k=3).route(make_case(text), get_question_spec("PLT"))

    assert "PLT 91 K/uL" in routed.text
    assert routed.sections[0].start_char == text.index("LABS:")


def test_router_falls_back_to_full_text_when_no_label_matches():
    """Would fail if a label miss dropped the note rather than preserving recall."""
    text = "Narrative without headings"

    routed = RecallFirstRouter(top_k=3).route(make_case(text), get_question_spec("bipolar"))

    assert routed.used_full_text_fallback is True
    assert routed.text == text


def test_router_keeps_multiple_labels_and_respects_top_k():
    """Would fail if multi-label sections lost their heuristic labels or exceeded top-k."""
    routed = RecallFirstRouter(top_k=2).route(multilabel_case(), get_question_spec("recent_stroke"))

    assert len(routed.sections) <= 2
    assert {"neurology", "history"} & set(routed.sections[0].labels)


def test_malformed_llm_labels_fall_back_to_heuristics():
    """Would fail if malformed optional labels overwrote safe heuristic labels."""
    client = ScriptedLLMClient([{"labels": "not-a-list"}])

    routed = RecallFirstRouter(top_k=2, label_client=client).route(lab_case(), get_question_spec("PLT"))

    assert "laboratory" in routed.sections[0].labels


def test_label_client_can_add_but_not_remove_heuristic_labels():
    """Would fail if optional labels replaced the deterministic section labels."""
    client = ScriptedLLMClient([{"labels": ["cardiology"]}])

    routed = RecallFirstRouter(top_k=1, label_client=client).route(
        make_case("LABS:\nPLT 91 K/uL"), get_question_spec("PLT")
    )

    assert {"laboratory", "cardiology"} <= set(routed.sections[0].labels)


def test_client_only_labels_cannot_outrank_later_deterministic_bipolar_evidence():
    """Would fail if a plausible but wrong client label displaced exact routed evidence."""
    text = "HISTORY:\nNo relevant history.\n\nPSYCHIATRY:\nBipolar disorder documented."
    client = ScriptedLLMClient([{"labels": ["diagnosis", "psychiatry"]}, {"labels": []}])

    routed = RecallFirstRouter(top_k=1, label_client=client).route(make_case(text), get_question_spec("bipolar"))

    assert routed.used_full_text_fallback is False
    assert routed.sections[0].start_char == text.index("PSYCHIATRY:")
    assert "Bipolar disorder documented." in routed.text


def test_client_only_labels_cannot_suppress_full_text_fallback():
    """Would fail if an ungrounded label turned a no-match route into a partial context."""
    text = "HISTORY:\nNo relevant history."
    client = ScriptedLLMClient([{"labels": ["diagnosis", "psychiatry"]}])

    routed = RecallFirstRouter(top_k=1, label_client=client).route(make_case(text), get_question_spec("bipolar"))

    assert routed.used_full_text_fallback is True
    assert routed.text == text


def test_client_labels_do_not_break_deterministic_alias_ties():
    """Would fail if client labels changed which otherwise tied deterministic section was selected."""
    text = "NARRATIVE:\nPLT 91 K/uL\n\nLABS:\nNo platelet value here."

    without_client = RecallFirstRouter(top_k=1).route(make_case(text), get_question_spec("PLT"))
    with_client = RecallFirstRouter(
        top_k=1,
        label_client=ScriptedLLMClient([{"labels": []}, {"labels": ["laboratory"]}]),
    ).route(make_case(text), get_question_spec("PLT"))

    assert without_client.sections[0].start_char == text.index("NARRATIVE:")
    assert with_client.sections[0].start_char == text.index("NARRATIVE:")


def test_client_only_section_metadata_cannot_prevent_full_text_fallback():
    """Would fail if selection treated client labels as deterministic routing labels."""
    text = "Unlabelled narrative"

    class ClientOnlySplitter:
        def split(self, _text: str):
            return (Section("client-1", text, 0, len(text), ("laboratory",), (), ("laboratory",)),)

    routed = RecallFirstRouter(top_k=1, splitter=ClientOnlySplitter()).route(make_case(text), get_question_spec("PLT"))

    assert routed.used_full_text_fallback is True
    assert routed.text == text


def test_splitter_uses_heading_start_as_section_offset():
    """Would fail if a heading were excluded from its section's full-note span."""
    text = "HISTORY:\nNone\n\nLABS:\nPLT 91 K/uL"

    sections = SectionSplitter().split(text)

    assert sections[-1].start_char == text.index("LABS:")
    assert text[sections[-1].start_char:sections[-1].end_char] == sections[-1].text


def test_splitter_preserves_preamble_before_the_first_heading():
    """Would fail if routing silently discarded an unheaded opening note span."""
    text = "PLT 91 K/uL\n\nLABS:\nPLT 143 K/uL"

    sections = SectionSplitter().split(text)

    assert sections[0].start_char == 0
    assert sections[0].text.startswith("PLT 91 K/uL")
