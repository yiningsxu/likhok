from clinical_trial_qa.aggregation import EvidenceGatedAggregator
from clinical_trial_qa.llm import ScriptedLLMClient
from clinical_trial_qa.models import DocumentStatus, NoteCase, QuestionItem
from clinical_trial_qa.pipelines import (
    P1Pipeline,
    P2Pipeline,
    P3Pipeline,
    P4Pipeline,
    P5Pipeline,
    P6Pipeline,
    PipelineComponents,
    build_pipeline,
)
from clinical_trial_qa.questions import get_question_spec
from clinical_trial_qa.sections import RecallFirstRouter, RoutedContext, Section


def _draft(status: str = "not_documented") -> dict[str, object]:
    return {
        "document_status": status,
        "answer": status if status in {"yes", "no"} else None,
        "unit": None,
        "evidence": [],
        "candidate_values": [],
        "inference": None,
        "confidence": 0.5,
    }


def _case(question_count: int = 1, text: str = "No relevant clinical documentation.") -> NoteCase:
    questions = tuple(QuestionItem(get_question_spec("bipolar")) for _ in range(question_count))
    return NoteCase("note-1", "hadm-1", text, questions)


def _numeric_case(text: str) -> NoteCase:
    return NoteCase("note-1", "hadm-1", text, (QuestionItem(get_question_spec("PLT")),))


def _numeric_draft(quote: str, inference: str) -> dict[str, object]:
    span = {"quote": quote, "raw_value": "91", "unit": "K/uL"}
    return {
        "document_status": "value_available",
        "answer": 91,
        "unit": "K/uL",
        "evidence": [{"quote": quote, "start_char": None, "end_char": None}],
        "candidate_values": [span],
        "inference": inference,
        "confidence": 0.8,
    }


class TrackingRouter:
    def __init__(self, *, fallback: bool = False) -> None:
        self.fallback = fallback
        self.calls = 0

    def route(self, case: NoteCase, _spec) -> RoutedContext:
        self.calls += 1
        section = Section("section-1", case.text, 0, len(case.text), ("other",), ("other",))
        return RoutedContext(case.text, (section,), used_full_text_fallback=self.fallback)


class TrackingVerifier:
    def __init__(self) -> None:
        self.calls = 0

    def verify(self, _case, _item, candidate):
        self.calls += 1
        return candidate


class RaisingVerifier:
    def __init__(self) -> None:
        self.calls = 0

    def verify(self, _case, _item, _candidate):
        self.calls += 1
        raise RuntimeError("verifier provider failure")


class RaisingAggregator:
    def __init__(self, message: str) -> None:
        self.message = message

    def aggregate(self, _case, _item, _proposals):
        raise RuntimeError(self.message)


class RaisingRolePanel:
    def __init__(self, message: str) -> None:
        self.message = message

    def answer(self, _case, _item, _context, _client):
        raise RuntimeError(self.message)


def _assert_value_error(callable_object) -> None:
    try:
        callable_object()
    except ValueError:
        return
    raise AssertionError("ValueError was not raised")


def test_pipeline_topologies_have_the_required_primary_router_and_verifier_calls():
    """Would fail if any named pipeline silently used another pipeline's topology."""
    expected = {
        "p1": (1, False, False),
        "p2": (3, False, False),
        "p3": (3, False, False),
        "p4": (1, True, True),
        "p5": (3, True, True),
        "p6": (3, True, True),
    }
    for name, (primary_calls, uses_router, uses_verifier) in expected.items():
        clients = tuple(ScriptedLLMClient([_draft()]) for _ in range(3))
        role_client = ScriptedLLMClient([_draft()] * 3)
        router = TrackingRouter()
        verifier = TrackingVerifier()
        components = PipelineComponents(
            primary_clients=clients,
            role_client=role_client,
            aggregator=EvidenceGatedAggregator(),
            router=router,
            verifier=verifier,
            max_rounds=0,
        )

        results = build_pipeline(name, components).run_case(_case())
        observed = sum(len(client.calls) for client in clients) + len(role_client.calls)

        assert len(results) == 1
        assert observed == primary_calls
        assert bool(router.calls) is uses_router
        assert bool(verifier.calls) is uses_verifier


def test_direct_pipeline_constructors_share_the_run_case_contract():
    """Would fail if a public P1-P6 class did not return one result per question."""
    router = TrackingRouter()
    pipelines = (
        P1Pipeline(ScriptedLLMClient([_draft()])),
        P2Pipeline((ScriptedLLMClient([_draft()]), ScriptedLLMClient([_draft()]))),
        P3Pipeline(ScriptedLLMClient([_draft()] * 3), max_rounds=0),
        P4Pipeline(ScriptedLLMClient([_draft()]), router=router, verifier=TrackingVerifier()),
        P5Pipeline(
            (ScriptedLLMClient([_draft()]), ScriptedLLMClient([_draft()])),
            router=router,
            verifier=TrackingVerifier(),
        ),
        P6Pipeline(
            ScriptedLLMClient([_draft()] * 3),
            router=router,
            verifier=TrackingVerifier(),
            max_rounds=0,
        ),
    )

    assert all(len(pipeline.run_case(_case())) == 1 for pipeline in pipelines)


def test_one_question_failure_does_not_abort_remaining_questions():
    """Would fail if malformed JSON for one criterion stopped the entire note."""
    client = ScriptedLLMClient([{"bad": "shape"}, _draft()])

    results = P1Pipeline(client).run_case(_case(question_count=2))

    assert len(results) == 2
    assert results[0].validation_errors == ("primary_invalid_response",)
    assert results[1].document_status is DocumentStatus.NOT_DOCUMENTED


def test_question_failure_errors_never_include_note_text_or_exception_messages():
    """Would fail if per-question isolation leaked clinical text through an exception string."""
    note_text = "SENSITIVE NOTE CONTENT"

    class RaisingClient:
        def generate(self, _request):
            raise RuntimeError(note_text)

    result = P1Pipeline(RaisingClient()).run_case(_case(text=note_text))[0]

    assert result.validation_errors == ("primary_invalid_response",)
    assert note_text not in repr(result)


def test_primary_answer_decoder_rejects_boolean_confidence():
    """Would fail if JSON true passed the numeric confidence contract as integer one."""
    malformed = {**_draft(), "confidence": True}

    result = P1Pipeline(ScriptedLLMClient([malformed])).run_case(_case())[0]

    assert result.document_status is DocumentStatus.INDETERMINATE
    assert result.confidence is None
    assert result.validation_errors == ("primary_invalid_response",)


def test_routed_pipelines_verify_exactly_once_even_after_full_text_fallback():
    """Would fail if fallback skipped or duplicated the mandatory full-note verifier audit."""
    builders = (
        lambda router, verifier: P4Pipeline(ScriptedLLMClient([_draft()]), router=router, verifier=verifier),
        lambda router, verifier: P5Pipeline(
            (ScriptedLLMClient([_draft()]), ScriptedLLMClient([_draft()])),
            router=router,
            verifier=verifier,
        ),
        lambda router, verifier: P6Pipeline(
            ScriptedLLMClient([_draft()] * 3), router=router, verifier=verifier, max_rounds=0
        ),
    )
    for builder in builders:
        router = TrackingRouter(fallback=True)
        verifier = TrackingVerifier()

        builder(router, verifier).run_case(_case())

        assert router.calls == 1
        assert verifier.calls == 1


def test_routed_pipeline_still_verifies_a_malformed_primary_result_once():
    """Would fail if primary failure returned before the routed verifier boundary."""
    router = TrackingRouter(fallback=True)
    verifier = TrackingVerifier()

    results = P4Pipeline(ScriptedLLMClient([{"bad": "shape"}]), router=router, verifier=verifier).run_case(_case())

    assert results[0].validation_errors == ("primary_invalid_response",)
    assert verifier.calls == 1


def test_routed_duplicate_quote_maps_to_the_selected_full_note_occurrence():
    """Would fail if a routed-relative offset selected the first duplicate in the full note."""
    text = "HISTORY:\nPLT 91 K/uL\n\nLABS:\nPLT 91 K/uL"
    routed_quote_offset = len("LABS:\n")
    response = {
        "document_status": "value_available",
        "answer": 91,
        "unit": "K/uL",
        "evidence": [
            {"quote": "PLT 91 K/uL", "start_char": routed_quote_offset, "end_char": None}
        ],
        "candidate_values": [
            {
                "quote": "PLT 91 K/uL",
                "start_char": routed_quote_offset,
                "raw_value": "91",
                "unit": "K/uL",
            }
        ],
        "inference": None,
        "confidence": 0.8,
    }

    result = P4Pipeline(
        ScriptedLLMClient([response]),
        router=RecallFirstRouter(top_k=1),
        verifier=TrackingVerifier(),
    ).run_case(_numeric_case(text))[0]

    selected_start = text.rindex("PLT 91 K/uL")
    assert result.answer == 91.0
    assert result.evidence[0].start_char == selected_start
    assert result.candidate_values[0].start_char == selected_start
    assert result.candidate_values[0].section_id == "labs-2"


def test_routed_evidence_outside_selected_sections_is_rejected():
    """Would fail if routed postprocessing searched an unselected full-note range."""
    text = "HISTORY:\nPLT 143 K/uL\n\nLABS:\nPLT 91 K/uL"
    response = {
        "document_status": "value_available",
        "answer": 143,
        "unit": "K/uL",
        "evidence": [{"quote": "PLT 143 K/uL", "start_char": 0, "end_char": None}],
        "candidate_values": [
            {"quote": "PLT 143 K/uL", "start_char": 0, "raw_value": "143", "unit": "K/uL"}
        ],
        "inference": None,
        "confidence": 0.8,
    }

    result = P4Pipeline(
        ScriptedLLMClient([response]),
        router=RecallFirstRouter(top_k=1),
        verifier=TrackingVerifier(),
    ).run_case(_numeric_case(text))[0]

    assert result.document_status is DocumentStatus.INDETERMINATE
    assert result.answer is None
    assert result.candidate_values == ()
    assert "invalid_numeric_candidate_quote" in result.validation_errors


def test_boolean_verifier_exception_returns_a_fully_sanitized_result():
    """Would fail if boolean inference or candidate metadata survived a verifier exception."""
    unique_sentence = "UNIQUE BOOLEAN NOTE SENTENCE"
    client = ScriptedLLMClient(
        [
            {
                "document_status": "yes",
                "answer": "yes",
                "unit": None,
                "evidence": [{"quote": unique_sentence, "start_char": None, "end_char": None}],
                "candidate_values": [],
                "inference": unique_sentence,
                "confidence": 0.9,
            }
        ]
    )

    result = P4Pipeline(client, router=TrackingRouter(), verifier=RaisingVerifier()).run_case(
        _case(text=unique_sentence)
    )[0]

    assert result.evidence == ()
    assert result.candidate_values == ()
    assert result.inference is None
    assert result.answer is None
    assert result.unit is None
    assert result.confidence == 0.0
    assert result.provenance == ("verifier:failed",)
    assert result.validation_errors == ("verifier_invalid_response:RuntimeError",)
    assert unique_sentence not in repr(result)
    assert unique_sentence not in repr(result.to_dict())


def test_numeric_verifier_exception_returns_a_fully_sanitized_result():
    """Would fail if numeric candidates or inference survived a verifier exception."""
    unique_sentence = "PLT 91 K/uL UNIQUE NUMERIC NOTE SENTENCE"

    result = P4Pipeline(
        ScriptedLLMClient([_numeric_draft(unique_sentence, unique_sentence)]),
        router=TrackingRouter(),
        verifier=RaisingVerifier(),
    ).run_case(_numeric_case(unique_sentence))[0]

    assert result.evidence == ()
    assert result.candidate_values == ()
    assert result.inference is None
    assert result.answer is None
    assert result.unit is None
    assert result.confidence == 0.0
    assert result.provenance == ("verifier:failed",)
    assert result.validation_errors == ("verifier_invalid_response:RuntimeError",)
    assert unique_sentence not in repr(result)
    assert unique_sentence not in repr(result.to_dict())


def test_p5_aggregation_failure_still_calls_verifier_once_with_a_sanitized_candidate():
    """Would fail if P5 returned through base isolation before attempting mandatory verification."""
    unique_sentence = "UNIQUE P5 AGGREGATION FAILURE NOTE"
    verifier = TrackingVerifier()
    pipeline = P5Pipeline(
        (ScriptedLLMClient([_draft()]), ScriptedLLMClient([_draft()])),
        router=TrackingRouter(),
        verifier=verifier,
        aggregator=RaisingAggregator(unique_sentence),
    )

    result = pipeline.run_case(_case(text=unique_sentence))[0]

    assert verifier.calls == 1
    assert result.document_status is DocumentStatus.INDETERMINATE
    assert result.confidence == 0.0
    assert result.provenance == ("candidate:failed",)
    assert result.validation_errors == ("candidate_build_failed:RuntimeError",)
    assert unique_sentence not in repr(result)
    assert unique_sentence not in repr(result.to_dict())


def test_p6_role_panel_failure_still_calls_verifier_once_with_a_sanitized_candidate():
    """Would fail if P6 returned through base isolation before attempting mandatory verification."""
    unique_sentence = "UNIQUE P6 ROLE PANEL FAILURE NOTE"
    verifier = TrackingVerifier()
    pipeline = P6Pipeline(
        ScriptedLLMClient([]), router=TrackingRouter(), verifier=verifier, max_rounds=0
    )
    pipeline.panel = RaisingRolePanel(unique_sentence)

    result = pipeline.run_case(_case(text=unique_sentence))[0]

    assert verifier.calls == 1
    assert result.document_status is DocumentStatus.INDETERMINATE
    assert result.confidence == 0.0
    assert result.provenance == ("candidate:failed",)
    assert result.validation_errors == ("candidate_build_failed:RuntimeError",)
    assert unique_sentence not in repr(result)
    assert unique_sentence not in repr(result.to_dict())


def test_multi_primary_and_role_constructor_constraints_are_enforced():
    """Would fail if ensemble size or bounded role rounds could violate the study topology."""
    one_client = (ScriptedLLMClient([_draft()]),)

    _assert_value_error(lambda: P2Pipeline(one_client))
    _assert_value_error(lambda: P5Pipeline(one_client, router=TrackingRouter(), verifier=TrackingVerifier()))
    _assert_value_error(lambda: P3Pipeline(None, max_rounds=0))
    _assert_value_error(lambda: P3Pipeline(ScriptedLLMClient([]), max_rounds=3))
    _assert_value_error(
        lambda: P6Pipeline(None, router=TrackingRouter(), verifier=TrackingVerifier(), max_rounds=0)
    )
    _assert_value_error(
        lambda: P6Pipeline(
            ScriptedLLMClient([]), router=TrackingRouter(), verifier=TrackingVerifier(), max_rounds=-1
        )
    )


def test_p2_and_p5_reject_the_same_stateful_client_object_twice():
    """Would fail if an ensemble reused one stateful client instead of making independent calls."""
    client = ScriptedLLMClient([_draft(), _draft()])

    _assert_value_error(lambda: P2Pipeline((client, client)))
    _assert_value_error(lambda: P5Pipeline((client, client), verifier=TrackingVerifier()))
    _assert_value_error(
        lambda: build_pipeline("p2", PipelineComponents(primary_clients=(client, client)))
    )
    _assert_value_error(
        lambda: build_pipeline(
            "p5",
            PipelineComponents(primary_clients=(client, client), verifier=TrackingVerifier()),
        )
    )


def test_p2_and_p5_allow_distinct_clients_with_the_same_model_label():
    """Would fail if independence were incorrectly inferred from a shared model string."""
    class LabelledClient:
        model = "shared-model"

        def generate(self, _request):
            raise AssertionError("constructor test does not generate")

    clients = (LabelledClient(), LabelledClient())

    assert isinstance(P2Pipeline(clients), P2Pipeline)
    assert isinstance(P5Pipeline(clients, verifier=TrackingVerifier()), P5Pipeline)


def test_factory_rejects_unknown_pipeline_names():
    """Would fail if a typo silently selected a different experimental pipeline."""
    components = PipelineComponents(primary_clients=(ScriptedLLMClient([_draft()]),))

    _assert_value_error(lambda: build_pipeline("p7", components))
