import json
from dataclasses import replace
from urllib import request as urlrequest

from clinical_trial_qa.llm import (
    ConfigurationError,
    LLMRequest,
    OpenAICompatibleClient,
    LLMResponseError,
    ScriptedLLMClient,
)
from clinical_trial_qa.models import NoteCase
from clinical_trial_qa.prompts import build_aggregation_request, build_answer_request, build_section_label_request
from clinical_trial_qa.questions import get_question_spec


def request() -> LLMRequest:
    return LLMRequest(task="answer", messages=(("user", "test request"),))


def _assert_raises(error_type, action, fragment=None):
    try:
        action()
    except error_type as exc:
        if fragment is not None:
            assert fragment in str(exc)
        return exc
    raise AssertionError(f"{error_type.__name__} was not raised")


def test_scripted_client_records_metadata_without_prompt_text():
    """Would fail if call telemetry retained the clinical prompt."""
    client = ScriptedLLMClient([{"document_status": "not_documented", "evidence": []}])

    response = client.generate(LLMRequest(task="answer", messages=(("user", "sensitive note"),)))

    assert response.data["document_status"] == "not_documented"
    assert client.calls[0].task == "answer"
    assert "sensitive note" not in repr(client.calls[0])


def test_openai_client_requires_key_before_network_call(monkeypatch):
    """Would fail if a missing key could initiate an HTTP request."""
    monkeypatch.delenv("MISSING_TEST_KEY", raising=False)
    client = OpenAICompatibleClient(base_url="https://example.invalid/v1", model="m", api_key_env="MISSING_TEST_KEY")

    _assert_raises(ConfigurationError, lambda: client.generate(request()))


def test_scripted_client_exhaustion_is_explicit():
    """Would fail if dry-run responses were silently fabricated after the script ends."""
    client = ScriptedLLMClient([])

    _assert_raises(RuntimeError, lambda: client.generate(request()), "exhausted")


def test_answer_prompt_is_versioned_json_only_and_preserves_evidence_contract():
    """Would fail if an answer prompt allowed prose or omitted numeric candidates."""
    case = NoteCase("n1", "h1", "LABS:\nPLT 91 K/uL", ())

    built = build_answer_request(case, get_question_spec("PLT"))
    prompt = "\n".join(content for _role, content in built.messages)

    assert built.task == "answer"
    assert built.prompt_version
    assert "JSON only" in prompt
    assert "candidate_values" in prompt
    assert "verbatim evidence" in prompt
    assert "inference" in prompt
    assert "PLT 91 K/uL" in prompt


def test_section_label_prompt_is_json_only_without_question_note_mixing():
    """Would fail if label prompts did not declare their labels JSON contract."""
    built = build_section_label_request("LABS:\nPLT 91 K/uL", ("laboratory", "history"))
    prompt = "\n".join(content for _role, content in built.messages)

    assert built.task == "section_label"
    assert "JSON only" in prompt
    assert '"labels"' in prompt
    assert "laboratory" in prompt


class _FakeHTTPResponse:
    def __init__(self, content: str):
        self._content = content

    def read(self) -> bytes:
        return json.dumps({"choices": [{"message": {"content": self._content}}]}).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def test_openai_client_retries_invalid_task_shape_then_returns_valid_response(monkeypatch):
    """Would fail if syntactically valid but contract-invalid JSON bypassed retries."""
    responses = iter([_FakeHTTPResponse('{"bad": "shape"}'), _FakeHTTPResponse('{"labels": ["laboratory"]}')])
    monkeypatch.setenv("LLM_TEST_KEY", "test-key")
    monkeypatch.setattr(urlrequest, "urlopen", lambda _request, timeout: next(responses))
    client = OpenAICompatibleClient(base_url="https://example.invalid/v1", model="m", api_key_env="LLM_TEST_KEY")

    response = client.generate(replace(build_section_label_request("LABS:\\nPLT 91 K/uL", ("laboratory",)), max_retries=1))

    assert response.data == {"labels": ["laboratory"]}
    assert len(client.calls) == 2
    assert client.calls[0].error == "ValueError"


def test_openai_client_reports_decoder_exhaustion_without_response_or_prompt_text(monkeypatch):
    """Would fail if invalid structured output leaked PHI when retries were exhausted."""
    monkeypatch.setenv("LLM_TEST_KEY", "test-key")
    monkeypatch.setattr(urlrequest, "urlopen", lambda _request, timeout: _FakeHTTPResponse('{"bad": "patient secret"}'))
    client = OpenAICompatibleClient(base_url="https://example.invalid/v1", model="m", api_key_env="LLM_TEST_KEY")
    request = build_section_label_request("sensitive clinical note", ("laboratory",))

    error = _assert_raises(LLMResponseError, lambda: client.generate(request))

    assert "patient secret" not in str(error)
    assert "sensitive clinical note" not in str(error)


def test_section_label_decoder_retries_an_unsupported_label_then_accepts_an_allowed_one(monkeypatch):
    """Would fail if a syntactically valid label outside this request's vocabulary passed."""
    responses = iter([_FakeHTTPResponse('{"labels": ["psychiatry"]}'), _FakeHTTPResponse('{"labels": ["laboratory"]}')])
    monkeypatch.setenv("LLM_TEST_KEY", "test-key")
    monkeypatch.setattr(urlrequest, "urlopen", lambda _request, timeout: next(responses))
    client = OpenAICompatibleClient(base_url="https://example.invalid/v1", model="m", api_key_env="LLM_TEST_KEY")

    response = client.generate(replace(build_section_label_request("LABS", ("laboratory",)), max_retries=1))

    assert response.data == {"labels": ["laboratory"]}
    assert len(client.calls) == 2


def test_aggregation_decoder_rejects_an_invented_proposal_id(monkeypatch):
    """Would fail if aggregation could select a proposal not included in its request."""
    monkeypatch.setenv("LLM_TEST_KEY", "test-key")
    monkeypatch.setattr(urlrequest, "urlopen", lambda _request, timeout: _FakeHTTPResponse('{"selected_proposal_ids": ["invented"], "confidence": 0.5}'))
    client = OpenAICompatibleClient(base_url="https://example.invalid/v1", model="m", api_key_env="LLM_TEST_KEY")
    request = build_aggregation_request(get_question_spec("PLT"), ({"proposal_id": "p-1"},))

    _assert_raises(LLMResponseError, lambda: client.generate(request))


def test_answer_decoder_retries_malformed_nested_evidence_then_accepts_valid_output(monkeypatch):
    """Would fail if missing required quote fields in nested evidence were accepted."""
    bad = '{"document_status":"value_available","answer":91,"unit":"K/uL","evidence":[{"start_char":0,"end_char":1}],"candidate_values":[],"inference":null,"confidence":0.8}'
    valid = '{"document_status":"value_available","answer":91,"unit":"K/uL","evidence":[{"quote":"PLT 91 K/uL","start_char":0,"end_char":11,"section_id":null}],"candidate_values":[{"quote":"PLT 91 K/uL","raw_value":"91","unit":"K/uL","time_text":null}],"inference":null,"confidence":0.8}'
    responses = iter([_FakeHTTPResponse(bad), _FakeHTTPResponse(valid)])
    monkeypatch.setenv("LLM_TEST_KEY", "test-key")
    monkeypatch.setattr(urlrequest, "urlopen", lambda _request, timeout: next(responses))
    client = OpenAICompatibleClient(base_url="https://example.invalid/v1", model="m", api_key_env="LLM_TEST_KEY")
    case = NoteCase("n1", "h1", "PLT 91 K/uL", ())

    response = client.generate(replace(build_answer_request(case, get_question_spec("PLT")), max_retries=1))

    assert response.data["evidence"][0]["quote"] == "PLT 91 K/uL"
    assert len(client.calls) == 2
