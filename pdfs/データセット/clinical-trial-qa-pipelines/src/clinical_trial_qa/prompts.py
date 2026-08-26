"""Versioned JSON-only prompts shared by all clinical trial QA pipelines."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from typing import Any, Iterable, Mapping

from .llm import LLMRequest
from .models import NoteCase, QuestionSpec, QuestionType


PROMPT_VERSION = "clinical-trial-qa-v1"


def build_answer_request(case: NoteCase, spec: QuestionSpec, context: Any | None = None, *, role: str | None = None) -> LLMRequest:
    """Build a single-question request that separates quotes from reasoning."""
    note_text = case.text if context is None else _context_text(context)
    status_values = ["yes", "no", "not_documented", "indeterminate"]
    if spec.question_type is QuestionType.NUMERIC:
        status_values = ["value_available", "not_documented", "indeterminate"]
    role_instruction = f"\nYour review role is: {role}." if role else ""
    system = (
        "Return JSON only. Do not use Markdown or explanatory text outside the JSON object. "
        "Evidence must be verbatim evidence: exact quotations copied from the supplied note. "
        "Keep any reasoning separate in the inference field; never mix inference into evidence. "
        "For numeric questions, enumerate every relevant numeric candidate, not only the chosen value."
    )
    contract = {
        "document_status": status_values,
        "answer": "string, number, or null",
        "unit": "string or null",
        "evidence": [{"quote": "verbatim note substring", "start_char": "integer or null", "end_char": "integer or null"}],
        "candidate_values": [{"quote": "verbatim note substring", "raw_value": "complete numeric token", "unit": "string or null"}],
        "inference": "string or null",
        "confidence": "number from 0 to 1 or null",
    }
    user = (
        f"Prompt version: {PROMPT_VERSION}\nQuestion criterion: {spec.criterion}\n"
        f"Question: {spec.question}\nExpected labels: {', '.join(spec.labels)}\n"
        f"JSON field contract: {json.dumps(contract, sort_keys=True)}{role_instruction}\n"
        f"Note:\n{note_text}"
    )
    return LLMRequest(
        "answer", (("system", system), ("user", user)), prompt_version=PROMPT_VERSION,
        decoder=_answer_decoder(tuple(status_values)),
    )


def build_role_request(case: NoteCase, spec: QuestionSpec, role: str, context: Any | None = None, current_result: Any | None = None) -> LLMRequest:
    """Build a role-review request while preserving the answer JSON contract."""
    request = build_answer_request(case, spec, context, role=role)
    current = "null" if current_result is None else json.dumps(_jsonable(current_result), sort_keys=True)
    messages = (*request.messages, ("user", f"Current result to review: {current}"))
    return LLMRequest("role", messages, prompt_version=PROMPT_VERSION, decoder=request.decoder)


def build_role_review_request(
    case: NoteCase,
    spec: QuestionSpec,
    role: str,
    context: Any | None = None,
    current_result: Any | None = None,
) -> LLMRequest:
    """Build one explicitly bounded review-round request."""
    request = build_answer_request(case, spec, context, role=role)
    current = "null" if current_result is None else json.dumps(_jsonable(current_result), sort_keys=True)
    messages = (*request.messages, ("user", f"Current result to review: {current}"))
    return LLMRequest("role_review", messages, prompt_version=PROMPT_VERSION, decoder=request.decoder)


def build_aggregation_request(spec: QuestionSpec, proposals: Iterable[Any]) -> LLMRequest:
    """Ask an aggregator to select only opaque proposal identifiers."""
    payload = [_jsonable(proposal) for proposal in proposals]
    proposal_ids = _proposal_ids(payload)
    system = "Return JSON only. Select only proposal IDs provided in the input; do not create quotations."
    user = (
        f"Prompt version: {PROMPT_VERSION}\nCriterion: {spec.criterion}\n"
        f"JSON field contract: {{\"selected_proposal_ids\": [\"provided ID\"], \"confidence\": "
        "number or null}\n"
        f"Proposals: {json.dumps(payload, sort_keys=True)}"
    )
    return LLMRequest(
        "aggregation", (("system", system), ("user", user)), prompt_version=PROMPT_VERSION,
        decoder=_aggregation_decoder(proposal_ids),
    )


def build_verification_request(case: NoteCase, spec: QuestionSpec, candidate: Any) -> LLMRequest:
    """Build a full-note evidence verification request."""
    payload, evidence_ids = _verification_payload(candidate)
    system = (
        "Return JSON only. Select only evidence IDs supplied with the candidate; do not create or copy quotations. "
        "Use the full note only to approve, abstain, or revise the candidate with those IDs."
    )
    user = (
        f"Prompt version: {PROMPT_VERSION}\nCriterion: {spec.criterion}\n"
        f"JSON field contract: {{\"approved\": true or false, \"result\": "
        "{\"document_status\": string, \"selected_evidence_ids\": [\"provided ID\"], "
        "\"answer\": string, number, or null, \"unit\": string or null, "
        "\"inference\": string or null, \"confidence\": number or null}}}\n"
        f"Candidate: {json.dumps(payload, sort_keys=True)}\nFull note:\n{case.text}"
    )
    return LLMRequest(
        "verification", (("system", system), ("user", user)), prompt_version=PROMPT_VERSION,
        decoder=_verification_decoder(evidence_ids),
    )


def build_section_label_request(section_text: str, allowed_labels: Iterable[str] = ()) -> LLMRequest:
    """Build an optional multi-label section classification request."""
    labels = tuple(sorted({str(label) for label in allowed_labels}))
    system = "Return JSON only. Return a labels array of strings; never return prose."
    user = (
        f"Prompt version: {PROMPT_VERSION}\nAllowed labels: {json.dumps(labels)}\n"
        'JSON field contract: {"labels": ["label"]}\n'
        f"Section:\n{section_text}"
    )
    return LLMRequest(
        "section_label", (("system", system), ("user", user)), prompt_version=PROMPT_VERSION,
        decoder=_section_label_decoder(frozenset(labels)),
    )


def _context_text(context: Any) -> str:
    text = getattr(context, "text", None)
    if isinstance(text, str):
        return text
    if isinstance(context, str):
        return context
    raise TypeError("context must be a string or expose a text string")


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _answer_decoder(status_values: tuple[str, ...]):
    required = {
        "document_status": str,
        "answer": (str, int, float, type(None)),
        "unit": (str, type(None)),
        "evidence": list,
        "candidate_values": list,
        "inference": (str, type(None)),
        "confidence": (int, float, type(None)),
    }

    def decode(data: dict[str, Any]) -> dict[str, Any]:
        _require_fields(data, required)
        if data["document_status"] not in status_values:
            raise ValueError("invalid document_status")
        if not all(isinstance(item, dict) for item in data["evidence"]):
            raise ValueError("evidence must contain objects")
        if not all(isinstance(item, dict) for item in data["candidate_values"]):
            raise ValueError("candidate_values must contain objects")
        for evidence in data["evidence"]:
            _validate_span(evidence, {"quote": str, "start_char": (int, type(None)), "end_char": (int, type(None))})
        for candidate in data["candidate_values"]:
            _validate_span(candidate, {"quote": str, "raw_value": str, "unit": (str, type(None))})
        _validate_confidence(data["confidence"])
        return data

    return decode


def _section_label_decoder(allowed_labels: frozenset[str]):
    def decode(data: dict[str, Any]) -> dict[str, Any]:
        _require_fields(data, {"labels": list})
        if not all(isinstance(label, str) and label.strip() for label in data["labels"]):
            raise ValueError("labels must be non-empty strings")
        if not set(data["labels"]).issubset(allowed_labels):
            raise ValueError("labels contain values outside the request vocabulary")
        return data

    return decode


def _aggregation_decoder(allowed_ids: frozenset[str]):
    def decode(data: dict[str, Any]) -> dict[str, Any]:
        _require_fields(data, {"selected_proposal_ids": list, "confidence": (int, float, type(None))})
        if not all(isinstance(identifier, str) and identifier for identifier in data["selected_proposal_ids"]):
            raise ValueError("selected_proposal_ids must contain strings")
        if not set(data["selected_proposal_ids"]).issubset(allowed_ids):
            raise ValueError("selected_proposal_ids contain values outside the request proposals")
        _validate_confidence(data["confidence"])
        return data

    return decode


def _verification_decoder(allowed_ids: frozenset[str]):
    def decode(data: dict[str, Any]) -> dict[str, Any]:
        _require_fields(data, {"approved": bool, "result": dict})
        selected = data["result"].get("selected_evidence_ids", [])
        if not isinstance(selected, list) or not all(isinstance(identifier, str) and identifier for identifier in selected):
            raise ValueError("selected_evidence_ids must contain strings")
        if not set(selected).issubset(allowed_ids):
            raise ValueError("selected_evidence_ids contain values outside the candidate ledger")
        if "confidence" in data["result"]:
            _validate_confidence(data["result"]["confidence"])
        return data

    return decode


def _require_fields(data: dict[str, Any], fields: Mapping[str, type | tuple[type, ...]]) -> None:
    for name, allowed_type in fields.items():
        if name not in data or not isinstance(data[name], allowed_type):
            raise ValueError(f"invalid response field: {name}")


def _validate_confidence(confidence: Any) -> None:
    if confidence is None:
        return
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise ValueError("confidence must be a number between 0 and 1")


def _validate_span(data: dict[str, Any], required: Mapping[str, type | tuple[type, ...]]) -> None:
    _require_fields(data, required)
    if not data["quote"]:
        raise ValueError("quote must not be empty")
    optional = {
        "section_id": (str, type(None)),
        "source_scope": (str, type(None)),
        "raw_value": (str, type(None)),
        "normalized_value": (int, float, type(None)),
        "unit": (str, type(None)),
        "time_text": (str, type(None)),
        "start_char": (int, type(None)),
        "end_char": (int, type(None)),
    }
    for name, allowed_type in optional.items():
        if name in data and not isinstance(data[name], allowed_type):
            raise ValueError(f"invalid response field: {name}")


def _proposal_ids(proposals: Iterable[Any]) -> frozenset[str]:
    ids: set[str] = set()
    for proposal in proposals:
        if isinstance(proposal, Mapping):
            identifier = proposal.get("proposal_id", proposal.get("id"))
            if isinstance(identifier, str) and identifier:
                ids.add(identifier)
    return frozenset(ids)


def _verification_payload(candidate: Any) -> tuple[dict[str, Any], frozenset[str]]:
    raw = _jsonable(candidate)
    if not isinstance(raw, Mapping):
        return {"proposal_id": "candidate-1", "candidate": raw, "evidence": []}, frozenset()
    spans: list[dict[str, Any]] = []
    seen: set[str] = set()
    for field in ("evidence", "candidate_values"):
        values = raw.get(field, [])
        if not isinstance(values, list):
            continue
        for value in values:
            if not isinstance(value, Mapping):
                continue
            fingerprint = json.dumps(value, sort_keys=True)
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            spans.append({"evidence_id": f"evidence-{len(spans) + 1}", **dict(value)})
    candidate_fields = {
        key: value
        for key, value in raw.items()
        if key not in {"evidence", "candidate_values", "validation_errors"}
    }
    candidate_fields.update({"proposal_id": "candidate-1", "evidence": spans})
    return candidate_fields, frozenset(span["evidence_id"] for span in spans)
