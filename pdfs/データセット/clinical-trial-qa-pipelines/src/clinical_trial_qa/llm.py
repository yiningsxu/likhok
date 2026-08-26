"""Vendor-neutral JSON-only boundary for language model providers."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
import time
from typing import Any, Callable, Mapping, Protocol, Sequence
from urllib import error as urlerror
from urllib import request as urlrequest
from uuid import uuid4


class ConfigurationError(ValueError):
    """Raised before a provider request when required client configuration is absent."""


class LLMResponseError(RuntimeError):
    """Raised when an LLM response is unusable after its bounded retry budget."""


ResponseDecoder = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class LLMRequest:
    """A provider-independent request containing only chat messages and task metadata."""

    task: str
    messages: tuple[tuple[str, str], ...]
    temperature: float = 0.0
    max_retries: int = 0
    prompt_version: str | None = None
    decoder: ResponseDecoder | None = None

    def __post_init__(self) -> None:
        if not self.task:
            raise ValueError("task must not be empty")
        if not 0 <= self.max_retries <= 2:
            raise ValueError("max_retries must be between 0 and 2")
        if not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        if not self.messages:
            raise ValueError("messages must not be empty")
        if any(not role or not isinstance(content, str) for role, content in self.messages):
            raise ValueError("messages must contain non-empty roles and string content")

    def decode(self, data: dict[str, Any]) -> dict[str, Any]:
        """Validate and normalize a provider JSON object for this request's task."""
        if self.decoder is None:
            return data
        decoded = self.decoder(data)
        if not isinstance(decoded, dict):
            raise ValueError("response decoder must return a JSON object")
        return decoded

    @property
    def input_chars(self) -> int:
        return sum(len(content) for _role, content in self.messages)

    @property
    def prompt_hash(self) -> str:
        digest = hashlib.sha256()
        for role, content in self.messages:
            digest.update(role.encode("utf-8"))
            digest.update(b"\0")
            digest.update(content.encode("utf-8"))
            digest.update(b"\0")
        return digest.hexdigest()


@dataclass(frozen=True)
class LLMResponse:
    """Parsed JSON data plus safe operational metadata."""

    data: dict[str, Any]
    model: str | None = None
    input_chars: int = 0
    output_chars: int = 0
    latency_ms: float = 0.0
    run_id: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LLMCallRecord:
    """Non-sensitive telemetry; it intentionally contains no prompt or response text."""

    task: str
    model: str | None
    temperature: float
    input_chars: int
    output_chars: int | None
    latency_ms: float
    prompt_hash: str
    run_id: str
    error: str | None = None


class LLMClient(Protocol):
    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate one JSON object for a structured request."""


class ScriptedLLMClient:
    """Deterministic client for tests and dry runs without provider credentials."""

    def __init__(self, script: Sequence[dict[str, Any] | LLMResponse]) -> None:
        self._script = list(script)
        self.calls: list[LLMCallRecord] = []

    def generate(self, request: LLMRequest) -> LLMResponse:
        started = time.perf_counter()
        run_id = str(uuid4())
        if not self._script:
            self._record(request, None, started, run_id, "script_exhausted")
            raise RuntimeError("scripted LLM client is exhausted")
        scripted = self._script.pop(0)
        if isinstance(scripted, LLMResponse):
            response = scripted
        elif isinstance(scripted, dict):
            response = LLMResponse(data=dict(scripted), run_id=run_id)
        else:  # defensive despite the typed constructor boundary
            self._record(request, None, started, run_id, "invalid_scripted_response")
            raise TypeError("scripted responses must be dictionaries or LLMResponse objects")
        output_chars = len(json.dumps(response.data, sort_keys=True, default=str))
        elapsed_ms = (time.perf_counter() - started) * 1000
        self.calls.append(
            LLMCallRecord(
                task=request.task,
                model=response.model,
                temperature=request.temperature,
                input_chars=request.input_chars,
                output_chars=output_chars,
                latency_ms=elapsed_ms,
                prompt_hash=request.prompt_hash,
                run_id=response.run_id or run_id,
            )
        )
        return LLMResponse(
            data=dict(response.data),
            model=response.model,
            input_chars=request.input_chars,
            output_chars=output_chars,
            latency_ms=elapsed_ms,
            run_id=response.run_id or run_id,
            metadata=dict(response.metadata),
        )

    def _record(self, request: LLMRequest, output_chars: int | None, started: float, run_id: str, error: str) -> None:
        self.calls.append(
            LLMCallRecord(
                task=request.task,
                model=None,
                temperature=request.temperature,
                input_chars=request.input_chars,
                output_chars=output_chars,
                latency_ms=(time.perf_counter() - started) * 1000,
                prompt_hash=request.prompt_hash,
                run_id=run_id,
                error=error,
            )
        )


class OpenAICompatibleClient:
    """HTTP client for OpenAI-compatible ``/chat/completions`` JSON APIs."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key_env: str = "OPENAI_API_KEY",
        timeout: float = 60.0,
    ) -> None:
        if not base_url:
            raise ValueError("base_url must not be empty")
        if not model:
            raise ValueError("model must not be empty")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key_env = api_key_env
        self.timeout = timeout
        self.calls: list[LLMCallRecord] = []

    def generate(self, request: LLMRequest) -> LLMResponse:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise ConfigurationError(f"environment variable {self.api_key_env} is required")

        attempts = request.max_retries + 1
        last_error: Exception | None = None
        for attempt in range(attempts):
            started = time.perf_counter()
            run_id = str(uuid4())
            try:
                response = self._send(request, api_key, run_id, started)
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, urlerror.URLError) as exc:
                last_error = exc
                self._record(request, None, started, run_id, exc.__class__.__name__)
                if attempt + 1 == attempts:
                    break
            else:
                return response
        assert last_error is not None
        raise LLMResponseError(f"LLM request failed after {attempts} attempt(s): {last_error.__class__.__name__}") from last_error

    def _send(self, request: LLMRequest, api_key: str, run_id: str, started: float) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": [{"role": role, "content": content} for role, content in request.messages],
            "temperature": request.temperature,
            "response_format": {"type": "json_object"},
        }
        body = json.dumps(payload).encode("utf-8")
        http_request = urlrequest.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urlrequest.urlopen(http_request, timeout=self.timeout) as http_response:
            provider_payload = json.loads(http_response.read().decode("utf-8"))
        content = provider_payload["choices"][0]["message"]["content"]
        data = json.loads(content)
        if not isinstance(data, dict):
            raise ValueError("LLM response content must be a JSON object")
        data = request.decode(data)
        usage = provider_payload.get("usage")
        metadata: dict[str, Any] = {}
        if isinstance(usage, dict):
            metadata["usage"] = dict(usage)
        model = provider_payload.get("model") if isinstance(provider_payload.get("model"), str) else self.model
        elapsed_ms = (time.perf_counter() - started) * 1000
        self._record(request, len(content), started, run_id, None)
        return LLMResponse(
            data=data,
            model=model,
            input_chars=request.input_chars,
            output_chars=len(content),
            latency_ms=elapsed_ms,
            run_id=run_id,
            metadata=metadata,
        )

    def _record(self, request: LLMRequest, output_chars: int | None, started: float, run_id: str, error: str | None) -> None:
        self.calls.append(
            LLMCallRecord(
                task=request.task,
                model=self.model,
                temperature=request.temperature,
                input_chars=request.input_chars,
                output_chars=output_chars,
                latency_ms=(time.perf_counter() - started) * 1000,
                prompt_hash=request.prompt_hash,
                run_id=run_id,
                error=error,
            )
        )
