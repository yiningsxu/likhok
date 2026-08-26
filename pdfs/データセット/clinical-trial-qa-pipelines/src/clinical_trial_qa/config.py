"""Strict JSON configuration and fresh construction of runtime pipeline clients."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

from .aggregation import EvidenceGatedAggregator
from .llm import ConfigurationError, LLMClient, OpenAICompatibleClient, ScriptedLLMClient
from .pipelines import PipelineComponents, build_pipeline
from .pipelines.base import Pipeline
from .sections import RecallFirstRouter
from .verification import FullDocumentVerifier


_PIPELINES = frozenset({"p1", "p2", "p3", "p4", "p5", "p6"})
_MODEL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,199}$")
_ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_CLIENT_ROLES = {
    "p1": frozenset({"primary"}),
    "p2": frozenset({"primary", "aggregator"}),
    "p3": frozenset({"role", "aggregator"}),
    "p4": frozenset({"primary", "verifier", "router_label"}),
    "p5": frozenset({"primary", "verifier", "aggregator", "router_label"}),
    "p6": frozenset({"role", "verifier", "aggregator", "router_label"}),
}


@dataclass(frozen=True)
class ClientConfig:
    """A credential-safe recipe that constructs one new stateful client per call."""

    provider: str
    model: str
    base_url: str | None = None
    api_key_env: str | None = None
    timeout: float = 60.0
    script: tuple[Mapping[str, Any], ...] = ()

    def build(self) -> LLMClient:
        if self.provider == "openai_compatible":
            assert self.base_url is not None
            assert self.api_key_env is not None
            return OpenAICompatibleClient(
                base_url=self.base_url,
                model=self.model,
                api_key_env=self.api_key_env,
                timeout=self.timeout,
            )
        if self.provider == "scripted":
            return ScriptedLLMClient([dict(response) for response in self.script])
        raise ConfigurationError("unsupported client provider")

    def safe_dict(self) -> dict[str, Any]:
        """Return reproducibility fields, hashing scripts rather than retaining their content."""
        value: dict[str, Any] = {"provider": self.provider, "model": self.model}
        if self.base_url is not None:
            value["base_url"] = self.base_url
        if self.api_key_env is not None:
            value["api_key_env"] = self.api_key_env
        if self.provider == "openai_compatible":
            value["timeout"] = self.timeout
        if self.provider == "scripted":
            encoded = json.dumps(list(self.script), sort_keys=True, separators=(",", ":"), default=str)
            value["script_hash"] = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
            value["script_responses"] = len(self.script)
        return value


@dataclass(frozen=True)
class ClientSetConfig:
    primary: tuple[ClientConfig, ...] = ()
    role: ClientConfig | None = None
    verifier: ClientConfig | None = None
    aggregator: ClientConfig | None = None
    router_label: ClientConfig | None = None

    def safe_dict(self) -> dict[str, Any]:
        return {
            "primary": [client.safe_dict() for client in self.primary],
            "role": self.role.safe_dict() if self.role else None,
            "verifier": self.verifier.safe_dict() if self.verifier else None,
            "aggregator": self.aggregator.safe_dict() if self.aggregator else None,
            "router_label": self.router_label.safe_dict() if self.router_label else None,
        }


@dataclass(frozen=True)
class RuntimeClient:
    purpose: str
    model: str
    client: LLMClient


@dataclass(frozen=True)
class PipelineRuntime:
    pipeline: Pipeline
    clients: tuple[RuntimeClient, ...]


@dataclass(frozen=True)
class AppConfig:
    pipeline: str
    seed: int
    debate_rounds: int
    clients: ClientSetConfig

    @property
    def config_hash(self) -> str:
        encoded = json.dumps(self.safe_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @property
    def model_names(self) -> tuple[str, ...]:
        recipes = (
            *self.clients.primary,
            *tuple(
                client
                for client in (
                    self.clients.role,
                    self.clients.verifier,
                    self.clients.aggregator,
                    self.clients.router_label,
                )
                if client
            ),
        )
        return tuple(dict.fromkeys(client.model for client in recipes))

    def safe_dict(self) -> dict[str, Any]:
        return {
            "pipeline": self.pipeline,
            "seed": self.seed,
            "debate_rounds": self.debate_rounds,
            "clients": self.clients.safe_dict(),
        }

    def build_runtime(self) -> PipelineRuntime:
        """Construct every configured client separately so ensembles never share state."""
        _validate_app_config(self)
        tracked: list[RuntimeClient] = []

        def make(purpose: str, recipe: ClientConfig | None) -> LLMClient | None:
            if recipe is None:
                return None
            client = recipe.build()
            tracked.append(RuntimeClient(purpose, recipe.model, client))
            return client

        primary = tuple(make(f"primary_{index}", recipe) for index, recipe in enumerate(self.clients.primary, 1))
        role = make("role", self.clients.role)
        verifier_client = make("verifier", self.clients.verifier)
        aggregator_client = make("aggregator", self.clients.aggregator)
        router_label_client = make("router_label", self.clients.router_label)
        components = PipelineComponents(
            primary_clients=primary,  # type: ignore[arg-type]
            role_client=role,
            aggregator=EvidenceGatedAggregator(aggregator_client),
            router=RecallFirstRouter(label_client=router_label_client),
            verifier=FullDocumentVerifier(verifier_client) if verifier_client is not None else None,
            max_rounds=self.debate_rounds,
        )
        return PipelineRuntime(build_pipeline(self.pipeline, components), tuple(tracked))


def load_config(path: Path | str) -> AppConfig:
    """Load a strictly shaped JSON configuration without reading any credential value."""
    try:
        raw = json.loads(
            Path(path).read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"configuration could not be loaded: {exc.__class__.__name__}") from exc
    if not isinstance(raw, dict):
        raise ConfigurationError("configuration root must be an object")
    _validate_finite_json(raw)
    _strict_fields(raw, {"pipeline", "seed", "debate_rounds", "clients"}, "root")
    pipeline = _required_string(raw, "pipeline").casefold()
    if pipeline not in _PIPELINES:
        raise ConfigurationError("unknown pipeline")
    seed = _required_integer(raw, "seed")
    debate_rounds = _required_integer(raw, "debate_rounds")
    if not 0 <= debate_rounds <= 2:
        raise ConfigurationError("debate_rounds must be between 0 and 2")
    clients_raw = raw.get("clients")
    if not isinstance(clients_raw, dict):
        raise ConfigurationError("clients must be an object")
    _strict_fields(clients_raw, {"primary", "role", "verifier", "aggregator", "router_label"}, "clients")
    primary_raw = clients_raw.get("primary", [])
    if not isinstance(primary_raw, list):
        raise ConfigurationError("clients.primary must be an array")
    clients = ClientSetConfig(
        primary=tuple(_parse_client(value) for value in primary_raw),
        role=_parse_optional_client(clients_raw.get("role")),
        verifier=_parse_optional_client(clients_raw.get("verifier")),
        aggregator=_parse_optional_client(clients_raw.get("aggregator")),
        router_label=_parse_optional_client(clients_raw.get("router_label")),
    )
    config = AppConfig(pipeline, seed, debate_rounds, clients)
    _validate_app_config(config)
    return config


def _parse_optional_client(value: Any) -> ClientConfig | None:
    return None if value is None else _parse_client(value)


def _parse_client(value: Any) -> ClientConfig:
    if not isinstance(value, dict):
        raise ConfigurationError("client entries must be objects")
    provider = _required_string(value, "provider").casefold()
    if provider == "openai_compatible":
        _strict_fields(value, {"provider", "base_url", "model", "api_key_env", "timeout"}, "client")
        timeout = value.get("timeout", 60.0)
        normalized_timeout = _finite_positive_number(timeout, "client timeout")
        model = _required_string(value, "model")
        _validate_model_name(model)
        return ClientConfig(
            provider=provider,
            base_url=_required_string(value, "base_url"),
            model=model,
            api_key_env=_required_environment_name(value, "api_key_env"),
            timeout=normalized_timeout,
        )
    if provider == "scripted":
        _strict_fields(value, {"provider", "model", "script"}, "client")
        script = value.get("script")
        if not isinstance(script, list) or not all(isinstance(entry, dict) for entry in script):
            raise ConfigurationError("scripted client script must be an array of objects")
        model = _required_string(value, "model")
        _validate_model_name(model)
        return ClientConfig(provider=provider, model=model, script=tuple(dict(item) for item in script))
    raise ConfigurationError("unsupported client provider")


def _validate_topology(pipeline: str, clients: ClientSetConfig) -> None:
    primary_count = len(clients.primary)
    if pipeline in {"p1", "p4"} and primary_count != 1:
        raise ConfigurationError(f"{pipeline} requires exactly one primary client")
    if pipeline in {"p2", "p5"} and primary_count < 2:
        raise ConfigurationError(f"{pipeline} requires at least two primary clients")
    if pipeline in {"p3", "p6"} and clients.role is None:
        raise ConfigurationError(f"{pipeline} requires a role client")
    if pipeline in {"p4", "p5", "p6"} and clients.verifier is None:
        raise ConfigurationError(f"{pipeline} requires a verifier client")
    configured_roles: set[str] = set()
    if primary_count:
        configured_roles.add("primary")
    for role, client in (
        ("role", clients.role),
        ("verifier", clients.verifier),
        ("aggregator", clients.aggregator),
        ("router_label", clients.router_label),
    ):
        if client is not None:
            configured_roles.add(role)
    irrelevant = configured_roles - _CLIENT_ROLES[pipeline]
    if irrelevant:
        raise ConfigurationError(f"{pipeline} has irrelevant client roles: {len(irrelevant)}")


def _validate_app_config(config: AppConfig) -> None:
    if not isinstance(config.pipeline, str) or config.pipeline not in _PIPELINES:
        raise ConfigurationError("unknown pipeline")
    if isinstance(config.seed, bool) or not isinstance(config.seed, int):
        raise ConfigurationError("seed must be an integer")
    if (
        isinstance(config.debate_rounds, bool)
        or not isinstance(config.debate_rounds, int)
        or not 0 <= config.debate_rounds <= 2
    ):
        raise ConfigurationError("debate_rounds must be between 0 and 2")
    if not isinstance(config.clients, ClientSetConfig):
        raise ConfigurationError("clients must use ClientSetConfig")
    recipes = (
        *config.clients.primary,
        *tuple(
            recipe
            for recipe in (
                config.clients.role,
                config.clients.verifier,
                config.clients.aggregator,
                config.clients.router_label,
            )
            if recipe is not None
        ),
    )
    if not all(isinstance(recipe, ClientConfig) for recipe in recipes):
        raise ConfigurationError("clients must contain ClientConfig entries")
    for recipe in recipes:
        _validate_client_config(recipe)
    _validate_topology(config.pipeline, config.clients)


def _validate_client_config(config: ClientConfig) -> None:
    _validate_model_name(config.model)
    if config.provider == "openai_compatible":
        if not isinstance(config.base_url, str) or not config.base_url.strip():
            raise ConfigurationError("base_url must be a non-empty string")
        if not isinstance(config.api_key_env, str) or _ENVIRONMENT_NAME.fullmatch(config.api_key_env) is None:
            raise ConfigurationError("api_key_env must be an environment identifier")
        _finite_positive_number(config.timeout, "client timeout")
        if config.script:
            raise ConfigurationError("openai clients must not contain a script")
        return
    if config.provider == "scripted":
        if config.base_url is not None or config.api_key_env is not None:
            raise ConfigurationError("scripted clients must not contain provider connection fields")
        if not isinstance(config.script, tuple) or not all(isinstance(item, Mapping) for item in config.script):
            raise ConfigurationError("scripted client script must contain mappings")
        for item in config.script:
            _validate_finite_json(dict(item))
        return
    raise ConfigurationError("unsupported client provider")


def _strict_fields(value: Mapping[str, Any], allowed: set[str], location: str) -> None:
    unknown = set(value) - allowed
    missing = allowed - set(value) if location == "root" else set()
    if unknown:
        raise ConfigurationError(f"unknown fields in {location}: {len(unknown)}")
    if missing:
        raise ConfigurationError(f"missing fields in {location}: {len(missing)}")


def _required_string(value: Mapping[str, Any], field: str) -> str:
    item = value.get(field)
    if not isinstance(item, str) or not item.strip():
        raise ConfigurationError(f"{field} must be a non-empty string")
    return item.strip()


def _required_integer(value: Mapping[str, Any], field: str) -> int:
    item = value.get(field)
    if isinstance(item, bool) or not isinstance(item, int):
        raise ConfigurationError(f"{field} must be an integer")
    return item


def _finite_positive_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{field} must be positive")
    try:
        normalized = float(value)
    except (OverflowError, ValueError) as exc:
        raise ConfigurationError(f"{field} must be finite") from exc
    if not math.isfinite(normalized) or normalized <= 0:
        raise ConfigurationError(f"{field} must be finite and positive")
    return normalized


def _required_environment_name(value: Mapping[str, Any], field: str) -> str:
    item = value.get(field)
    if not isinstance(item, str) or _ENVIRONMENT_NAME.fullmatch(item) is None:
        raise ConfigurationError(f"{field} must be an environment identifier")
    return item


def _validate_model_name(model: Any) -> None:
    if not isinstance(model, str) or _MODEL_NAME.fullmatch(model) is None:
        raise ConfigurationError("model must be a compact identifier")


def _reject_json_constant(_value: str) -> None:
    raise ConfigurationError("configuration numbers must be finite JSON numbers")


def _validate_finite_json(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ConfigurationError("configuration numbers must be finite JSON numbers")
    if isinstance(value, dict):
        for item in value.values():
            _validate_finite_json(item)
    elif isinstance(value, list):
        for item in value:
            _validate_finite_json(item)


__all__ = ["AppConfig", "ClientConfig", "ClientSetConfig", "ConfigurationError", "PipelineRuntime", "load_config"]
