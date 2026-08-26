"""Construction of named P1-P6 experimental pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..aggregation import EvidenceGatedAggregator
from ..llm import LLMClient
from .base import Pipeline
from .full import P1Pipeline, P2Pipeline, P3Pipeline
from .routed import P4Pipeline, P5Pipeline, P6Pipeline


@dataclass(frozen=True)
class PipelineComponents:
    primary_clients: tuple[LLMClient, ...] = ()
    role_client: LLMClient | None = None
    aggregator: EvidenceGatedAggregator | None = None
    router: Any | None = None
    verifier: Any | None = None
    max_rounds: int = 2


def build_pipeline(name: str, components: PipelineComponents) -> Pipeline:
    """Build exactly the requested topology or reject incomplete components."""
    normalized = name.strip().casefold()
    if normalized == "p1":
        return P1Pipeline(_first_primary(components))
    if normalized == "p2":
        return P2Pipeline(components.primary_clients, components.aggregator)
    if normalized == "p3":
        return P3Pipeline(
            _role_client(components), max_rounds=components.max_rounds, aggregator=components.aggregator
        )
    if normalized == "p4":
        return P4Pipeline(_first_primary(components), components.router, components.verifier)
    if normalized == "p5":
        return P5Pipeline(
            components.primary_clients, components.router, components.verifier, components.aggregator
        )
    if normalized == "p6":
        return P6Pipeline(
            _role_client(components),
            components.router,
            components.verifier,
            max_rounds=components.max_rounds,
            aggregator=components.aggregator,
        )
    raise ValueError(f"unknown pipeline: {name}")


def _first_primary(components: PipelineComponents) -> LLMClient:
    if not components.primary_clients:
        raise ValueError("at least one primary client is required")
    return components.primary_clients[0]


def _role_client(components: PipelineComponents) -> LLMClient:
    if components.role_client is None:
        raise ValueError("a role client is required")
    return components.role_client
