"""Tracer setup. One call at app startup:

    from agentobs import init_tracing
    init_tracing("fraud-triage-agent")               # exporter from env
    init_tracing("finance-copilot", exporter="otlp") # explicit

Exporter is chosen by the AGENTOBS_EXPORTER env var unless passed:
    console  — spans pretty-printed to stdout
    otlp     — OTLP/HTTP to AGENTOBS_OTLP_ENDPOINT (default http://localhost:4318)
    memory   — in-memory, for tests (retrieve via get_memory_exporter())
    none     — no-op (default): zero overhead, no spans exported
"""

import os

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

_memory_exporter: InMemorySpanExporter | None = None
_provider: TracerProvider | None = None


def init_tracing(service_name: str, exporter: str | None = None) -> None:
    """Configure the global tracer provider. Safe to call once per process."""
    global _memory_exporter, _provider

    exporter = (exporter or os.getenv("AGENTOBS_EXPORTER", "none")).lower()
    if exporter == "none":
        return

    provider = TracerProvider(
        resource=Resource.create({"service.name": service_name})
    )

    if exporter == "console":
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    elif exporter == "otlp":
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )

        endpoint = os.getenv("AGENTOBS_OTLP_ENDPOINT", "http://localhost:4318")
        provider.add_span_processor(BatchSpanProcessor(
            OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces")
        ))
    elif exporter == "memory":
        _memory_exporter = InMemorySpanExporter()
        provider.add_span_processor(SimpleSpanProcessor(_memory_exporter))
    else:
        raise ValueError(f"unknown AGENTOBS_EXPORTER {exporter!r}")

    trace.set_tracer_provider(provider)
    _provider = provider


def get_tracer():
    return trace.get_tracer("agentobs")


def get_memory_exporter() -> InMemorySpanExporter | None:
    """The in-memory exporter when initialized with exporter='memory'."""
    return _memory_exporter


def flush() -> None:
    """Force-export pending spans (call before process exit with otlp)."""
    if _provider is not None:
        _provider.force_flush()
