"""Span helpers for manual agent loops. Attribute names follow the OTel
GenAI semantic conventions (gen_ai.*), with agentobs.* for our extensions.

    with agent_turn("triage", session_id=...):
        with llm_call("claude-opus-4-8") as span:
            response = client.messages.create(...)
            record_llm_usage(span, "claude-opus-4-8",
                             response.usage.input_tokens,
                             response.usage.output_tokens)
        with tool_call("get_customer_profile"):
            result = ...
"""

from contextlib import contextmanager

from opentelemetry import trace

from .cost import estimate_cost
from .tracing import get_tracer


@contextmanager
def agent_turn(name: str, **attributes):
    """Root span for one user turn / one pipeline run."""
    with get_tracer().start_as_current_span(f"agent_turn {name}") as span:
        span.set_attribute("gen_ai.operation.name", "agent_turn")
        span.set_attribute("agentobs.turn.name", name)
        for key, value in attributes.items():
            span.set_attribute(f"agentobs.{key}", value)
        yield span


@contextmanager
def tool_call(name: str, **attributes):
    with get_tracer().start_as_current_span(f"tool {name}") as span:
        span.set_attribute("gen_ai.operation.name", "execute_tool")
        span.set_attribute("gen_ai.tool.name", name)
        for key, value in attributes.items():
            span.set_attribute(f"agentobs.{key}", value)
        yield span


@contextmanager
def llm_call(model: str, **attributes):
    """Span for one model request. Call record_llm_usage() before exit."""
    with get_tracer().start_as_current_span(f"llm {model}") as span:
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.request.model", model)
        for key, value in attributes.items():
            span.set_attribute(f"agentobs.{key}", value)
        yield span


def record_llm_usage(span: trace.Span, model: str,
                     input_tokens: int = 0, output_tokens: int = 0,
                     cache_read_tokens: int = 0, cache_write_tokens: int = 0) -> None:
    """Attach token usage + estimated USD cost to an LLM span."""
    span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
    span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
    if cache_read_tokens:
        span.set_attribute("gen_ai.usage.cache_read_tokens", cache_read_tokens)
    if cache_write_tokens:
        span.set_attribute("gen_ai.usage.cache_write_tokens", cache_write_tokens)
    cost = estimate_cost(model, input_tokens, output_tokens,
                         cache_read_tokens, cache_write_tokens)
    if cost is not None:
        span.set_attribute("agentobs.cost.usd", cost)
