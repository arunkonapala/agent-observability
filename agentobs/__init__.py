"""agentobs — OpenTelemetry observability for LLM agent loops.

Spans per agent turn, tool call, and LLM request, with token usage and
USD cost attribution following the OTel GenAI semantic conventions.
"""

from .cost import estimate_cost, register_pricing
from .spans import agent_turn, llm_call, record_llm_usage, tool_call
from .tracing import get_tracer, init_tracing

__version__ = "0.1.0"

__all__ = [
    "init_tracing",
    "get_tracer",
    "agent_turn",
    "tool_call",
    "llm_call",
    "record_llm_usage",
    "estimate_cost",
    "register_pricing",
]
