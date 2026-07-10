"""LangChain / LangGraph adapter: an OTel callback handler.

Zero-code-change instrumentation for any LangChain runnable or LangGraph
graph — pass the handler via config:

    from agentobs import init_tracing
    from agentobs.integrations.langchain import OTelCallbackHandler

    init_tracing("my-agent", exporter="otlp")
    graph.invoke(state, config={"callbacks": [OTelCallbackHandler()]})

Emits `llm <model>` spans (with token usage + cost from usage_metadata)
and `tool <name>` spans, parented by run_id so nesting mirrors the
actual execution tree.
"""

from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from opentelemetry import context as otel_context
from opentelemetry import trace

from ..cost import estimate_cost
from ..tracing import get_tracer


class OTelCallbackHandler(BaseCallbackHandler):
    run_inline = True  # keep OTel context propagation on the caller thread

    def __init__(self):
        self._spans: dict[UUID, trace.Span] = {}

    # -- internals ---------------------------------------------------------

    def _start(self, run_id: UUID, parent_run_id: UUID | None, name: str) -> trace.Span:
        parent = self._spans.get(parent_run_id) if parent_run_id else None
        ctx = trace.set_span_in_context(parent) if parent else otel_context.get_current()
        span = get_tracer().start_span(name, context=ctx)
        self._spans[run_id] = span
        return span

    def _end(self, run_id: UUID, error: BaseException | None = None) -> None:
        span = self._spans.pop(run_id, None)
        if span is None:
            return
        if error is not None:
            span.set_status(trace.StatusCode.ERROR, str(error))
            span.record_exception(error)
        span.end()

    @staticmethod
    def _model_from(serialized: dict | None, kwargs: dict) -> str:
        params = (kwargs.get("invocation_params") or {})
        for key in ("model", "model_name", "model_id"):
            if params.get(key):
                return str(params[key])
        if serialized:
            for key in ("model", "model_name"):
                value = (serialized.get("kwargs") or {}).get(key)
                if value:
                    return str(value)
        return "unknown"

    # -- LLM lifecycle ------------------------------------------------------

    def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, **kwargs):
        model = self._model_from(serialized, kwargs)
        span = self._start(run_id, parent_run_id, f"llm {model}")
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.request.model", model)

    def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id=None, **kwargs):
        self.on_chat_model_start(serialized, prompts, run_id=run_id,
                                 parent_run_id=parent_run_id, **kwargs)

    def on_llm_end(self, response, *, run_id, **kwargs):
        span = self._spans.get(run_id)
        if span is not None:
            usage, model = self._usage_from(response)
            if usage:
                input_tokens = usage.get("input_tokens", 0)
                output_tokens = usage.get("output_tokens", 0)
                span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
                span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
                if model:
                    span.set_attribute("gen_ai.response.model", model)
                cost = estimate_cost(model or "", input_tokens, output_tokens)
                if cost is not None:
                    span.set_attribute("agentobs.cost.usd", cost)
        self._end(run_id)

    def on_llm_error(self, error, *, run_id, **kwargs):
        self._end(run_id, error)

    @staticmethod
    def _usage_from(response) -> tuple[dict | None, str | None]:
        """Token usage + response model from an LLMResult, tolerating the
        provider-specific places LangChain puts them."""
        model = None
        llm_output = getattr(response, "llm_output", None) or {}
        model = llm_output.get("model") or llm_output.get("model_name")
        try:
            generation = response.generations[0][0]
            message = getattr(generation, "message", None)
            usage = getattr(message, "usage_metadata", None)
            if usage:
                return dict(usage), model or (message.response_metadata or {}).get("model_name")
        except (IndexError, AttributeError):
            pass
        usage = llm_output.get("token_usage") or llm_output.get("usage")
        if usage:
            return {
                "input_tokens": usage.get("prompt_tokens", usage.get("input_tokens", 0)),
                "output_tokens": usage.get("completion_tokens", usage.get("output_tokens", 0)),
            }, model
        return None, model

    # -- Tool lifecycle -----------------------------------------------------

    def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None, **kwargs):
        name = (serialized or {}).get("name", "tool")
        span = self._start(run_id, parent_run_id, f"tool {name}")
        span.set_attribute("gen_ai.operation.name", "execute_tool")
        span.set_attribute("gen_ai.tool.name", name)

    def on_tool_end(self, output: Any, *, run_id, **kwargs):
        self._end(run_id)

    def on_tool_error(self, error, *, run_id, **kwargs):
        self._end(run_id, error)
