from uuid import uuid4

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from agentobs.integrations.langchain import OTelCallbackHandler


def _llm_result(input_tokens=500, output_tokens=80, model="llama-3.3-70b-versatile"):
    message = AIMessage(
        content="done",
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
        response_metadata={"model_name": model},
    )
    return LLMResult(generations=[[ChatGeneration(message=message)]])


def test_llm_span_with_usage_and_cost(exporter):
    handler = OTelCallbackHandler()
    run_id = uuid4()
    handler.on_chat_model_start(
        {}, [], run_id=run_id, parent_run_id=None,
        invocation_params={"model": "llama-3.3-70b-versatile"},
    )
    handler.on_llm_end(_llm_result(), run_id=run_id)

    (span,) = exporter.get_finished_spans()
    assert span.name == "llm llama-3.3-70b-versatile"
    assert span.attributes["gen_ai.usage.input_tokens"] == 500
    assert span.attributes["gen_ai.usage.output_tokens"] == 80
    assert span.attributes["agentobs.cost.usd"] > 0


def test_tool_span_parented_under_llm_parent(exporter):
    handler = OTelCallbackHandler()
    parent_id, child_id = uuid4(), uuid4()
    handler.on_chat_model_start({}, [], run_id=parent_id, parent_run_id=None,
                                invocation_params={"model": "m"})
    handler.on_tool_start({"name": "check_geo_feasibility"}, "{}",
                          run_id=child_id, parent_run_id=parent_id)
    handler.on_tool_end("ok", run_id=child_id)
    handler.on_llm_end(_llm_result(), run_id=parent_id)

    spans = {s.name: s for s in exporter.get_finished_spans()}
    tool = spans["tool check_geo_feasibility"]
    llm = spans["llm m"]
    assert tool.parent.span_id == llm.context.span_id


def test_tool_error_marks_span(exporter):
    handler = OTelCallbackHandler()
    run_id = uuid4()
    handler.on_tool_start({"name": "boom"}, "{}", run_id=run_id, parent_run_id=None)
    handler.on_tool_error(RuntimeError("kaput"), run_id=run_id)

    (span,) = exporter.get_finished_spans()
    assert not span.status.is_ok
    assert span.events[0].name == "exception"


def test_unmatched_end_is_ignored(exporter):
    handler = OTelCallbackHandler()
    handler.on_llm_end(_llm_result(), run_id=uuid4())  # no start — no crash
    assert exporter.get_finished_spans() == ()
