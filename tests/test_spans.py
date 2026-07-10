from agentobs import agent_turn, llm_call, record_llm_usage, tool_call


def test_span_hierarchy_and_attributes(exporter):
    with agent_turn("triage", session_id="s1"):
        with llm_call("claude-opus-4-8") as span:
            record_llm_usage(span, "claude-opus-4-8",
                             input_tokens=1200, output_tokens=300,
                             cache_read_tokens=5000)
        with tool_call("get_customer_profile"):
            pass

    spans = {s.name: s for s in exporter.get_finished_spans()}
    assert set(spans) == {"agent_turn triage", "llm claude-opus-4-8",
                          "tool get_customer_profile"}

    root = spans["agent_turn triage"]
    llm = spans["llm claude-opus-4-8"]
    tool = spans["tool get_customer_profile"]

    # nesting: llm and tool are children of the turn
    assert llm.parent.span_id == root.context.span_id
    assert tool.parent.span_id == root.context.span_id

    assert root.attributes["agentobs.session_id"] == "s1"
    assert llm.attributes["gen_ai.request.model"] == "claude-opus-4-8"
    assert llm.attributes["gen_ai.usage.input_tokens"] == 1200
    assert llm.attributes["gen_ai.usage.output_tokens"] == 300
    assert llm.attributes["gen_ai.usage.cache_read_tokens"] == 5000
    expected_cost = (1200 * 5 + 300 * 25 + 5000 * 5 * 0.1) / 1e6
    assert abs(llm.attributes["agentobs.cost.usd"] - expected_cost) < 1e-9
    assert tool.attributes["gen_ai.tool.name"] == "get_customer_profile"


def test_unknown_model_omits_cost(exporter):
    with llm_call("mystery-model") as span:
        record_llm_usage(span, "mystery-model", 100, 100)
    span = exporter.get_finished_spans()[0]
    assert "agentobs.cost.usd" not in span.attributes
    assert span.attributes["gen_ai.usage.input_tokens"] == 100
