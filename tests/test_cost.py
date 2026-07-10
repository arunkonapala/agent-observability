from agentobs import estimate_cost, register_pricing


def test_claude_opus_pricing():
    # 1M input + 1M output at $5/$25
    assert estimate_cost("claude-opus-4-8", 1_000_000, 1_000_000) == 30.0


def test_groq_llama_pricing():
    cost = estimate_cost("llama-3.3-70b-versatile", 10_000, 2_000)
    assert cost == round((10_000 * 0.59 + 2_000 * 0.79) / 1e6, 8)


def test_cache_read_discount():
    # cache reads bill at 10% of input price
    full = estimate_cost("claude-sonnet-4-6", input_tokens=100_000)
    cached = estimate_cost("claude-sonnet-4-6", cache_read_tokens=100_000)
    assert cached == round(full * 0.1, 8)


def test_unknown_model_returns_none():
    assert estimate_cost("mystery-model-9000", 1000, 1000) is None


def test_prefix_and_suffix_tolerance():
    # Bedrock-style prefix and dated suffix still resolve
    assert estimate_cost("anthropic.claude-opus-4-8", 1000, 0) is not None
    assert estimate_cost("claude-haiku-4-5-20251001", 1000, 0) is not None


def test_register_custom_pricing():
    register_pricing("my-fine-tune", 1.0, 2.0)
    assert estimate_cost("my-fine-tune", 1_000_000, 0) == 1.0
