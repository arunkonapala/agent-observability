"""Per-model USD cost estimation from token usage.

Prices are USD per million tokens (input, output). Cache reads are billed
at ~10% of input price on Anthropic models; cache writes at 1.25x.
Extend at runtime with register_pricing().
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Pricing:
    input_per_mtok: float
    output_per_mtok: float
    cache_read_multiplier: float = 0.1
    cache_write_multiplier: float = 1.25


_PRICING: dict[str, Pricing] = {
    # Anthropic (per platform pricing, mid-2026)
    "claude-fable-5": Pricing(10.00, 50.00),
    "claude-opus-4-8": Pricing(5.00, 25.00),
    "claude-opus-4-7": Pricing(5.00, 25.00),
    "claude-opus-4-6": Pricing(5.00, 25.00),
    "claude-sonnet-4-6": Pricing(3.00, 15.00),
    "claude-haiku-4-5": Pricing(1.00, 5.00),
    # Groq (on-demand pricing)
    "llama-3.3-70b-versatile": Pricing(0.59, 0.79),
    "llama-3.1-8b-instant": Pricing(0.05, 0.08),
}


def register_pricing(model: str, input_per_mtok: float, output_per_mtok: float,
                     **kwargs) -> None:
    _PRICING[model] = Pricing(input_per_mtok, output_per_mtok, **kwargs)


def _lookup(model: str) -> Pricing | None:
    if model in _PRICING:
        return _PRICING[model]
    # tolerate provider prefixes and date suffixes (anthropic.claude-…, -20251001)
    for known, pricing in _PRICING.items():
        if known in model:
            return pricing
    return None


def estimate_cost(model: str, input_tokens: int = 0, output_tokens: int = 0,
                  cache_read_tokens: int = 0, cache_write_tokens: int = 0) -> float | None:
    """Estimated USD cost for one LLM call, or None for unknown models."""
    pricing = _lookup(model)
    if pricing is None:
        return None
    cost = (
        input_tokens * pricing.input_per_mtok
        + output_tokens * pricing.output_per_mtok
        + cache_read_tokens * pricing.input_per_mtok * pricing.cache_read_multiplier
        + cache_write_tokens * pricing.input_per_mtok * pricing.cache_write_multiplier
    ) / 1_000_000
    return round(cost, 8)
