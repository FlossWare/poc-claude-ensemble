"""
Pricing constants for Claude models and integration partners.
Updated: 2026-09-25
Source: official Claude pricing and integration partners
"""

CLAUDE_PRICING = {
    "haiku": {"input": 0.80, "output": 2.40, "name": "Claude Haiku 4.5"},
    "sonnet": {"input": 3.00, "output": 15.00, "name": "Claude Sonnet 4.5"},
    "opus": {"input": 15.00, "output": 45.00, "name": "Claude Opus 5"},
    "opus-4.8": {"input": 12.00, "output": 36.00, "name": "Claude Opus 4.8"},
}

GEMINI_PRICING = {
    "input": 0.075,
    "output": 0.30,
    "name": "Google Gemini",
}

GPT_PRICING = {
    "gpt-4o": {"input": 5.00, "output": 15.00, "name": "GPT-4o"},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00, "name": "GPT-4 Turbo"},
}

# Gateway-served models with no published rates. They are assumed to cost as much as the most
# expensive Claude tier (Opus), so their cost is never recorded as zero. Replace with real rates
# when the provider publishes them.
GATEWAY_ASSUMED_PRICING = {
    "rits/zai-org/glm-5-3": dict(CLAUDE_PRICING["opus"], name="GLM 5.3 (assumed Opus rate)"),
    "rits/deepseek-ai/deepseek-v4.1-flash-1017": dict(
        CLAUDE_PRICING["opus"], name="DeepSeek V4.1 Flash (assumed Opus rate)"),
}


def _resolve_pricing(model: str) -> dict:
    model_lower = model.lower()
    if model_lower in CLAUDE_PRICING:
        return CLAUDE_PRICING[model_lower]
    if model_lower in GATEWAY_ASSUMED_PRICING:
        return GATEWAY_ASSUMED_PRICING[model_lower]
    if model_lower == "gemini" or model_lower.startswith("gemini-") or model_lower.startswith("models/gemini-"):
        return GEMINI_PRICING
    if model_lower.startswith("claude-"):
        if "haiku" in model_lower:
            return CLAUDE_PRICING["haiku"]
        if "sonnet" in model_lower:
            return CLAUDE_PRICING["sonnet"]
        if "opus-4.8" in model_lower:
            return CLAUDE_PRICING["opus-4.8"]
        if "opus" in model_lower:
            return CLAUDE_PRICING["opus"]
    raise ValueError(f"Unknown model: {model}. Known models: {list(CLAUDE_PRICING.keys())}")


# Prompt-cache multipliers relative to the base input rate: writes cost 1.25x, reads cost 0.1x.
CACHE_WRITE_MULTIPLIER = 1.25
CACHE_READ_MULTIPLIER = 0.1


def calculate_cost(model: str, input_tokens: int, output_tokens: int,
                   cache_write_tokens: int = 0, cache_read_tokens: int = 0) -> float:
    """Calculate an estimated cost for a supported model family, including prompt-cache tokens."""
    pricing = _resolve_pricing(model)
    return (
        (input_tokens / 1_000_000) * pricing["input"]
        + (output_tokens / 1_000_000) * pricing["output"]
        + (cache_write_tokens / 1_000_000) * pricing["input"] * CACHE_WRITE_MULTIPLIER
        + (cache_read_tokens / 1_000_000) * pricing["input"] * CACHE_READ_MULTIPLIER
    )


def get_model_display_name(model: str) -> str:
    """Get display name for a model."""
    model_lower = model.lower()
    if model_lower in CLAUDE_PRICING:
        return CLAUDE_PRICING[model_lower]["name"]
    if model_lower.startswith(("claude-", "gemini-", "models/gemini-")):
        return model
    return model
