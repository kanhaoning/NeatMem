"""LLM client utilities — vendor-neutral thinking control and response extraction.

This module centralizes model-specific ``extra_body`` parameters (e.g. MiniMax
``thinking``, DashScope/Qwen ``chat_template_kwargs``, DeepSeek ``thinking``)
so that callers do not scatter vendor-specific hacks across the codebase.

Multi-provider support (2026-08): an explicit provider name selects the
verified parameter shape from PROVIDER_TABLE; when no provider is given the
legacy model-name matching is used unchanged. Every table entry is backed by
smoke-test evidence (see tmp/20260808-provider_compat/REPORT.md).
"""

import re
from typing import Optional

# ---------------------------------------------------------------------------
# Provider table
# ---------------------------------------------------------------------------
# thinking shapes (all verified against live APIs, evidence in the smoke pack):
#   "type"             -> extra_body={"thinking": {"type": "enabled"/"disabled"}}
#   "ctk"              -> extra_body={"chat_template_kwargs": {"enable_thinking": bool}}
#   "dual"             -> both of the above (MiniMax legacy; ctk is inert on M3
#                         but kept for byte-identical backward compatibility)
#   "reasoning_effort" -> standard request param (Gemini); OFF -> "none",
#                         ON -> omit (server default thinks)
#   "passthrough"      -> send nothing (OpenRouter normalizes upstream)
#   None               -> no thinking switch (OpenAI; reasoning params filtered
#                         via apply_provider_constraints instead)
#
# constraints:
#   "drop_temperature" -> Moonshot pins temperature per thinking state (400 on
#                         any other explicit value); omitting temperature is
#                         accepted in all states, so we drop it.
PROVIDER_TABLE = {
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "thinking": "type",
    },
    "dashscope": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "thinking": "ctk",
    },
    "zhipu": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "thinking": "type",  # ctk shape is silently swallowed -- do not use
    },
    "moonshot": {
        "base_url": "https://api.moonshot.cn/v1",
        "thinking": "type",
        "drop_temperature": True,
    },
    "volcengine": {
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "thinking": "type",  # OFF moves CoT into content; keep ON for JSON paths
    },
    "minimax": {
        "base_url": "https://api.minimaxi.com/v1",
        "thinking": "dual",  # M2.5 has no off switch at all (reasoning-only)
    },
    "siliconflow": {
        "base_url": "https://api.siliconflow.cn/v1",
        "thinking": "type",  # verified for hosted GLM-5.1/Kimi-K2.6/Qwen3.5;
        # hosted MiniMax-M2.5 cannot disable thinking (no leak though)
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "thinking": None,
        "reasoning_filter": True,
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "thinking": "reasoning_effort",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "thinking": "passthrough",
    },
}

# Brand aliases -> canonical platform names (rule: canonical = platform where
# the API key is issued; aliases = model brand names users may type instead).
PROVIDER_ALIASES = {
    "qwen": "dashscope",
    "glm": "zhipu",
    "zai": "zhipu",
    "kimi": "moonshot",
    "doubao": "volcengine",
    "ark": "volcengine",
}


def normalize_provider(provider: Optional[str]) -> Optional[str]:
    """Map a user-supplied provider string to its canonical table key.

    Returns None for None/empty input (legacy model-name fallback). Raises
    ValueError listing valid values for unknown providers -- a typo here must
    fail loudly at startup, not silently fall back to uncontrolled behavior.
    """
    if not provider:
        return None
    p = provider.strip().lower()
    p = PROVIDER_ALIASES.get(p, p)
    if p not in PROVIDER_TABLE:
        valid = sorted(set(PROVIDER_TABLE) | set(PROVIDER_ALIASES))
        raise ValueError(
            f"Unknown LLM provider {provider!r}. Valid values: {', '.join(valid)}. "
            "For unlisted OpenAI-compatible endpoints, leave LLM_PROVIDER unset "
            "and configure OPENAI_BASE_URL directly."
        )
    return p


def provider_default_base_url(provider: Optional[str]) -> Optional[str]:
    """Default endpoint for an explicit provider; None when unset/unknown."""
    p = normalize_provider(provider)
    return PROVIDER_TABLE[p]["base_url"] if p else None


def build_thinking_extra(model: Optional[str], enable: bool,
                         provider: Optional[str] = None) -> dict:
    """Return the appropriate extra_body parameters to control reasoning/thinking.

    Args:
        model: Model identifier (e.g. "MiniMax-M3", "qwen-max-latest",
            "deepseek-v4-pro"). Case-insensitive.
        enable: Whether to enable reasoning/thinking mode.
        provider: Explicit provider name (see PROVIDER_TABLE). When given, the
            verified per-provider shape is used. When None, falls back to the
            legacy model-name matching below, unchanged.

    Returns:
        A dict suitable for ``openai_client.chat.completions.create(extra_body=...)``.
        Returns an empty dict for unknown providers to avoid sending unsupported keys.
    """
    p = normalize_provider(provider)
    if p is not None:
        shape = PROVIDER_TABLE[p]["thinking"]
        if shape == "type":
            return {"thinking": {"type": "enabled" if enable else "disabled"}}
        if shape == "ctk":
            return {"chat_template_kwargs": {"enable_thinking": enable}}
        if shape == "dual":
            return {
                "chat_template_kwargs": {"enable_thinking": enable},
                "thinking": {"type": "adaptive" if enable else "disabled"},
            }
        # reasoning_effort (request-level param, not extra_body), passthrough,
        # and None all produce no extra_body.
        return {}

    model = (model or "").lower()

    if "minimax" in model:
        # MiniMax uses both the thinking field and chat_template_kwargs to
        # reliably control reasoning mode.
        return {
            "chat_template_kwargs": {"enable_thinking": enable},
            "thinking": {"type": "adaptive" if enable else "disabled"},
        }

    if "qwen" in model:
        return {"chat_template_kwargs": {"enable_thinking": enable}}

    if "deepseek" in model:
        return {"thinking": {"type": "enabled" if enable else "disabled"}}

    # Unknown provider: do not send any thinking-related parameters.
    return {}


# ---------------------------------------------------------------------------
# Provider constraints (request-level parameters)
# ---------------------------------------------------------------------------
# The reasoning-model detection below is ported from mem0
# (https://github.com/mem0ai/mem0), mem0/llms/base.py (_is_reasoning_model /
# _uses_max_completion_tokens / _get_supported_params).
# Copyright (c) Mem0 — Apache License, Version 2.0.
# Modifications: free functions instead of LLMBase methods; config override
# replaced by an explicit argument; only the parameter-filtering core is kept.

_REASONING_MODELS = {
    "o1", "o1-preview", "o3-mini", "o3",
    "gpt-5", "gpt-5o", "gpt-5o-mini", "gpt-5o-micro",
}


def is_reasoning_model(model: str, explicit: Optional[bool] = None) -> bool:
    """True for OpenAI reasoning models / GPT-5 series that reject sampling params.

    An explicit ``is_reasoning_model`` value takes precedence over the
    name-based heuristic (custom/versioned deployment names may not match).
    """
    if explicit is not None:
        return explicit
    base_model = (model or "").lower().rsplit("/", 1)[-1]
    if base_model in _REASONING_MODELS:
        return True
    # o1/o3 family with suffixes (o1-2024-12-17), NOT gpt-5.x variants
    return any(base_model.startswith(p) for p in ("o1-", "o1.", "o3-", "o3."))


def uses_max_completion_tokens(model: str) -> bool:
    """True for the GPT-5 family, which requires max_completion_tokens."""
    base_model = (model or "").lower().rsplit("/", 1)[-1]
    return base_model.startswith("gpt-5")


def apply_provider_constraints(params: dict, model: Optional[str],
                               provider: Optional[str] = None,
                               thinking_enable: Optional[bool] = None) -> dict:
    """Adjust request params for provider-specific constraints. Returns a new dict.

    No-op when provider is None (legacy path stays byte-identical):
    constraints are only applied for explicitly declared providers, since the
    legacy model-name path must not change behavior for existing users.

    Applied rules (each backed by smoke evidence):
      - moonshot: drop temperature (pinned per thinking state; explicit values
        other than the pin are a hard 400, omitting is always accepted).
      - openai reasoning models / GPT-5: drop temperature/top_p/top_k, rename
        max_tokens -> max_completion_tokens.
      - gemini: thinking OFF -> reasoning_effort="none" (request-level param);
        ON -> omit (server default thinks).
    """
    p = normalize_provider(provider)
    if p is None:
        return params

    params = dict(params)
    row = PROVIDER_TABLE[p]

    if row.get("drop_temperature"):
        params.pop("temperature", None)

    if row.get("reasoning_filter"):
        if is_reasoning_model(model or ""):
            for k in ("temperature", "top_p", "top_k"):
                params.pop(k, None)
        if uses_max_completion_tokens(model or "") and "max_tokens" in params:
            params["max_completion_tokens"] = params.pop("max_tokens")

    if row["thinking"] == "reasoning_effort" and thinking_enable is not None:
        if thinking_enable:
            params.pop("reasoning_effort", None)
        else:
            params["reasoning_effort"] = "none"

    return params


def complete_chat(client, model, messages, enable, provider=None, **params):
    """Single choke point for chat completions: thinking shape + constraints.

    Assembles extra_body via build_thinking_extra and request params via
    apply_provider_constraints, then issues the call. With provider=None both
    steps reproduce the legacy behavior exactly (legacy model-name extra_body,
    no constraint rewriting).
    """
    params["extra_body"] = build_thinking_extra(model, enable, provider)
    params = apply_provider_constraints(params, model, provider,
                                        thinking_enable=enable)
    return client.chat.completions.create(model=model, messages=messages, **params)


def extract_response_text(response) -> str:
    """Extract the final answer text from an OpenAI-compatible response.

    Only ``message.content`` is treated as the answer.  Reasoning content
    (``reasoning_content`` or ``<think>`` tags) is stripped to avoid leaking the
    model's internal reasoning into downstream prompts or JSON parsing.
    """
    msg = response.choices[0].message
    text = msg.content or ""
    text = re.sub(r"<think\b[^>]*>.*?</thinking\s*>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"<think\b[^>]*>.*?</think\s*>", "", text, flags=re.DOTALL).strip()
    return text
