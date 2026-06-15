"""LLM client utilities — vendor-neutral thinking control and response extraction.

This module centralizes model-specific ``extra_body`` parameters (e.g. MiniMax
``thinking``, DashScope/Qwen ``chat_template_kwargs``, DeepSeek ``thinking``)
so that callers do not scatter vendor-specific hacks across the codebase.
"""

import re
from typing import Optional


def build_thinking_extra(model: Optional[str], enable: bool) -> dict:
    """Return the appropriate extra_body parameters to control reasoning/thinking.

    Args:
        model: Model identifier (e.g. "MiniMax-M3", "qwen-max-latest",
            "deepseek-v4-pro"). Case-insensitive.
        enable: Whether to enable reasoning/thinking mode.

    Returns:
        A dict suitable for ``openai_client.chat.completions.create(extra_body=...)``.
        Returns an empty dict for unknown providers to avoid sending unsupported keys.
    """
    model = (model or "").lower()

    if "minimax" in model:
        return {"thinking": {"type": "adaptive" if enable else "disabled"}}

    if "qwen" in model:
        return {"chat_template_kwargs": {"enable_thinking": enable}}

    if "deepseek" in model:
        return {"thinking": {"type": "enabled" if enable else "disabled"}}

    # Unknown provider: do not send any thinking-related parameters.
    return {}


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
