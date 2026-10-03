"""Strip LLM think tags from text (closed, unclosed, and orphan forms).

Single source of truth for think-tag cleaning, introduced for query-rewrite
context assembly (see docs/internal-notes/20260929-query-rewrite-context-injection-plan.md,
2026-10-01 补充 R3).

Existing call sites (llm_client.extract_response_text, memory_add.strip_thinking)
are intentionally NOT refactored onto this helper: their current behavior is
part of the LOCOMO anchor measurement chain, and changing what gets stripped
there would break score comparability for the §5.1 production gate.
"""

import re

_CLOSED_THINK_RE = re.compile(r"<think\b[^>]*>.*?</think(?:ing)?\s*>", re.DOTALL)
_UNCLOSED_THINK_RE = re.compile(r"<think\b[^>]*>.*", re.DOTALL)
_ORPHAN_CLOSE_RE = re.compile(r"^.*?</think(?:ing)?\s*>", re.DOTALL)


def strip_think_tags(text: str) -> str:
    """Remove think-tag segments from ``text``.

    Handles three forms:
    - closed pairs ``<think ...>...</think>`` (and the ``</thinking>`` variant)
    - unclosed ``<think ...>`` (truncated output) — stripped to end of text
    - orphan ``</think>`` (head truncated away) — stripped from start through
      the tag, keeping the visible text after it
    """
    if not text or "think" not in text:
        return (text or "").strip()
    out = _CLOSED_THINK_RE.sub("", text)
    out = _UNCLOSED_THINK_RE.sub("", out)
    out = _ORPHAN_CLOSE_RE.sub("", out)
    return out.strip()
