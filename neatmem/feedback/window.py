"""Window slicing — turn the (messages, injections) streams into judgeable
question/answer windows.

Plan: docs/internal-notes/20261005-citation-feedback-memory-importance-plan.md
§4 (window semantics) and §5.4 (fine-grained messages).

Semantics (fixed by the case-gate decisions):
- An injection's window opens at its anchor message — the first message after
  the injection point (physically, injections only occur between LLM calls).
- The window closes at the NEXT injection's anchor (exclusive) or at the end
  of the message stream: whatever the assistant says before the next injection
  is this injection's judgment material.
- A window is judgeable only with a complete assistant turn: no assistant
  message, an empty answer after think-stripping (truncated mid-think), or an
  anchor that no longer resolves in the message stream all yield ``no_window``
  — detected here, never sent to the judge, and excluded from both counters.

This module is pure: no I/O, no config. The runner feeds it chronological
messages and injection events; tests construct both directly.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from neatmem.utils.think_tags import strip_think_tags

logger = logging.getLogger(__name__)

STATUS_OK = "ok"
STATUS_NO_WINDOW = "no_window"
STATUS_MISSING_ANCHOR = "missing_anchor"


@dataclass
class WindowOutcome:
    """Slicing result for one injection event."""

    injection_event_id: int
    status: str  # STATUS_OK | STATUS_NO_WINDOW | STATUS_MISSING_ANCHOR
    question: str = ""
    answer: str = ""
    detail: str = ""  # human-readable reason for non-ok statuses


def slice_windows(
    messages: List[Dict[str, Any]],
    injections: List[Dict[str, Any]],
) -> List[WindowOutcome]:
    """Slice one window per injection event.

    messages: chronological (seq-ascending) message dicts with at least
        ``message_id``, ``role``, ``content``.
    injections: chronological (id-ascending) injection event dicts with ``id``
        and ``payload.anchor`` (the anchored message_id).

    Returns one WindowOutcome per injection, in input order.
    """
    index_by_message_id = {
        m.get("message_id"): i for i, m in enumerate(messages) if m.get("message_id")
    }
    outcomes: List[WindowOutcome] = []
    for pos, inj in enumerate(injections):
        anchor = (inj.get("payload") or {}).get("anchor")
        start = index_by_message_id.get(anchor)
        if start is None:
            outcomes.append(
                WindowOutcome(
                    injection_event_id=inj["id"],
                    status=STATUS_MISSING_ANCHOR,
                    detail=f"anchor message {anchor!r} not in stream",
                )
            )
            continue
        # Window closes at the next injection whose anchor resolves downstream.
        end = len(messages)
        for later in injections[pos + 1 :]:
            later_start = index_by_message_id.get((later.get("payload") or {}).get("anchor"))
            if later_start is not None and later_start > start:
                end = later_start
                break
        outcomes.append(_judgeable_window(inj["id"], messages[start:end]))
    return outcomes


def _judgeable_window(event_id: int, window: List[Dict[str, Any]]) -> WindowOutcome:
    question_parts = [m["content"] for m in window if m.get("role") == "user"]
    answer_parts = [m["content"] for m in window if m.get("role") == "assistant"]
    if not answer_parts:
        return WindowOutcome(
            injection_event_id=event_id,
            status=STATUS_NO_WINDOW,
            question="\n".join(question_parts),
            detail="no assistant message before window close (interrupted turn?)",
        )
    answer = strip_think_tags("\n".join(answer_parts)).strip()
    if not answer:
        return WindowOutcome(
            injection_event_id=event_id,
            status=STATUS_NO_WINDOW,
            question="\n".join(question_parts),
            detail="assistant answer empty after think-strip (truncated mid-think?)",
        )
    return WindowOutcome(
        injection_event_id=event_id,
        status=STATUS_OK,
        question="\n".join(question_parts),
        answer=answer,
    )
