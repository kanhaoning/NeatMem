"""Event recording helpers for the serving endpoints (plan §5.2).

Keeps main.py thin: the endpoints call these two functions, flag-gated on
MEMORY_FEEDBACK_ENABLED. Both are synchronous (SQLite writes are
microseconds); endpoints invoke them via asyncio.to_thread.

Conventions fixed by the plan:
- kind='search': payload.hits = the returned candidate list (rank/score),
  including hits the client ends up NOT injecting — the training-set control
  group (§4.1). Zero-hit searches are recorded too (漏召信号).
- kind='injection': subject_id = the anchor message_id (the message that
  carried ``preceded_by_injection``), so idx_events_subject answers both
  "judgments of memory X" and "injection at message Y" with indexed lookups.
  payload.memories pins the text snapshot at injection time (memory text
  mutates via dedup rewrite — judging/training need what was injected).
"""

import logging
from typing import Any, Callable, Dict, List, Optional

from neatmem.storage.activity import ActivityStore, KIND_INJECTION, KIND_SEARCH

logger = logging.getLogger(__name__)


def record_search_event(
    store: ActivityStore,
    *,
    query: str,
    filters: Dict[str, Any],
    run_id: Optional[str],
    memories: List[Dict[str, Any]],
) -> int:
    """Record one search event; returns the new event id."""
    hits = [
        {
            "memory_id": m.get("id"),
            "rank": i + 1,
            "score": m.get("score"),
        }
        for i, m in enumerate(memories)
    ]
    return store.record_event(
        KIND_SEARCH,
        app_id=filters.get("app_id"),
        user_id=filters.get("user_id"),
        agent_id=filters.get("agent_id"),
        run_id=run_id,
        payload={"query": query, "hits": hits},
    )


def record_injection_events(
    store: ActivityStore,
    *,
    messages: List[Dict[str, Any]],
    saved: List[Dict[str, Any]],
    filters: Dict[str, Any],
    memory_text: Optional[Callable[[str], Optional[str]]] = None,
) -> int:
    """Record injection events for messages carrying ``preceded_by_injection``.

    messages/saved are positionally aligned (save_messages returns one entry
    per input message, in order). ``memory_text`` resolves a memory id to its
    current text for the payload snapshot; None skips the snapshot.

    Re-ingest safety: an anchor that already has an injection event is not
    re-recorded (deduped message replay must not double-count injections).
    Returns the number of events recorded.
    """
    recorded = 0
    for msg, saved_item in zip(messages, saved):
        marker = msg.get("preceded_by_injection")
        if not marker:
            continue
        anchor = saved_item.get("message_id")
        memory_ids = [str(mid) for mid in marker.get("memory_ids", [])]
        source = str(marker.get("source", ""))
        if not memory_ids or not anchor:
            logger.warning(
                "[feedback] malformed preceded_by_injection skipped: %r", marker
            )
            continue
        if any(True for _ in store.iter_events(KIND_INJECTION, subject_id=anchor)):
            logger.info("[feedback] injection at %s already recorded, skip", anchor)
            continue
        payload: Dict[str, Any] = {
            "memory_ids": memory_ids,
            "source": source,
            "anchor": anchor,
        }
        if memory_text is not None:
            payload["memories"] = [
                {"id": mid, "text": memory_text(mid) or ""} for mid in memory_ids
            ]
        store.record_event(
            KIND_INJECTION,
            app_id=filters.get("app_id"),
            user_id=filters.get("user_id"),
            agent_id=filters.get("agent_id"),
            run_id=filters.get("run_id"),
            subject_id=anchor,
            payload=payload,
        )
        recorded += 1
    return recorded
