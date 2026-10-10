"""Feedback batch runner — judge pending injections and update the projection.

Plan: docs/internal-notes/20261005-citation-feedback-memory-importance-plan.md
§5 (offline batch; never on the serving hot path).

One batch pass:
1. Take pending injection events (no judgment event yet).
2. Group by scope, load the scope's chronological messages, slice one window
   per injection (feedback.window).
3. Skip non-counted sources (v1 counts user_prompt only; midtask is
   events-only until fine-grained messages land) and no_window outcomes —
   both get marker judgment events so they leave the pending queue without
   touching the counters (no_window counts neither inject nor used).
4. Judge each ok window (chunked), append one judgment event per memory,
   increment the double counter, then evaluate the eviction gate
   (inject_count >= min_injections AND used_count == 0, gated on
   MEMORY_FEEDBACK_EVICTION_ENABLED) and mark eviction_state='derived'.

Rule 7: a JudgeError on one injection fails that injection loudly (logged,
left pending for the next run) — it is never silently counted.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from neatmem import config
from neatmem.feedback.judge import JudgeError, judge_memories, make_judge_client
from neatmem.feedback.window import STATUS_OK, slice_windows
from neatmem.storage.activity import (
    ActivityStore,
    EVICTION_DERIVED,
    KIND_JUDGMENT,
    VERDICT_USED,
)

logger = logging.getLogger(__name__)

# v1 判定范围（plan §4.1）：仅 user_prompt 计数；其余 source 只记事件。
COUNTED_SOURCES = frozenset({"user_prompt"})

# Judge one injection in chunks of this size (LOCOMO 锚点 top_k=20 留余量).
JUDGE_CHUNK_SIZE = 25

Scope = Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]

_SCOPE_KEYS = ("app_id", "user_id", "agent_id", "run_id")


def _scope_of(event: Dict[str, Any]) -> Scope:
    return (event.get("app_id"), event.get("user_id"),
            event.get("agent_id"), event.get("run_id"))


def run_judge_batch(
    store: ActivityStore,
    load_messages: Callable[[Dict[str, Any]], List[Dict[str, Any]]],
    *,
    client=None,
    judge_fn: Callable[..., Dict[str, Dict[str, str]]] = judge_memories,
    min_injections: Optional[int] = None,
    eviction_enabled: Optional[bool] = None,
    limit: Optional[int] = None,
) -> Dict[str, int]:
    """Run one judge batch over all pending injections.

    load_messages: scope filters -> chronological message dicts
        (message_id/role/content). Dependency injection keeps the runner
        storage-agnostic and testable.
    client/judge_fn: DI for tests; production resolves via make_judge_client.
    Returns a run summary (counts per outcome).
    """
    if min_injections is None:
        min_injections = config.MEMORY_FEEDBACK_EVICTION_MIN_INJECTIONS
    if eviction_enabled is None:
        eviction_enabled = config.MEMORY_FEEDBACK_EVICTION_ENABLED

    pending = store.pending_injections()
    if limit is not None:
        pending = pending[:limit]
    summary = {"pending": len(pending), "judged": 0, "no_window": 0,
               "skipped_source": 0, "skipped_claimed": 0, "failed": 0,
               "evicted": 0}
    if not pending:
        return summary

    judge_client, judge_model = make_judge_client(client)

    by_scope: Dict[Scope, List[Dict[str, Any]]] = {}
    for event in pending:
        by_scope.setdefault(_scope_of(event), []).append(event)

    for scope, events in by_scope.items():
        filters = {k: v for k, v in zip(_SCOPE_KEYS, scope) if v}
        messages = load_messages(filters)
        outcomes = {o.injection_event_id: o for o in slice_windows(messages, events)}
        for event in events:
            # Claim before judging (plan 20261010 §3.2): the serve auto-judge
            # thread and manual CLI runs must never double-judge — that would
            # double-count inject/used. Unclaimed = another runner owns it.
            if not store.claim_injection(event["id"]):
                summary["skipped_claimed"] += 1
                continue
            outcome = outcomes[event["id"]]
            payload = event["payload"]
            source = payload.get("source", "")
            try:
                result = _process_injection(
                    store, event, outcome, source,
                    judge_client, judge_model, judge_fn,
                    min_injections, eviction_enabled,
                )
            except JudgeError as e:
                summary["failed"] += 1
                logger.error(
                    "[feedback] injection event %s failed, left pending: %s",
                    event["id"], e,
                )
                continue
            summary[result] += 1
    return summary


def _process_injection(
    store: ActivityStore,
    event: Dict[str, Any],
    outcome,
    source: str,
    judge_client,
    judge_model: str,
    judge_fn,
    min_injections: int,
    eviction_enabled: bool,
) -> str:
    """Process one injection; returns the summary key to increment."""
    if source not in COUNTED_SOURCES:
        _mark_done(store, event, reason=f"source {source!r} not counted in v1")
        return "skipped_source"
    if outcome.status != STATUS_OK:
        _mark_done(store, event, reason=f"{outcome.status}: {outcome.detail}")
        return "no_window"

    memory_ids = [str(mid) for mid in event["payload"].get("memory_ids", [])]
    if not memory_ids:
        _mark_done(store, event, reason="empty memory_ids")
        return "no_window"

    texts = _memory_texts(event)
    missing_text = [mid for mid in memory_ids if mid not in texts]
    if missing_text:
        # Snapshot is pinned at injection time (plan §4.1); judging against
        # empty text would silently degrade to all-unused — fail loud instead.
        raise JudgeError(
            f"injection event {event['id']} missing text snapshot for {missing_text}"
        )
    verdicts: Dict[str, str] = {}
    evidence: Dict[str, str] = {}
    for start in range(0, len(memory_ids), JUDGE_CHUNK_SIZE):
        chunk_ids = memory_ids[start : start + JUDGE_CHUNK_SIZE]
        chunk = [{"id": mid, "text": texts.get(mid, "")} for mid in chunk_ids]
        judged = judge_fn(judge_client, judge_model, outcome.question, chunk, outcome.answer)
        verdicts.update(judged["verdicts"])
        evidence.update(judged["evidence"])

    for mid in memory_ids:
        verdict = verdicts[mid]  # judge_memories guarantees full coverage
        store.record_event(
            KIND_JUDGMENT,
            app_id=event.get("app_id"),
            user_id=event.get("user_id"),
            agent_id=event.get("agent_id"),
            run_id=event.get("run_id"),
            subject_id=mid,
            payload={
                "injection_event_id": event["id"],
                "verdict": verdict,
                "evidence": evidence.get(mid, ""),
                "judge_model": judge_model,
                "source": source,
            },
        )
        store.upsert_feedback_counts(
            mid, inject_delta=1, used_delta=1 if verdict == VERDICT_USED else 0
        )
        if eviction_enabled:
            _evaluate_gate(store, mid, min_injections)
    return "judged"


def _evaluate_gate(store: ActivityStore, memory_id: str, min_injections: int) -> None:
    """Mark eviction_state='derived' when the gate fires.

    Gate: inject_count >= min_injections AND used_count == 0. Only 'none' rows
    are transitioned — 'manual' is a human decision the rule never touches.
    """
    row = store.get_feedback(memory_id)
    if not row or row["eviction_state"] != "none":
        return
    if row["inject_count"] >= min_injections and row["used_count"] == 0:
        store.set_eviction(memory_id, EVICTION_DERIVED)
        logger.info("[feedback] evicted (derived): %s", memory_id)


def _mark_done(store: ActivityStore, event: Dict[str, Any], *, reason: str) -> None:
    """Marker judgment event: leaves the pending queue, touches no counters."""
    store.record_event(
        KIND_JUDGMENT,
        app_id=event.get("app_id"),
        user_id=event.get("user_id"),
        agent_id=event.get("agent_id"),
        run_id=event.get("run_id"),
        payload={"injection_event_id": event["id"], "skipped": reason},
    )


def _memory_texts(event: Dict[str, Any]) -> Dict[str, str]:
    """Text snapshot pinned at injection time (payload ``memories``).

    The snapshot is written by the ingestion-side reporter because memory text
    mutates via dedup rewrite — judging against today's text would misjudge
    what was actually injected (plan §4.1 training-set convention).
    """
    texts: Dict[str, str] = {}
    for item in event["payload"].get("memories", []):
        texts[str(item.get("id"))] = item.get("text", "")
    return texts
