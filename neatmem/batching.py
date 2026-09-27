"""Batch policy for cursor-driven message scheduling.

Pure mechanics shared by the ``/v1/messages/next-batch/`` endpoint and the
in-process scheduler: given a scope, decide which pending messages form the
next batch. Extraction semantics live elsewhere (``add_memories``); this
module only slices the pending stream.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional

from neatmem.storage.message.base import AbstractMessageStore

logger = logging.getLogger(__name__)

# Store track consumed by this batch policy. "graph" is reserved for the
# future graph track with its own cursor row.
VECTOR_STORE_TRACK = "vector"


class FlushConflictError(Exception):
    """Cursor moved concurrently while a flush was in progress."""


def batch_event_at(rows: List[Dict[str, Any]]) -> Optional[str]:
    """Event time of a batch: the minimum ``event_at`` across its messages.

    ``event_at`` is the client-supplied event time; rows without it (older
    clients, pre-migration DBs) fall back to the row's ``created_at`` (server
    receipt time). Returns None for an empty batch.

    All producers write UTC ISO-8601 strings, so lexicographic min matches
    chronological min.
    """
    stamps = [(r.get("event_at") or r.get("created_at")) for r in rows]
    stamps = [s for s in stamps if s]
    return min(stamps) if stamps else None


def compute_next_batch(
    message_store: AbstractMessageStore,
    scope: Dict[str, str],
    *,
    store_track: str = VECTOR_STORE_TRACK,
    batch_size: int,
    deadline_secs: int,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Decide the next batch for a scope.

    Policy:
        - pending >= batch_size  -> the oldest ``batch_size`` messages
        - 0 < pending < batch_size and the oldest pending message is older
          than ``deadline_secs`` -> all pending messages (deadline flush)
        - otherwise -> empty batch

    Args:
        message_store: Store to read cursor and pending messages from.
        scope: Dict with user_id/agent_id/run_id ("" for unset).
        store_track: Cursor track ("vector" / "graph").
        batch_size: Full batch size.
        deadline_secs: Max age of the oldest pending message before a
            partial batch is flushed.
        now: Override for the current time (tests).

    Returns:
        {"message_ids": [...], "seqs": [...], "pending_count": int}
        with empty lists when no batch is due.
    """
    cursor = message_store.get_cursor(
        scope.get("user_id", ""), scope.get("agent_id", ""), scope.get("run_id", ""),
        store_track,
    )
    pending_count = message_store.count_pending_messages(scope, cursor)

    batch: List[Dict[str, Any]] = []
    if pending_count >= batch_size:
        batch = message_store.get_pending_messages(scope, cursor, batch_size)
    elif pending_count > 0:
        pending = message_store.get_pending_messages(scope, cursor, pending_count)
        oldest = datetime.fromisoformat(pending[0]["created_at"])
        age = ((now or datetime.now(timezone.utc)) - oldest).total_seconds()
        if age > deadline_secs:
            batch = pending

    return {
        "message_ids": [m["message_id"] for m in batch],
        "seqs": [m["seq"] for m in batch],
        "pending_count": pending_count,
    }


async def process_scope_batch(
    message_store: AbstractMessageStore,
    scope: Dict[str, str],
    *,
    extract_batch: Callable[[Dict[str, str], List[str], str], Awaitable[None]],
    consecutive_failures: Dict[Any, int],
    store_track: str = VECTOR_STORE_TRACK,
    batch_size: int,
    deadline_secs: int,
    max_consecutive_failures: int,
) -> None:
    """Extract the next pending batch for one scope, with poison-batch breaker.

    A batch that fails deterministically (e.g. an embedding query over the
    model's token limit) would otherwise be retried forever, silently
    blocking the scope's cursor (2026-09-22 incident: one scope stuck 26h).
    After ``max_consecutive_failures`` consecutive failures the batch is
    skipped — the cursor advances past it with an error log. Skipped messages
    stay in the messages table and can be replayed by resetting the cursor.

    Args:
        extract_batch: Async callable (scope, message_ids, req_id) performing
            the actual extraction; raises on failure.
        consecutive_failures: Mutable scope_key -> failure-count map shared
            across scheduler iterations (reset on success and after a skip).
    """
    batch = await asyncio.to_thread(
        compute_next_batch,
        message_store,
        scope,
        store_track=store_track,
        batch_size=batch_size,
        deadline_secs=deadline_secs,
    )
    if not batch["message_ids"]:
        return
    scope_key = (scope.get("user_id", ""), scope.get("agent_id", ""), scope.get("run_id", ""))
    req_id = uuid.uuid4().hex[:8]
    try:
        await extract_batch(scope, batch["message_ids"], req_id)
    except Exception:
        failures = consecutive_failures.get(scope_key, 0) + 1
        consecutive_failures[scope_key] = failures
        if failures >= max_consecutive_failures:
            logger.error(
                "[%s] poison batch skipped scope=%s seqs=%s..%s after %s consecutive "
                "failures, advancing cursor; messages retained in store for replay",
                req_id, scope, batch["seqs"][0], batch["seqs"][-1], failures,
            )
            consecutive_failures[scope_key] = 0
            await asyncio.to_thread(
                message_store.advance_cursor,
                scope.get("user_id", ""), scope.get("agent_id", ""), scope.get("run_id", ""),
                store_track, batch["seqs"][-1],
            )
        else:
            logger.exception(
                "[%s] batch extraction failed scope=%s seqs=%s..%s (consecutive "
                "failures=%s), cursor not advanced, retry next round",
                req_id, scope, batch["seqs"][0], batch["seqs"][-1], failures,
            )
        return
    consecutive_failures.pop(scope_key, None)
    await asyncio.to_thread(
        message_store.advance_cursor,
        scope.get("user_id", ""), scope.get("agent_id", ""), scope.get("run_id", ""),
        store_track, batch["seqs"][-1],
    )
    logger.info(
        "[%s] batch extraction done scope=%s batch=%s msgs seqs=%s..%s pending=%s",
        req_id, scope, len(batch["seqs"]), batch["seqs"][0],
        batch["seqs"][-1], batch["pending_count"],
    )


async def flush_scope(
    message_store: AbstractMessageStore,
    scope: Dict[str, str],
    *,
    store_track: str = VECTOR_STORE_TRACK,
    batch_size: int,
    extract_batch: Callable[[Dict[str, str], List[str]], Awaitable[None]],
) -> Dict[str, Any]:
    """Synchronously extract ALL pending messages for a scope.

    Loops "take next pending chunk (<= batch_size) -> extract -> advance
    cursor" until nothing is pending, ignoring the full-batch/deadline
    policy of ``compute_next_batch``. The final chunk may be partial.

    Args:
        extract_batch: async callable (scope, message_ids) that runs the
            extraction; must raise on failure (cursor then stays put and the
            error propagates — earlier batches stay committed).

    Returns:
        {"batches": int, "extracted_count": int, "last_processed_seq": int}
        (last_processed_seq is the current cursor when nothing was pending).

    Raises:
        FlushConflictError: the cursor was advanced concurrently mid-flush.
    """
    batches = 0
    extracted = 0
    last = message_store.get_cursor(
        scope.get("user_id", ""), scope.get("agent_id", ""), scope.get("run_id", ""),
        store_track,
    )
    while True:
        cursor = message_store.get_cursor(
            scope.get("user_id", ""), scope.get("agent_id", ""), scope.get("run_id", ""),
            store_track,
        )
        pending = message_store.get_pending_messages(scope, cursor, batch_size)
        if not pending:
            break
        await extract_batch(scope, [m["message_id"] for m in pending])
        last = pending[-1]["seq"]
        if not message_store.advance_cursor(
            scope.get("user_id", ""), scope.get("agent_id", ""), scope.get("run_id", ""),
            store_track, last,
        ):
            raise FlushConflictError(
                f"cursor moved concurrently during flush (scope={scope}, seq={last})"
            )
        batches += 1
        extracted += len(pending)
    return {"batches": batches, "extracted_count": extracted, "last_processed_seq": last}
