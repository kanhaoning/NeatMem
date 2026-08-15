"""Batch policy for cursor-driven message scheduling.

Pure mechanics shared by the ``/v1/messages/next-batch/`` endpoint and the
in-process scheduler: given a scope, decide which pending messages form the
next batch. Extraction semantics live elsewhere (``add_memories``); this
module only slices the pending stream.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from neatmem.storage.message.base import AbstractMessageStore

logger = logging.getLogger(__name__)

# Store track consumed by this batch policy. "graph" is reserved for the
# future graph track with its own cursor row.
VECTOR_STORE_TRACK = "vector"


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
