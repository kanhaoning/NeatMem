"""Message store factory for NeatMem.

This module is the public entry point for acquiring a message-history backend.
It keeps the rest of the codebase decoupled from the concrete implementation
and allows downstream applications to choose between the self-managed SQLite
store and a no-op store (external-injection mode).
"""

import os
from typing import Optional

from neatmem.storage.message.base import AbstractMessageStore
from neatmem.storage.message.noop import NoOpMessageStore
from neatmem.storage.message.sqlite import SQLiteMessageStore


def create_message_store(
    db_path: Optional[str] = None,
    *,
    extract_last_k: int = 10,
    backend: str = "sqlite",
) -> AbstractMessageStore:
    """Return a configured message store instance.

    Args:
        db_path: Path to the SQLite database.  Defaults to ``HISTORY_DB_PATH``
            from ``neatmem.config`` when available, otherwise ``history.db`` in
            the current working directory.  Ignored when ``backend="none"``.
        extract_last_k: Number of recent messages to retrieve for extraction
            context.
        backend: ``"sqlite"`` (default) for the self-managed SQLite store, or
            ``"none"`` for a no-op store (external-injection mode where the host
            application provides last_k_messages via the request body).

    Raises:
        ValueError: If ``backend`` is not supported.
    """
    if backend == "none":
        return NoOpMessageStore(extract_last_k=extract_last_k)

    if backend != "sqlite":
        raise ValueError(f"Unsupported message store backend: {backend}")

    if db_path is None:
        try:
            from neatmem.config import HISTORY_DB_PATH

            db_path = HISTORY_DB_PATH
        except Exception:
            db_path = os.path.join(os.getcwd(), "history.db")

    return SQLiteMessageStore(
        db_path,
        extract_last_k=extract_last_k,
    )
