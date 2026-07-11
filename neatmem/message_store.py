"""Backward-compatible factory alias.

This module exists so that scripts written against the 7-03 experiment copy's
import path (``from neatmem.message_store import MessageStore``) continue to
work with the main package's ``neatmem.storage.message.factory`` module.

The canonical entry point is ``neatmem.storage.message.factory.create_message_store``.
"""

from neatmem.storage.message.factory import create_message_store


def MessageStore(
    db_path=None,
    *,
    extract_last_k: int = 10,
    max_messages_per_scope: int = 1000,  # accepted for backward compat, ignored
    backend: str = "sqlite",
):
    """Backward-compatible alias for ``create_message_store``.

    Args:
        db_path: Path to the SQLite database.
        extract_last_k: Number of recent messages for extraction context.
        max_messages_per_scope: Ignored (retention is hardcoded to 1000).
        backend: ``"sqlite"`` or ``"none"``.
    """
    return create_message_store(
        db_path,
        extract_last_k=extract_last_k,
        backend=backend,
    )
