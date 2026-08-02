"""Pure-sqlite3 simulator for mem0's ``messages`` table behavior.

This module does **not** import mem0, so it can be used as a golden reference
when the current environment cannot load ``mem0.memory.storage`` (for example
due to missing shared libraries).  It reproduces the relevant parts of
``SQLiteManager`` as faithfully as possible based on the mem0ai==2.0.0 source.
"""

import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class Mem0MessagesSimulator:
    """In-memory sqlite3 reproduction of mem0's messages table."""

    def __init__(self):
        self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE messages (
                id            TEXT PRIMARY KEY,
                session_scope TEXT,
                role          TEXT,
                content       TEXT,
                name          TEXT,
                created_at    DATETIME
            )
            """
        )

    @staticmethod
    def _build_session_scope(filters: Dict[str, Any]) -> str:
        """Exact copy of mem0's private helper."""
        parts = []
        for key in sorted(["user_id", "agent_id", "run_id"]):
            val = filters.get(key)
            if val:
                parts.append(f"{key}={val}")
        return "&".join(parts)

    def save_messages(
        self,
        messages: List[Dict[str, Any]],
        filters: Dict[str, Any],
        created_at: Optional[str] = None,
    ) -> None:
        """Reproduce mem0's save_messages behavior.

        - scope excludes app_id
        - all messages in one call share the same created_at
        - retention keeps the most recent 10 messages (unstable on ties)

        Args:
            created_at: Optional ISO timestamp override.  When omitted, the
                simulator behaves like mem0 and uses ``datetime.now()``.
        """
        session_scope = self._build_session_scope(filters)
        if not session_scope:
            return
        if not messages:
            return

        if created_at is None:
            created_at = datetime.now(timezone.utc).isoformat()
        for msg in messages:
            self._conn.execute(
                """
                INSERT INTO messages (id, session_scope, role, content, name, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    session_scope,
                    msg.get("role", ""),
                    msg.get("content", ""),
                    msg.get("name"),
                    created_at,
                ),
            )

        # mem0's exact cleanup query
        self._conn.execute(
            """
            DELETE FROM messages
            WHERE session_scope = ?
              AND id NOT IN (
                  SELECT id FROM (
                      SELECT id FROM messages
                      WHERE session_scope = ?
                      ORDER BY created_at DESC
                      LIMIT 10
                  )
              )
            """,
            (session_scope, session_scope),
        )

    def get_last_messages(
        self,
        filters: Dict[str, Any],
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Reproduce mem0's get_last_messages behavior.

        Returns rows ordered by ``created_at DESC`` exactly as mem0 does.  When
        multiple messages share a timestamp, SQLite falls back to rowid order,
        which means the result is effectively insertion order for tied batches.
        """
        session_scope = self._build_session_scope(filters)
        if not session_scope:
            return []

        cur = self._conn.execute(
            """
            SELECT role, content, name, created_at
            FROM messages
            WHERE session_scope = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (session_scope, limit),
        )
        rows = cur.fetchall()
        return [
            {"role": r[0], "content": r[1], "name": r[2], "created_at": r[3]}
            for r in rows
        ]

    def close(self) -> None:
        self._conn.close()
