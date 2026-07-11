"""
Self-managed SQLite message store for NeatMem.

This implementation does not depend on mem0 internal APIs.  It owns its own
schema, connection and retention policy.
"""

import logging
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from neatmem.storage.message.base import AbstractMessageStore

logger = logging.getLogger(__name__)

# Scope dimensions used by NeatMem, in fixed order for index consistency.
_SCOPE_KEYS = ("app_id", "user_id", "agent_id", "run_id")

# Hardcoded retention limit per scope.  Keeps the most recent messages and
# deletes older ones, fixing mem0 issues #5661/#5632 where oldest messages
# were kept instead of newest.
_MAX_MESSAGES_PER_SCOPE = 1000

_MESSAGES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS messages (
    seq         INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id  TEXT UNIQUE NOT NULL,
    app_id      TEXT,
    user_id     TEXT,
    agent_id    TEXT,
    run_id      TEXT,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    name        TEXT,
    created_at  DATETIME NOT NULL
)
"""

_MESSAGES_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_messages_scope_time
ON messages(app_id, user_id, agent_id, run_id, created_at)
"""


def _build_scope_where(
    filters: Dict[str, Any],
    *,
    exclude: Optional[set] = None,
) -> Tuple[List[str], List[Any]]:
    """Build WHERE conditions for scope fields present in ``filters``.

    Only processes keys that are present and non-empty in ``filters``.  This
    avoids ``run_id = ? AND run_id IS NOT NULL`` conflicts in list_sessions.

    Args:
        filters: Dict possibly containing app_id/user_id/agent_id/run_id.
        exclude: Optional set of keys to skip (e.g. ``{"run_id"}`` for
            list_sessions).

    Returns:
        (conditions, params) — conditions is a list of ``"key = ?"`` strings,
        params is the corresponding list of values.
    """
    exclude = exclude or set()
    conditions: List[str] = []
    params: List[Any] = []
    for key in _SCOPE_KEYS:
        if key in exclude:
            continue
        val = filters.get(key)
        if val:
            conditions.append(f"{key} = ?")
            params.append(val)
    return conditions, params


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SQLiteMessageStore(AbstractMessageStore):
    """SQLite-backed message store with per-scope retention.

    Features:
        - independent schema (4 scope fields, no session_scope string)
        - message_id (not id) for cross-table clarity
        - hardcoded retention of 1000 messages per scope (newest kept)
        - stable cleanup via ``ORDER BY created_at DESC, seq DESC``
        - composite index on ``(app_id, user_id, agent_id, run_id, created_at)``
    """

    def __init__(
        self,
        db_path: str,
        *,
        extract_last_k: int = 10,
    ):
        self.db_path = db_path
        self.extract_last_k = extract_last_k
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(
            db_path,
            check_same_thread=False,
            isolation_level=None,  # autocommit; transactions managed explicitly
        )
        self._connection.row_factory = sqlite3.Row
        self._ensure_schema()
        logger.info(
            "SQLiteMessageStore initialized: %s (max_per_scope=%s, extract_last_k=%s)",
            db_path,
            _MAX_MESSAGES_PER_SCOPE,
            extract_last_k,
        )

    def _ensure_schema(self) -> None:
        """Create tables and indexes if they do not exist."""
        with self._lock:
            self._connection.execute(_MESSAGES_TABLE_SQL)
            self._connection.execute(_MESSAGES_INDEX_SQL)

    # ------------------------------------------------------------------ #
    # write path
    # ------------------------------------------------------------------ #

    def save_messages(
        self,
        messages: List[Dict[str, Any]],
        filters: Dict[str, Any],
    ) -> None:
        """Save raw messages with the scope fields from ``filters``."""
        conditions, _ = _build_scope_where(filters)
        if not conditions:
            logger.warning(
                "SQLiteMessageStore.save_messages called with empty scope; skipping"
            )
            return
        if not messages:
            return

        app_id = filters.get("app_id")
        user_id = filters.get("user_id")
        agent_id = filters.get("agent_id")
        run_id = filters.get("run_id")
        created_at = _utc_now()

        with self._lock:
            try:
                self._connection.execute("BEGIN")
                for msg in messages:
                    self._connection.execute(
                        """
                        INSERT INTO messages
                            (message_id, app_id, user_id, agent_id, run_id,
                             role, content, name, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            str(uuid.uuid4()),
                            app_id,
                            user_id,
                            agent_id,
                            run_id,
                            msg.get("role", ""),
                            msg.get("content", ""),
                            msg.get("name"),
                            created_at,
                        ),
                    )
                self._enforce_retention(filters)
                self._connection.execute("COMMIT")
            except Exception:
                self._connection.execute("ROLLBACK")
                raise

    def _enforce_retention(self, filters: Dict[str, Any]) -> None:
        """Keep only the most recent 1000 messages for the scope.

        Secondary ordering by ``seq DESC`` makes cleanup stable even when many
        messages share the same ``created_at`` timestamp.  Caller must hold
        ``self._lock``.
        """
        conditions, params = _build_scope_where(filters)
        if not conditions:
            return
        where_clause = " AND ".join(conditions)
        self._connection.execute(
            f"""
            DELETE FROM messages
            WHERE {where_clause}
              AND seq NOT IN (
                  SELECT seq FROM (
                      SELECT seq FROM messages
                      WHERE {where_clause}
                      ORDER BY created_at DESC, seq DESC
                      LIMIT ?
                  )
              )
            """,
            (*params, *params, _MAX_MESSAGES_PER_SCOPE),
        )

    # ------------------------------------------------------------------ #
    # read path
    # ------------------------------------------------------------------ #

    def get_last_messages(
        self,
        filters: Dict[str, Any],
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Return the most recent ``limit`` messages for the scope, oldest first.

        If ``limit`` is not provided, falls back to ``self.extract_last_k``.
        """
        conditions, params = _build_scope_where(filters)
        if not conditions:
            return []

        effective_limit = limit if limit is not None else self.extract_last_k
        if effective_limit <= 0:
            return []

        where_clause = " AND ".join(conditions)
        with self._lock:
            cur = self._connection.execute(
                f"""
                SELECT message_id, app_id, user_id, agent_id, run_id,
                       role, content, name, created_at, seq
                FROM (
                    SELECT message_id, app_id, user_id, agent_id, run_id,
                           role, content, name, created_at, seq
                    FROM messages
                    WHERE {where_clause}
                    ORDER BY created_at DESC, seq DESC
                    LIMIT ?
                )
                ORDER BY created_at ASC, seq ASC
                """,
                (*params, effective_limit),
            )
            rows = cur.fetchall()

        return [_row_to_message(r) for r in rows]

    def query_messages(
        self,
        filters: Dict[str, Any],
        *,
        content_like: Optional[str] = None,
        after: Optional[str] = None,
        before: Optional[str] = None,
        roles: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0,
        order: str = "desc",
    ) -> List[Dict[str, Any]]:
        """Query messages with filtering and pagination."""
        conditions, params = _build_scope_where(filters)
        if not conditions:
            return []

        if content_like:
            conditions.append("content LIKE ?")
            params.append(f"%{content_like}%")
        if after:
            conditions.append("created_at >= ?")
            params.append(after)
        if before:
            conditions.append("created_at <= ?")
            params.append(before)
        if roles:
            placeholders = ",".join("?" * len(roles))
            conditions.append(f"role IN ({placeholders})")
            params.extend(roles)

        order_by = (
            "created_at ASC, seq ASC" if order == "asc" else "created_at DESC, seq DESC"
        )
        where_clause = " AND ".join(conditions)

        query = f"""
            SELECT message_id, app_id, user_id, agent_id, run_id,
                   role, content, name, created_at
            FROM messages
            WHERE {where_clause}
            ORDER BY {order_by}
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])

        with self._lock:
            cur = self._connection.execute(query, params)
            rows = cur.fetchall()

        return [_row_to_message(r) for r in rows]

    def count_messages(
        self,
        filters: Dict[str, Any],
        *,
        content_like: Optional[str] = None,
        after: Optional[str] = None,
        before: Optional[str] = None,
        roles: Optional[List[str]] = None,
    ) -> int:
        """Count messages matching the filters."""
        conditions, params = _build_scope_where(filters)
        if not conditions:
            return 0

        if content_like:
            conditions.append("content LIKE ?")
            params.append(f"%{content_like}%")
        if after:
            conditions.append("created_at >= ?")
            params.append(after)
        if before:
            conditions.append("created_at <= ?")
            params.append(before)
        if roles:
            placeholders = ",".join("?" * len(roles))
            conditions.append(f"role IN ({placeholders})")
            params.extend(roles)

        where_clause = " AND ".join(conditions)
        query = f"SELECT COUNT(*) FROM messages WHERE {where_clause}"

        with self._lock:
            cur = self._connection.execute(query, params)
            return cur.fetchone()[0]

    def list_sessions(
        self,
        filters: Dict[str, Any],
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List distinct ``run_id`` values for the scope, with last-active time."""
        conditions, params = _build_scope_where(filters, exclude={"run_id"})
        if not conditions:
            return []

        conditions.append("run_id IS NOT NULL")
        where_clause = " AND ".join(conditions)

        query = f"""
            SELECT run_id, MAX(created_at) AS last_active_at
            FROM messages
            WHERE {where_clause}
            GROUP BY run_id
            ORDER BY last_active_at DESC
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])

        with self._lock:
            cur = self._connection.execute(query, params)
            rows = cur.fetchall()

        return [
            {"run_id": r["run_id"], "last_active_at": r["last_active_at"]}
            for r in rows
        ]

    # ------------------------------------------------------------------ #
    # delete / reset
    # ------------------------------------------------------------------ #

    def delete_messages(self, filters: Dict[str, Any]) -> int:
        """Delete messages matching the scope filters.

        Returns the number of deleted rows.  Refuses to delete when no scope
        field is provided; use ``reset()`` for full-table clear.
        """
        conditions, params = _build_scope_where(filters)
        if not conditions:
            logger.warning(
                "delete_messages called with empty scope; refusing to delete all. "
                "Use reset() for full-table clear."
            )
            return 0

        where_clause = " AND ".join(conditions)
        with self._lock:
            cur = self._connection.execute(
                f"DELETE FROM messages WHERE {where_clause}",
                params,
            )
            return cur.rowcount

    def reset(self) -> None:
        """Drop and recreate the messages table (full reset)."""
        with self._lock:
            self._connection.execute("DROP TABLE IF EXISTS messages")
            self._connection.execute(_MESSAGES_TABLE_SQL)
            self._connection.execute(_MESSAGES_INDEX_SQL)

    def close(self) -> None:
        with self._lock:
            self._connection.close()


def _row_to_message(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "message_id": row["message_id"],
        "app_id": row["app_id"],
        "user_id": row["user_id"],
        "agent_id": row["agent_id"],
        "run_id": row["run_id"],
        "role": row["role"],
        "content": row["content"],
        "name": row["name"],
        "created_at": row["created_at"],
    }
