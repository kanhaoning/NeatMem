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

# Extraction cursor: one row per (user_id, agent_id, run_id, store) tracking
# how far the message stream has been processed. ``store`` is the mem0 config
# vocabulary: "vector" (L1 fact track), "graph" (reserved).
_CURSOR_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS message_cursor (
    user_id             TEXT NOT NULL,
    agent_id            TEXT NOT NULL DEFAULT '',
    run_id              TEXT NOT NULL DEFAULT '',
    store               TEXT NOT NULL,
    last_processed_seq  INTEGER NOT NULL DEFAULT 0,
    updated_at          TEXT NOT NULL,
    PRIMARY KEY (user_id, agent_id, run_id, store)
)
"""


def _norm_scope_value(value: Any) -> str:
    """Normalize a cursor scope field: None -> "" (SQLite treats NULLs as
    distinct in unique constraints, so cursor rows store empty strings)."""
    return value if value else ""


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
            self._connection.execute(_CURSOR_TABLE_SQL)

    # ------------------------------------------------------------------ #
    # write path
    # ------------------------------------------------------------------ #

    def save_messages(
        self,
        messages: List[Dict[str, Any]],
        filters: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Save raw messages with the scope fields from ``filters``.

        Returns ``[{"message_id", "seq"}]`` for the saved rows, in insertion
        order.
        """
        conditions, _ = _build_scope_where(filters)
        if not conditions:
            logger.warning(
                "SQLiteMessageStore.save_messages called with empty scope; skipping"
            )
            return []
        if not messages:
            return []

        app_id = filters.get("app_id")
        user_id = filters.get("user_id")
        agent_id = filters.get("agent_id")
        run_id = filters.get("run_id")
        created_at = _utc_now()

        saved: List[Dict[str, Any]] = []
        with self._lock:
            try:
                self._connection.execute("BEGIN")
                for msg in messages:
                    message_id = str(uuid.uuid4())
                    cur = self._connection.execute(
                        """
                        INSERT INTO messages
                            (message_id, app_id, user_id, agent_id, run_id,
                             role, content, name, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            message_id,
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
                    saved.append({"message_id": message_id, "seq": cur.lastrowid})
                self._enforce_retention(filters)
                self._connection.execute("COMMIT")
            except Exception:
                self._connection.execute("ROLLBACK")
                raise
        return saved

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
        # Warn when retention is about to trim messages the extraction cursor
        # has not reached yet (they would never be extracted). Retention is not
        # cursor-aware by design; the warning makes the loss visible.
        cursor_seq = self._get_cursor_locked(
            _norm_scope_value(filters.get("user_id")),
            _norm_scope_value(filters.get("agent_id")),
            _norm_scope_value(filters.get("run_id")),
            "vector",
        )
        unprocessed = self._connection.execute(
            f"""
            SELECT COUNT(*) FROM messages
            WHERE {where_clause}
              AND seq > ?
              AND seq NOT IN (
                  SELECT seq FROM (
                      SELECT seq FROM messages
                      WHERE {where_clause}
                      ORDER BY created_at DESC, seq DESC
                      LIMIT ?
                  )
              )
            """,
            (*params, cursor_seq, *params, _MAX_MESSAGES_PER_SCOPE),
        ).fetchone()[0]
        if unprocessed:
            logger.warning(
                "retention trimming %d unprocessed messages (seq > cursor %d) "
                "for scope user=%s agent=%s run=%s; they will never be extracted",
                unprocessed,
                cursor_seq,
                filters.get("user_id"),
                filters.get("agent_id"),
                filters.get("run_id"),
            )
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
        before_seq: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Return the most recent ``limit`` messages for the scope, oldest first.

        If ``limit`` is not provided, falls back to ``self.extract_last_k``.
        If ``before_seq`` is provided, only messages with ``seq < before_seq``
        are considered.
        """
        conditions, params = _build_scope_where(filters)
        if not conditions:
            return []

        effective_limit = limit if limit is not None else self.extract_last_k
        if effective_limit <= 0:
            return []

        if before_seq is not None:
            conditions.append("seq < ?")
            params.append(before_seq)

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

    def get_messages_by_ids(
        self,
        message_ids: List[str],
    ) -> List[Dict[str, Any]]:
        """Return messages for the given IDs, ordered by ``seq`` ascending."""
        if not message_ids:
            return []
        placeholders = ",".join("?" * len(message_ids))
        with self._lock:
            cur = self._connection.execute(
                f"""
                SELECT message_id, app_id, user_id, agent_id, run_id,
                       role, content, name, created_at, seq
                FROM messages
                WHERE message_id IN ({placeholders})
                ORDER BY seq ASC
                """,
                list(message_ids),
            )
            rows = cur.fetchall()
        result = []
        for r in rows:
            msg = _row_to_message(r)
            msg["seq"] = r["seq"]
            result.append(msg)
        return result

    # ------------------------------------------------------------------ #
    # batch scheduling (cursor-driven)
    # ------------------------------------------------------------------ #

    def get_pending_messages(
        self,
        filters: Dict[str, Any],
        after_seq: int,
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Return up to ``limit`` pending messages (``seq > after_seq``)."""
        where_clause, params = self._cursor_scope_where(filters)
        with self._lock:
            cur = self._connection.execute(
                f"""
                SELECT message_id, seq, created_at
                FROM messages
                WHERE {where_clause} AND seq > ?
                ORDER BY seq ASC
                LIMIT ?
                """,
                (*params, after_seq, limit),
            )
            rows = cur.fetchall()
        return [
            {"message_id": r["message_id"], "seq": r["seq"], "created_at": r["created_at"]}
            for r in rows
        ]

    def count_pending_messages(
        self,
        filters: Dict[str, Any],
        after_seq: int,
    ) -> int:
        """Count pending messages (``seq > after_seq``) for the scope."""
        where_clause, params = self._cursor_scope_where(filters)
        with self._lock:
            cur = self._connection.execute(
                f"SELECT COUNT(*) FROM messages WHERE {where_clause} AND seq > ?",
                (*params, after_seq),
            )
            return cur.fetchone()[0]

    def list_message_scopes(self) -> List[Dict[str, str]]:
        """List distinct (user_id, agent_id, run_id) scopes, NULLs as ""."""
        with self._lock:
            cur = self._connection.execute(
                """
                SELECT DISTINCT COALESCE(user_id, '') AS user_id,
                       COALESCE(agent_id, '') AS agent_id,
                       COALESCE(run_id, '') AS run_id
                FROM messages
                """
            )
            rows = cur.fetchall()
        return [
            {"user_id": r["user_id"], "agent_id": r["agent_id"], "run_id": r["run_id"]}
            for r in rows
        ]

    @staticmethod
    def _cursor_scope_where(filters: Dict[str, Any]) -> Tuple[str, List[Any]]:
        """WHERE clause matching user/agent/run NULL-insensitively.

        Cursor scopes treat NULL and "" as the same value, unlike
        ``_build_scope_where`` which only filters on present fields.
        """
        conditions = []
        params: List[Any] = []
        for key in ("user_id", "agent_id", "run_id"):
            conditions.append(f"COALESCE({key}, '') = ?")
            params.append(_norm_scope_value(filters.get(key)))
        return " AND ".join(conditions), params

    # ------------------------------------------------------------------ #
    # extraction cursor
    # ------------------------------------------------------------------ #

    def _get_cursor_locked(
        self, user_id: str, agent_id: str, run_id: str, store: str
    ) -> int:
        """Cursor read; caller must hold ``self._lock``."""
        cur = self._connection.execute(
            """
            SELECT last_processed_seq FROM message_cursor
            WHERE user_id = ? AND agent_id = ? AND run_id = ? AND store = ?
            """,
            (user_id, agent_id, run_id, store),
        )
        row = cur.fetchone()
        return row[0] if row else 0

    def get_cursor(
        self, user_id: str, agent_id: str, run_id: str, store: str
    ) -> int:
        """Return ``last_processed_seq`` for the cursor key, 0 if unset."""
        with self._lock:
            return self._get_cursor_locked(
                _norm_scope_value(user_id),
                _norm_scope_value(agent_id),
                _norm_scope_value(run_id),
                store,
            )

    def advance_cursor(
        self,
        user_id: str,
        agent_id: str,
        run_id: str,
        store: str,
        last_processed_seq: int,
    ) -> bool:
        """Advance the cursor; refuses to move backwards (returns False)."""
        user_id = _norm_scope_value(user_id)
        agent_id = _norm_scope_value(agent_id)
        run_id = _norm_scope_value(run_id)
        with self._lock:
            current = self._get_cursor_locked(user_id, agent_id, run_id, store)
            if last_processed_seq <= current:
                return False
            self._connection.execute(
                """
                INSERT INTO message_cursor
                    (user_id, agent_id, run_id, store, last_processed_seq, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT (user_id, agent_id, run_id, store)
                DO UPDATE SET last_processed_seq = excluded.last_processed_seq,
                              updated_at = excluded.updated_at
                """,
                (user_id, agent_id, run_id, store, last_processed_seq, _utc_now()),
            )
            return True

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
        """Drop and recreate the messages table (full reset).

        Also drops the cursor table: cursors reference message seqs and would
        point at nonexistent rows after a reset.
        """
        with self._lock:
            self._connection.execute("DROP TABLE IF EXISTS messages")
            self._connection.execute("DROP TABLE IF EXISTS message_cursor")
            self._connection.execute(_MESSAGES_TABLE_SQL)
            self._connection.execute(_MESSAGES_INDEX_SQL)
            self._connection.execute(_CURSOR_TABLE_SQL)

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
