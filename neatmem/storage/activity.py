"""Activity store: append-only event stream + eviction-gate projection.

Plan: docs/internal-notes/20261005-citation-feedback-memory-importance-plan.md §4.

Two tables in ``activity.db`` (own DB, not history.db: single owner, write-lock
isolation from the hot read path, independent lifecycle — activity can be
rebuilt from scratch):

- ``events`` (P0): generic append-only event stream. All system behavior facts
  (search / injection / judgment / future custom kinds). New offline task =
  new kind, zero schema migration. Re-running a judge simply appends more
  judgment events; nothing is overwritten.
- ``memory_feedback`` (P2): eviction-gate projection keyed by memory_id.
  Rebuildable from ``events`` at any time (pure arithmetic replay), so rule
  changes (N) are a rebuild, not a migration. ``eviction_state`` keeps the
  *source* of the eviction — 'derived' (rule-decided, recomputed on rebuild)
  vs 'manual' (user-evicted via CLI, survives rebuilds) — so a projection
  rebuild can never silently wipe a human decision.

Judgment events are written per memory (one event per judged memory,
subject_id = memory_id) so ``idx_events_subject`` answers "all judgments of
memory X" with an indexed lookup instead of a JSON scan.
"""

import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

# Event kinds (open set; new offline tasks add new kinds without migration).
KIND_SEARCH = "search"
KIND_INJECTION = "injection"
KIND_JUDGMENT = "judgment"
# Mutual-exclusion claim between the serve auto-judge thread and manual
# `feedback judge` runs (plan 20261010 §3.2): a runner claims a pending
# injection before judging it so two runners never double-judge. Claims are
# ts-expiring (crash-safe: a dead runner's claim goes stale and the injection
# becomes claimable again); a judgment event ends the claim's relevance.
KIND_JUDGMENT_CLAIM = "judgment_claim"

# A claim covers one injection's judge time (a few chunked LLM calls), not a
# whole batch — 30 min is generous. Not an env knob (plan 20261010: fixed
# constants over tunables).
CLAIM_TTL_SECONDS = 1800

# Verdict vocabulary (judgment payload).
VERDICT_USED = "used"
VERDICT_UNUSED = "unused"

# Eviction state vocabulary (memory_feedback.eviction_state).
EVICTION_NONE = "none"
EVICTION_DERIVED = "derived"
EVICTION_MANUAL = "manual"

_EVENTS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         DATETIME NOT NULL,
    kind       TEXT NOT NULL,
    app_id     TEXT,
    user_id    TEXT,
    agent_id   TEXT,
    run_id     TEXT,
    subject_id TEXT,
    payload    TEXT NOT NULL
)
"""

_EVENTS_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_events_kind_scope_ts
ON events(kind, app_id, user_id, agent_id, run_id, ts)
"""

_EVENTS_SUBJECT_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_events_subject
ON events(subject_id, kind)
"""

_FEEDBACK_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS memory_feedback (
    memory_id      TEXT PRIMARY KEY,
    inject_count   INTEGER NOT NULL DEFAULT 0,
    used_count     INTEGER NOT NULL DEFAULT 0,
    eviction_state TEXT NOT NULL DEFAULT 'none'
                   CHECK (eviction_state IN ('none', 'derived', 'manual')),
    updated_at     DATETIME NOT NULL
)
"""


class ActivityStore:
    """SQLite-backed activity store (events + eviction-gate projection)."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(
            db_path,
            check_same_thread=False,
            isolation_level=None,  # autocommit; transactions managed explicitly
        )
        self._connection.row_factory = sqlite3.Row
        self._ensure_schema()
        logger.info("ActivityStore initialized: %s", db_path)

    def _ensure_schema(self) -> None:
        with self._lock:
            self._connection.execute(_EVENTS_TABLE_SQL)
            self._connection.execute(_EVENTS_INDEX_SQL)
            self._connection.execute(_EVENTS_SUBJECT_INDEX_SQL)
            self._connection.execute(_FEEDBACK_TABLE_SQL)

    # ------------------------------------------------------------------ events

    def record_event(
        self,
        kind: str,
        *,
        app_id: Optional[str] = None,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        subject_id: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Append one event; returns the new row id."""
        ts = datetime.now(timezone.utc).isoformat()
        with self._lock:
            cur = self._connection.execute(
                "INSERT INTO events (ts, kind, app_id, user_id, agent_id, run_id,"
                " subject_id, payload) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (ts, kind, app_id, user_id, agent_id, run_id, subject_id,
                 json.dumps(payload or {}, ensure_ascii=False)),
            )
            return int(cur.lastrowid)

    def iter_events(
        self,
        kind: Optional[str] = None,
        *,
        user_id: Optional[str] = None,
        subject_id: Optional[str] = None,
    ) -> Iterable[Dict[str, Any]]:
        """Stream events (id-ascending = chronological) as dicts."""
        clauses: List[str] = []
        params: List[Any] = []
        if kind is not None:
            clauses.append("kind = ?")
            params.append(kind)
        if user_id is not None:
            clauses.append("user_id = ?")
            params.append(user_id)
        if subject_id is not None:
            clauses.append("subject_id = ?")
            params.append(subject_id)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        with self._lock:
            rows = self._connection.execute(
                f"SELECT * FROM events{where} ORDER BY id", params
            ).fetchall()
        for row in rows:
            yield {
                "id": row["id"],
                "ts": row["ts"],
                "kind": row["kind"],
                "app_id": row["app_id"],
                "user_id": row["user_id"],
                "agent_id": row["agent_id"],
                "run_id": row["run_id"],
                "subject_id": row["subject_id"],
                "payload": json.loads(row["payload"]),
            }

    def pending_injections(self) -> List[Dict[str, Any]]:
        """Injection events with no judgment event yet (judge-batch queue).

        A judgment links back via payload.injection_event_id. no_window
        injections are marked judged with an empty verdict map by the batch
        runner, so they leave the queue without an LLM call.
        """
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT i.* FROM events i
                WHERE i.kind = ?
                  AND NOT EXISTS (
                      SELECT 1 FROM events j
                      WHERE j.kind = ?
                        AND json_extract(j.payload, '$.injection_event_id') = i.id
                  )
                ORDER BY i.id
                """,
                (KIND_INJECTION, KIND_JUDGMENT),
            ).fetchall()
        return [
            {
                "id": row["id"],
                "ts": row["ts"],
                "app_id": row["app_id"],
                "user_id": row["user_id"],
                "agent_id": row["agent_id"],
                "run_id": row["run_id"],
                "payload": json.loads(row["payload"]),
            }
            for row in rows
        ]

    # ------------------------------------------------------- judgment claims

    def claim_injection(
        self, injection_event_id: int, ttl_seconds: int = CLAIM_TTL_SECONDS
    ) -> bool:
        """Claim one pending injection for judging; False if already done/claimed.

        One atomic critical section: already-judged (a judgment event exists)
        and live-claimed injections both refuse. Claim events are append-only
        like everything else; expiry is computed at read time, so a crashed
        runner's claim goes stale on its own — no sweeper needed.

        BEGIN IMMEDIATE takes the SQLite write lock up front, so the
        check-then-insert is atomic across processes too (serve thread vs
        manual CLI) — a lost race would double-count inject/used, not just
        waste an LLM call.
        """
        now = datetime.now(timezone.utc)
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                judged = self._connection.execute(
                    "SELECT 1 FROM events WHERE kind = ?"
                    " AND json_extract(payload, '$.injection_event_id') = ? LIMIT 1",
                    (KIND_JUDGMENT, injection_event_id),
                ).fetchone()
                if judged:
                    self._connection.execute("ROLLBACK")
                    return False
                claims = self._connection.execute(
                    "SELECT ts FROM events WHERE kind = ?"
                    " AND json_extract(payload, '$.injection_event_id') = ?",
                    (KIND_JUDGMENT_CLAIM, injection_event_id),
                ).fetchall()
                for row in claims:
                    claimed_at = datetime.fromisoformat(row["ts"])
                    if (now - claimed_at).total_seconds() < ttl_seconds:
                        self._connection.execute("ROLLBACK")
                        return False
                self._connection.execute(
                    "INSERT INTO events (ts, kind, subject_id, payload)"
                    " VALUES (?, ?, ?, ?)",
                    (
                        now.isoformat(),
                        KIND_JUDGMENT_CLAIM,
                        str(injection_event_id),
                        json.dumps({"injection_event_id": injection_event_id}),
                    ),
                )
                self._connection.execute("COMMIT")
                return True
            except Exception:
                self._connection.execute("ROLLBACK")
                raise

    def live_judge_claims(
        self, ttl_seconds: int = CLAIM_TTL_SECONDS
    ) -> List[int]:
        """Injection event ids with a live claim and no judgment yet."""
        now = datetime.now(timezone.utc)
        with self._lock:
            rows = self._connection.execute(
                "SELECT id, ts, payload FROM events WHERE kind = ?",
                (KIND_JUDGMENT_CLAIM,),
            ).fetchall()
            judged_rows = self._connection.execute(
                "SELECT payload FROM events WHERE kind = ?", (KIND_JUDGMENT,)
            ).fetchall()
        judged_ids = {
            json.loads(r["payload"]).get("injection_event_id") for r in judged_rows
        }
        live: List[int] = []
        for row in rows:
            inj_id = json.loads(row["payload"]).get("injection_event_id")
            if inj_id is None or inj_id in judged_ids:
                continue
            claimed_at = datetime.fromisoformat(row["ts"])
            if (now - claimed_at).total_seconds() < ttl_seconds:
                live.append(inj_id)
        return sorted(set(live))

    # ------------------------------------------------------- memory_feedback

    def upsert_feedback_counts(
        self,
        memory_id: str,
        *,
        inject_delta: int = 0,
        used_delta: int = 0,
    ) -> None:
        """Increment the double counter for one memory (insert if absent)."""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._connection.execute(
                """
                INSERT INTO memory_feedback (memory_id, inject_count, used_count, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(memory_id) DO UPDATE SET
                    inject_count = inject_count + excluded.inject_count,
                    used_count   = used_count + excluded.used_count,
                    updated_at   = excluded.updated_at
                """,
                (memory_id, inject_delta, used_delta, now),
            )

    def get_feedback(self, memory_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM memory_feedback WHERE memory_id = ?", (memory_id,)
            ).fetchone()
        return dict(row) if row else None

    def set_eviction(self, memory_id: str, state: str) -> None:
        """Set eviction_state ('derived' | 'manual' | 'none')."""
        if state not in (EVICTION_NONE, EVICTION_DERIVED, EVICTION_MANUAL):
            raise ValueError(f"invalid eviction state: {state!r}")
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._connection.execute(
                """
                INSERT INTO memory_feedback (memory_id, eviction_state, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(memory_id) DO UPDATE SET
                    eviction_state = excluded.eviction_state,
                    updated_at     = excluded.updated_at
                """,
                (memory_id, state, now),
            )

    def list_feedback(
        self,
        *,
        eviction_state: Optional[str] = None,
        min_injections: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """List projection rows, optionally filtered (status/evictable views)."""
        clauses: List[str] = []
        params: List[Any] = []
        if eviction_state is not None:
            clauses.append("eviction_state = ?")
            params.append(eviction_state)
        if min_injections is not None:
            clauses.append("inject_count >= ?")
            params.append(min_injections)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        with self._lock:
            rows = self._connection.execute(
                f"SELECT * FROM memory_feedback{where} ORDER BY inject_count DESC",
                params,
            ).fetchall()
        return [dict(row) for row in rows]

    def list_evicted_ids(self) -> List[str]:
        """Ids currently demoted (derived or manual) — the recall filter set."""
        with self._lock:
            rows = self._connection.execute(
                "SELECT memory_id FROM memory_feedback WHERE eviction_state <> 'none'"
            ).fetchall()
        return [row["memory_id"] for row in rows]

    def close(self) -> None:
        with self._lock:
            self._connection.close()
