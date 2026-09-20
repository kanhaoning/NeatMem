"""Tests for the event_at chain (same-session embargo plan §3.7).

Chain: plugin uploads packet-level ``event_at`` per message -> messages table
``event_at`` column -> batch min -> ``metadata["timestamp"]`` on extracted
memories -> surfaced to clients by ``_format_candidate``.

Plan: docs/internal-notes/20260920-same-session-embargo-plan.md

Run from the repository root with:

    pytest tests/test_event_at.py -v
"""

import os
import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone

import pytest

from neatmem.batching import batch_event_at
from neatmem.memory_search import _format_candidate
from neatmem.storage.message.sqlite import SQLiteMessageStore


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


@pytest.fixture
def store():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    try:
        yield SQLiteMessageStore(path, extract_last_k=10)
    finally:
        if os.path.exists(path):
            os.unlink(path)


class TestEventAtColumn:
    def test_schema_has_event_at_column(self, store):
        conn = sqlite3.connect(store.db_path)
        try:
            cols = {row[1] for row in conn.execute("PRAGMA table_info(messages)")}
            assert "event_at" in cols
        finally:
            conn.close()

    def test_save_and_read_event_at(self, store):
        filters = {"user_id": "u"}
        ts = _iso(datetime.now(timezone.utc) - timedelta(minutes=5))
        store.save_messages(
            [
                {"role": "user", "content": "with-event", "event_at": ts},
                {"role": "user", "content": "without-event"},
            ],
            filters,
        )
        rows = store.get_last_messages(filters, limit=10)
        by_content = {r["content"]: r for r in rows}
        assert by_content["with-event"]["event_at"] == ts
        assert by_content["without-event"]["event_at"] is None

        # query_messages and get_messages_by_ids expose the column too
        queried = store.query_messages(filters, order="asc")
        assert queried[0]["event_at"] == ts
        ids = [r["message_id"] for r in rows]
        by_ids = store.get_messages_by_ids(ids)
        assert {r["content"]: r["event_at"] for r in by_ids} == {
            "with-event": ts,
            "without-event": None,
        }

    def test_old_schema_db_is_migrated(self):
        """A messages table without event_at gets the column via ALTER TABLE."""
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            conn = sqlite3.connect(path)
            conn.execute(
                """
                CREATE TABLE messages (
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
            )
            conn.execute(
                "INSERT INTO messages (message_id, user_id, role, content, created_at)"
                " VALUES ('old-1', 'u', 'user', 'legacy', '2026-01-01T00:00:00+00:00')"
            )
            conn.commit()
            conn.close()

            s = SQLiteMessageStore(path, extract_last_k=10)
            cols = {row[1] for row in s._connection.execute("PRAGMA table_info(messages)")}
            assert "event_at" in cols

            rows = s.get_last_messages({"user_id": "u"}, limit=10)
            assert len(rows) == 1
            assert rows[0]["event_at"] is None  # pre-migration rows stay NULL

            # and the migrated table accepts new writes with event_at
            ts = _iso(datetime.now(timezone.utc))
            s.save_messages(
                [{"role": "user", "content": "new", "event_at": ts}],
                {"user_id": "u"},
            )
            contents = {r["content"]: r["event_at"] for r in s.get_last_messages({"user_id": "u"}, limit=10)}
            assert contents == {"legacy": None, "new": ts}
        finally:
            if os.path.exists(path):
                os.unlink(path)


class TestEventAtValidation:
    def test_future_event_at_is_dropped(self, store):
        """event_at beyond server now + 60s is discarded (NULL), not stored."""
        filters = {"user_id": "u"}
        future = _iso(datetime.now(timezone.utc) + timedelta(hours=1))
        store.save_messages(
            [{"role": "user", "content": "future", "event_at": future}],
            filters,
        )
        rows = store.get_last_messages(filters, limit=10)
        assert rows[0]["event_at"] is None

    def test_near_future_event_at_within_tolerance_is_kept(self, store):
        filters = {"user_id": "u"}
        soon = _iso(datetime.now(timezone.utc) + timedelta(seconds=30))
        store.save_messages(
            [{"role": "user", "content": "soon", "event_at": soon}],
            filters,
        )
        rows = store.get_last_messages(filters, limit=10)
        assert rows[0]["event_at"] == soon

    def test_unparseable_event_at_is_dropped(self, store):
        store.save_messages(
            [{"role": "user", "content": "bad", "event_at": "not-a-timestamp"}],
            {"user_id": "u"},
        )
        rows = store.get_last_messages({"user_id": "u"}, limit=10)
        assert rows[0]["event_at"] is None


class TestBatchEventAt:
    def test_min_of_mixed_batch(self):
        rows = [
            {"event_at": "2026-09-20T10:05:00+00:00", "created_at": "2026-09-20T10:06:00+00:00"},
            {"event_at": "2026-09-20T10:01:00+00:00", "created_at": "2026-09-20T10:06:00+00:00"},
            {"event_at": "2026-09-20T10:03:00+00:00", "created_at": "2026-09-20T10:06:00+00:00"},
        ]
        assert batch_event_at(rows) == "2026-09-20T10:01:00+00:00"

    def test_null_event_at_falls_back_to_created_at(self):
        rows = [
            {"event_at": None, "created_at": "2026-09-20T10:02:00+00:00"},
            {"event_at": "2026-09-20T10:01:00+00:00", "created_at": "2026-09-20T10:06:00+00:00"},
        ]
        assert batch_event_at(rows) == "2026-09-20T10:01:00+00:00"

    def test_all_null_uses_min_created_at(self):
        rows = [
            {"event_at": None, "created_at": "2026-09-20T10:04:00+00:00"},
            {"event_at": None, "created_at": "2026-09-20T10:02:00+00:00"},
        ]
        assert batch_event_at(rows) == "2026-09-20T10:02:00+00:00"

    def test_empty_batch_returns_none(self):
        assert batch_event_at([]) is None


class TestFormatCandidateTimestamp:
    def test_flat_payload_timestamp_is_surfaced_into_metadata(self):
        """mem0 flattens metadata into the payload top level; the timestamp
        must be collected back into the returned metadata dict."""
        cand = {
            "id": "abc",
            "score": 0.9,
            "payload": {
                "data": "some memory",
                "created_at": "2026-09-20T10:00:00+00:00",
                "timestamp": "2026-09-20T09:58:00+00:00",
            },
        }
        out = _format_candidate(cand)
        assert out["metadata"]["timestamp"] == "2026-09-20T09:58:00+00:00"

    def test_no_timestamp_leaves_metadata_empty(self):
        cand = {
            "id": "abc",
            "score": 0.9,
            "payload": {"data": "some memory", "created_at": "2026-09-20T10:00:00+00:00"},
        }
        out = _format_candidate(cand)
        assert out["metadata"] == {}
