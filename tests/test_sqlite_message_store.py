"""Tests for the self-managed SQLite message store.

Run from the repository root with:

    pytest tests/test_sqlite_message_store.py -v
"""

import os
import sqlite3
import tempfile
from typing import Any, Dict, List

import pytest

from neatmem.storage.message.sqlite import SQLiteMessageStore, _MAX_MESSAGES_PER_SCOPE
from neatmem.storage.message.noop import NoOpMessageStore
from mem0_messages_simulator import Mem0MessagesSimulator


def _make_messages(n: int, prefix: str = "msg") -> List[Dict[str, Any]]:
    return [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"{prefix}-{i:02d}"}
        for i in range(n)
    ]


@pytest.fixture
def store():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    try:
        yield SQLiteMessageStore(path, extract_last_k=10)
    finally:
        if os.path.exists(path):
            os.unlink(path)


@pytest.fixture
def simulator():
    sim = Mem0MessagesSimulator()
    try:
        yield sim
    finally:
        sim.close()


@pytest.fixture
def small_retention(monkeypatch):
    """Monkey-patch retention limit to 5 for testing."""
    import neatmem.storage.message.sqlite as mod
    monkeypatch.setattr(mod, "_MAX_MESSAGES_PER_SCOPE", 5)
    yield


class TestGoldenParity:
    """Compare SQLiteMessageStore against the mem0 simulator."""

    def test_save_and_get_last_content_matches_mem0(self, store, simulator):
        filters = {"user_id": "alice", "agent_id": "a1"}
        messages = _make_messages(5)

        store.save_messages(messages, filters)
        simulator.save_messages(messages, filters)

        # mem0's get_last_messages orders by created_at DESC.  With identical
        # timestamps SQLite falls back to insertion order, so the exact order
        # may differ from NeatMem's oldest-first ordering.  We compare content.
        store_content = {r["content"] for r in store.get_last_messages(filters, limit=10)}
        sim_content = {r["content"] for r in simulator.get_last_messages(filters, limit=10)}
        assert store_content == sim_content == {m["content"] for m in messages}

    def test_retention_matches_mem0_with_unique_timestamps(
        self, store, simulator
    ):
        """With < 10 messages, no retention triggers in either system.

        Verifies content parity when retention is not in play.
        """
        filters = {"user_id": "bob"}
        import time
        for i in range(5):
            msg = {"role": "user", "content": f"msg-{i:02d}"}
            store.save_messages([msg], filters)
            simulator.save_messages([msg], filters)
            time.sleep(0.001)

        store_rows = store.get_last_messages(filters, limit=100)
        sim_rows = simulator.get_last_messages(filters, limit=100)
        assert len(store_rows) == len(sim_rows) == 5
        assert {r["content"] for r in store_rows} == {r["content"] for r in sim_rows}

    def test_retention_diverges_from_mem0_on_ties_by_design(
        self, store, simulator, small_retention
    ):
        """Same-timestamp batches expose mem0's unstable LIMIT behavior.

        NeatMem keeps the last 5 (seq DESC tiebreaker, retention=5 via monkeypatch).
        mem0 keeps all 10 (its hardcoded LIMIT 10).  Both are deterministic;
        NeatMem is simply stable by design.
        """
        filters = {"user_id": "tie"}
        messages = _make_messages(10)

        store.save_messages(messages, filters)
        simulator.save_messages(messages, filters)

        store_content = {r["content"] for r in store.get_last_messages(filters, limit=100)}
        sim_content = {r["content"] for r in simulator.get_last_messages(filters, limit=100)}
        assert len(store_content) == 5
        assert len(sim_content) == 10
        # NeatMem keeps the last 5 inserted
        assert store_content == {f"msg-{i:02d}" for i in range(5, 10)}
        # mem0 keeps all 10 (retention limit not triggered with 10 messages)
        assert sim_content == {f"msg-{i:02d}" for i in range(10)}


class TestScopeAndRetention:
    def test_app_id_isolation(self, store):
        store.save_messages(
            [{"role": "user", "content": "app-a"}],
            {"app_id": "app-a", "user_id": "u1"},
        )
        store.save_messages(
            [{"role": "user", "content": "app-b"}],
            {"app_id": "app-b", "user_id": "u1"},
        )

        a_msgs = store.get_last_messages({"app_id": "app-a", "user_id": "u1"})
        b_msgs = store.get_last_messages({"app_id": "app-b", "user_id": "u1"})

        assert [m["content"] for m in a_msgs] == ["app-a"]
        assert [m["content"] for m in b_msgs] == ["app-b"]

    def test_retention_keeps_newest(self, small_retention):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            s = SQLiteMessageStore(path, extract_last_k=100)
            filters = {"user_id": "retention"}
            s.save_messages(_make_messages(12), filters)
            rows = s.get_last_messages(filters, limit=100)
            assert len(rows) == 5
            assert [r["content"] for r in rows] == ["msg-07", "msg-08", "msg-09", "msg-10", "msg-11"]
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_timestamp_stability(self, small_retention):
        """Same-timestamp batch cleanup must be deterministic."""
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            s = SQLiteMessageStore(path, extract_last_k=10)
            filters = {"user_id": "stable"}
            s.save_messages(_make_messages(20), filters)
            first = [r["content"] for r in s.get_last_messages(filters, limit=100)]

            fd2, path2 = tempfile.mkstemp(suffix=".db")
            os.close(fd2)
            try:
                s2 = SQLiteMessageStore(path2, extract_last_k=10)
                s2.save_messages(_make_messages(20), filters)
                second = [r["content"] for r in s2.get_last_messages(filters, limit=100)]
                assert first == second
            finally:
                if os.path.exists(path2):
                    os.unlink(path2)
        finally:
            if os.path.exists(path):
                os.unlink(path)


class TestQueryAndCount:
    def test_query_and_count_basic(self, store):
        filters = {"user_id": "list"}
        store.save_messages(_make_messages(5, "x"), filters)

        assert store.count_messages(filters) == 5
        page = store.query_messages(filters, limit=2, offset=1, order="asc")
        assert len(page) == 2
        assert page[0]["content"] == "x-01"
        assert page[1]["content"] == "x-02"

    def test_query_descending(self, store):
        filters = {"user_id": "desc"}
        store.save_messages(_make_messages(5, "d"), filters)
        rows = store.query_messages(filters, order="desc")
        assert [r["content"] for r in rows] == ["d-04", "d-03", "d-02", "d-01", "d-00"]

    def test_query_default_order_is_desc(self, store):
        """query_messages default order is desc (reverse chronological)."""
        filters = {"user_id": "default-order"}
        store.save_messages(_make_messages(3, "do"), filters)
        rows = store.query_messages(filters)
        assert [r["content"] for r in rows] == ["do-02", "do-01", "do-00"]

    def test_query_content_like(self, store):
        filters = {"user_id": "like"}
        store.save_messages(
            [
                {"role": "user", "content": "I love Python"},
                {"role": "assistant", "content": "Python is great"},
                {"role": "user", "content": "I also like Rust"},
            ],
            filters,
        )
        rows = store.query_messages(filters, content_like="Python")
        assert len(rows) == 2
        assert all("Python" in r["content"] for r in rows)

    def test_query_roles_filter(self, store):
        filters = {"user_id": "roles"}
        store.save_messages(_make_messages(4, "r"), filters)
        rows = store.query_messages(filters, roles=["user"])
        assert all(r["role"] == "user" for r in rows)
        assert len(rows) == 2

    def test_query_after_before_closed_interval(self, store):
        """after/before use closed interval (>=, <=)."""
        import time
        filters = {"user_id": "time-range"}
        # Save first batch
        store.save_messages([{"role": "user", "content": "old"}], filters)
        ts1 = store.query_messages(filters, order="asc")[0]["created_at"]
        time.sleep(0.01)
        # Save second batch
        store.save_messages([{"role": "user", "content": "new"}], filters)
        ts2 = store.query_messages(filters, order="asc")[-1]["created_at"]

        # Query with after=ts1 should include ts1 (closed interval)
        rows = store.query_messages(filters, after=ts1, order="asc")
        assert len(rows) == 2  # both old and new

        # Query with before=ts1 should include ts1 (closed interval)
        rows = store.query_messages(filters, before=ts1, order="asc")
        assert len(rows) == 1
        assert rows[0]["content"] == "old"

    def test_count_with_content_like(self, store):
        filters = {"user_id": "count-like"}
        store.save_messages(
            [
                {"role": "user", "content": "hello world"},
                {"role": "user", "content": "hello python"},
                {"role": "user", "content": "goodbye"},
            ],
            filters,
        )
        assert store.count_messages(filters, content_like="hello") == 2
        assert store.count_messages(filters, content_like="goodbye") == 1
        assert store.count_messages(filters) == 3

    def test_query_returns_scope_fields(self, store):
        """query_messages returns message_id + 4 scope fields."""
        filters = {"app_id": "test-app", "user_id": "alice", "agent_id": "a1", "run_id": "r1"}
        store.save_messages([{"role": "user", "content": "msg"}], filters)
        rows = store.query_messages(filters)
        assert len(rows) == 1
        msg = rows[0]
        assert msg["message_id"]  # message_id is present
        assert msg["app_id"] == "test-app"
        assert msg["user_id"] == "alice"
        assert msg["agent_id"] == "a1"
        assert msg["run_id"] == "r1"
        assert msg["role"] == "user"
        assert msg["content"] == "msg"

    def test_empty_scope_returns_empty(self, store):
        """query_messages with empty scope returns empty (no full-table scan)."""
        store.save_messages(
            [{"role": "user", "content": "msg"}],
            {"user_id": "someone"},
        )
        assert store.query_messages({}) == []
        assert store.count_messages({}) == 0


class TestGetLastLimit:
    def test_get_last_uses_store_default_when_limit_is_none(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            s = SQLiteMessageStore(path, extract_last_k=3)
            filters = {"user_id": "default-k"}
            s.save_messages(_make_messages(10), filters)
            rows = s.get_last_messages(filters)
            assert len(rows) == 3
            assert [r["content"] for r in rows] == ["msg-07", "msg-08", "msg-09"]
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_get_last_explicit_limit_overrides_store_default(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            s = SQLiteMessageStore(path, extract_last_k=3)
            filters = {"user_id": "explicit-k"}
            s.save_messages(_make_messages(10), filters)
            rows = s.get_last_messages(filters, limit=5)
            assert len(rows) == 5
            assert [r["content"] for r in rows] == ["msg-05", "msg-06", "msg-07", "msg-08", "msg-09"]
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_get_last_zero_or_negative_limit_returns_empty(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            s = SQLiteMessageStore(path, extract_last_k=3)
            filters = {"user_id": "zero-k"}
            s.save_messages(_make_messages(5), filters)
            assert s.get_last_messages(filters, limit=0) == []
            assert s.get_last_messages(filters, limit=-1) == []
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_get_last_returns_scope_fields(self, store):
        """get_last_messages returns message_id + 4 scope fields."""
        filters = {"app_id": "a", "user_id": "u", "run_id": "r"}
        store.save_messages([{"role": "user", "content": "msg"}], filters)
        rows = store.get_last_messages(filters)
        assert len(rows) == 1
        msg = rows[0]
        assert msg["message_id"]
        assert msg["app_id"] == "a"
        assert msg["user_id"] == "u"
        assert msg["run_id"] == "r"


class TestListSessions:
    def test_list_sessions_basic(self, store):
        store.save_messages(
            [{"role": "user", "content": "msg1"}],
            {"user_id": "alice", "app_id": "app", "run_id": "sess-001"},
        )
        store.save_messages(
            [{"role": "user", "content": "msg2"}],
            {"user_id": "alice", "app_id": "app", "run_id": "sess-002"},
        )

        sessions = store.list_sessions({"user_id": "alice", "app_id": "app"})
        assert len(sessions) == 2
        run_ids = {s["run_id"] for s in sessions}
        assert run_ids == {"sess-001", "sess-002"}
        for s in sessions:
            assert s["last_active_at"]

    def test_list_sessions_excludes_null_run_id(self, store):
        """Messages without run_id should not appear as sessions."""
        store.save_messages(
            [{"role": "user", "content": "no-run"}],
            {"user_id": "alice", "app_id": "app"},
        )
        store.save_messages(
            [{"role": "user", "content": "with-run"}],
            {"user_id": "alice", "app_id": "app", "run_id": "sess-001"},
        )

        sessions = store.list_sessions({"user_id": "alice", "app_id": "app"})
        assert len(sessions) == 1
        assert sessions[0]["run_id"] == "sess-001"

    def test_list_sessions_empty_scope_returns_empty(self, store):
        assert store.list_sessions({}) == []

    def test_list_sessions_app_id_isolation(self, store):
        store.save_messages(
            [{"role": "user", "content": "a"}],
            {"user_id": "u", "app_id": "app-a", "run_id": "r1"},
        )
        store.save_messages(
            [{"role": "user", "content": "b"}],
            {"user_id": "u", "app_id": "app-b", "run_id": "r2"},
        )

        a_sessions = store.list_sessions({"user_id": "u", "app_id": "app-a"})
        b_sessions = store.list_sessions({"user_id": "u", "app_id": "app-b"})
        assert len(a_sessions) == 1
        assert a_sessions[0]["run_id"] == "r1"
        assert len(b_sessions) == 1
        assert b_sessions[0]["run_id"] == "r2"


class TestDeleteMessages:
    def test_delete_by_run_id(self, store):
        store.save_messages(
            [{"role": "user", "content": "msg1"}, {"role": "user", "content": "msg2"}],
            {"user_id": "alice", "app_id": "app", "run_id": "sess-001"},
        )
        store.save_messages(
            [{"role": "user", "content": "msg3"}],
            {"user_id": "alice", "app_id": "app", "run_id": "sess-002"},
        )

        deleted = store.delete_messages({"user_id": "alice", "run_id": "sess-001"})
        assert deleted == 2
        assert store.count_messages({"user_id": "alice"}) == 1

    def test_delete_by_user_id(self, store):
        store.save_messages(
            [{"role": "user", "content": "msg1"}],
            {"user_id": "alice"},
        )
        store.save_messages(
            [{"role": "user", "content": "msg2"}],
            {"user_id": "bob"},
        )

        deleted = store.delete_messages({"user_id": "alice"})
        assert deleted == 1
        assert store.count_messages({"user_id": "alice"}) == 0
        assert store.count_messages({"user_id": "bob"}) == 1

    def test_delete_empty_scope_returns_zero(self, store):
        """delete_messages with empty scope refuses to delete all."""
        store.save_messages(
            [{"role": "user", "content": "msg"}],
            {"user_id": "alice"},
        )
        deleted = store.delete_messages({})
        assert deleted == 0
        assert store.count_messages({"user_id": "alice"}) == 1


class TestReset:
    def test_reset_clears_all(self, store):
        store.save_messages(
            [{"role": "user", "content": "msg1"}],
            {"user_id": "alice"},
        )
        store.save_messages(
            [{"role": "user", "content": "msg2"}],
            {"user_id": "bob"},
        )
        assert store.count_messages({"user_id": "alice"}) == 1
        assert store.count_messages({"user_id": "bob"}) == 1

        store.reset()
        assert store.count_messages({"user_id": "alice"}) == 0
        assert store.count_messages({"user_id": "bob"}) == 0

    def test_reset_then_save_works(self, store):
        """After reset, the table is recreated and can accept new messages."""
        store.save_messages(
            [{"role": "user", "content": "old"}],
            {"user_id": "alice"},
        )
        store.reset()
        store.save_messages(
            [{"role": "user", "content": "new"}],
            {"user_id": "alice"},
        )
        msgs = store.get_last_messages({"user_id": "alice"})
        assert len(msgs) == 1
        assert msgs[0]["content"] == "new"


class TestNoOpMessageStore:
    def test_all_methods_noop(self):
        s = NoOpMessageStore(extract_last_k=10)
        s.save_messages([{"role": "user", "content": "msg"}], {"user_id": "u"})
        assert s.get_last_messages({"user_id": "u"}) == []
        assert s.query_messages({"user_id": "u"}) == []
        assert s.count_messages({"user_id": "u"}) == 0
        assert s.list_sessions({"user_id": "u"}) == []
        assert s.delete_messages({"user_id": "u"}) == 0
        s.reset()
        s.close()  # should not raise

    def test_extract_last_k_attribute(self):
        """NoOpMessageStore has extract_last_k for getattr fallback in add_memories."""
        s = NoOpMessageStore(extract_last_k=7)
        assert s.extract_last_k == 7


class TestSchemaAndEdgeCases:
    def test_schema_created(self, store):
        conn = sqlite3.connect(store.db_path)
        try:
            cur = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_messages_scope_time'"
            )
            assert cur.fetchone() is not None
            # Verify 5-column composite index
            cur = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='index' AND name='idx_messages_scope_time'"
            )
            sql = cur.fetchone()[0]
            for col in ("app_id", "user_id", "agent_id", "run_id", "created_at"):
                assert col in sql, f"index should include {col}"
        finally:
            conn.close()

    def test_message_id_column_exists(self, store):
        """Schema uses message_id (not id) for the unique text column."""
        conn = sqlite3.connect(store.db_path)
        try:
            cur = conn.execute("PRAGMA table_info(messages)")
            cols = {row[1] for row in cur.fetchall()}
            assert "message_id" in cols
            assert "id" not in cols  # old column name should not exist
            assert "session_scope" not in cols  # old scope string should not exist
            for scope_col in ("app_id", "user_id", "agent_id", "run_id"):
                assert scope_col in cols
        finally:
            conn.close()

    def test_empty_scope_save_is_ignored(self, store):
        store.save_messages([{"role": "user", "content": "orphan"}], {})
        assert store.get_last_messages({}) == []
        assert store.count_messages({}) == 0

    def test_close_idempotent_usage(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            s = SQLiteMessageStore(path)
            s.close()
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_hardcoded_retention_is_1000(self):
        """_MAX_MESSAGES_PER_SCOPE is hardcoded to 1000 (not configurable)."""
        assert _MAX_MESSAGES_PER_SCOPE == 1000
