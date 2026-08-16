"""Tests for cursor-driven message batching (queue mode).

Covers the store-level mechanics behind /v1/messages/add|next-batch|mark-processed/:
save_messages returning IDs+seqs, before_seq context fetch, pending queries,
cursor advance/regression, and the compute_next_batch policy.

Run from the repository root with:

    pytest tests/test_message_batch_cursor.py -v
"""

import os
import tempfile
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import pytest

from neatmem.batching import FlushConflictError, compute_next_batch, flush_scope
from neatmem.storage.message.sqlite import SQLiteMessageStore


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


class TestSaveReturnsIdsAndSeqs:
    def test_returns_message_id_and_seq_in_order(self, store):
        saved = store.save_messages(_make_messages(3), {"user_id": "u1"})
        assert len(saved) == 3
        assert all(set(m) == {"message_id", "seq"} for m in saved)
        assert [m["seq"] for m in saved] == [1, 2, 3]

    def test_empty_scope_returns_empty_list(self, store):
        assert store.save_messages(_make_messages(2), {}) == []

    def test_empty_messages_returns_empty_list(self, store):
        assert store.save_messages([], {"user_id": "u1"}) == []


class TestGetLastMessagesBeforeSeq:
    def test_before_seq_excludes_batch_itself(self, store):
        filters = {"user_id": "u1"}
        store.save_messages(_make_messages(5, "ctx"), filters)
        batch = store.save_messages(_make_messages(2, "batch"), filters)
        last_k = store.get_last_messages(filters, limit=10, before_seq=batch[0]["seq"])
        assert [m["content"] for m in last_k] == [f"ctx-{i:02d}" for i in range(5)]

    def test_before_seq_respects_limit(self, store):
        filters = {"user_id": "u1"}
        saved = store.save_messages(_make_messages(10), filters)
        last_k = store.get_last_messages(filters, limit=3, before_seq=saved[8]["seq"])
        assert [m["content"] for m in last_k] == ["msg-05", "msg-06", "msg-07"]

    def test_no_before_seq_unchanged(self, store):
        filters = {"user_id": "u1"}
        store.save_messages(_make_messages(4), filters)
        assert len(store.get_last_messages(filters, limit=10)) == 4


class TestGetMessagesByIds:
    def test_returns_ordered_by_seq(self, store):
        saved = store.save_messages(_make_messages(4), {"user_id": "u1"})
        rows = store.get_messages_by_ids([saved[3]["message_id"], saved[1]["message_id"]])
        assert [r["seq"] for r in rows] == [2, 4]
        assert rows[0]["content"] == "msg-01"

    def test_missing_ids_omitted(self, store):
        saved = store.save_messages(_make_messages(2), {"user_id": "u1"})
        rows = store.get_messages_by_ids([saved[0]["message_id"], "nonexistent"])
        assert len(rows) == 1

    def test_empty_input(self, store):
        assert store.get_messages_by_ids([]) == []


class TestPendingQueries:
    def test_pending_after_seq(self, store):
        filters = {"user_id": "u1", "agent_id": "a1"}
        store.save_messages(_make_messages(5), filters)
        pending = store.get_pending_messages(filters, after_seq=2, limit=10)
        assert [m["seq"] for m in pending] == [3, 4, 5]
        assert store.count_pending_messages(filters, after_seq=2) == 3
        assert "content" not in pending[0]  # scheduling does not need content

    def test_pending_limit(self, store):
        filters = {"user_id": "u1"}
        store.save_messages(_make_messages(5), filters)
        pending = store.get_pending_messages(filters, after_seq=0, limit=2)
        assert [m["seq"] for m in pending] == [1, 2]

    def test_scope_is_null_insensitive(self, store):
        # Saved with run_id=None; queried with run_id="" must match.
        store.save_messages(_make_messages(3), {"user_id": "u1"})
        pending = store.get_pending_messages(
            {"user_id": "u1", "agent_id": "", "run_id": ""}, after_seq=0, limit=10
        )
        assert len(pending) == 3

    def test_scope_isolation(self, store):
        store.save_messages(_make_messages(2, "a"), {"user_id": "u1"})
        store.save_messages(_make_messages(3, "b"), {"user_id": "u2"})
        assert store.count_pending_messages({"user_id": "u1"}, after_seq=0) == 2
        assert store.count_pending_messages({"user_id": "u2"}, after_seq=0) == 3


class TestListScopes:
    def test_distinct_scopes_nulls_normalized(self, store):
        store.save_messages(_make_messages(1), {"user_id": "u1"})
        store.save_messages(_make_messages(1), {"user_id": "u1", "run_id": "s1"})
        store.save_messages(_make_messages(1), {"user_id": "u2"})
        scopes = store.list_message_scopes()
        assert {tuple(sorted(s.items())) for s in scopes} == {
            tuple(sorted({"user_id": "u1", "agent_id": "", "run_id": ""}.items())),
            tuple(sorted({"user_id": "u1", "agent_id": "", "run_id": "s1"}.items())),
            tuple(sorted({"user_id": "u2", "agent_id": "", "run_id": ""}.items())),
        }


class TestCursor:
    def test_default_zero(self, store):
        assert store.get_cursor("u1", "", "", "vector") == 0

    def test_advance_and_read_back(self, store):
        assert store.advance_cursor("u1", "", "", "vector", 10) is True
        assert store.get_cursor("u1", "", "", "vector") == 10

    def test_regression_rejected(self, store):
        store.advance_cursor("u1", "", "", "vector", 10)
        assert store.advance_cursor("u1", "", "", "vector", 10) is False
        assert store.advance_cursor("u1", "", "", "vector", 5) is False
        assert store.get_cursor("u1", "", "", "vector") == 10

    def test_tracks_are_independent(self, store):
        store.advance_cursor("u1", "", "", "vector", 10)
        assert store.get_cursor("u1", "", "", "graph") == 0
        assert store.advance_cursor("u1", "", "", "graph", 3) is True
        assert store.get_cursor("u1", "", "", "vector") == 10

    def test_reset_clears_cursor(self, store):
        store.save_messages(_make_messages(2), {"user_id": "u1"})
        store.advance_cursor("u1", "", "", "vector", 2)
        store.reset()
        assert store.get_cursor("u1", "", "", "vector") == 0
        assert store.list_message_scopes() == []


class TestComputeNextBatch:
    def test_full_batch_when_enough_pending(self, store):
        filters = {"user_id": "u1"}
        store.save_messages(_make_messages(12), filters)
        batch = compute_next_batch(
            store, filters, batch_size=10, deadline_secs=600
        )
        assert batch["seqs"] == list(range(1, 11))
        assert len(batch["message_ids"]) == 10
        assert batch["pending_count"] == 12

    def test_empty_when_below_size_and_fresh(self, store):
        store.save_messages(_make_messages(5), {"user_id": "u1"})
        batch = compute_next_batch(
            store, {"user_id": "u1"}, batch_size=10, deadline_secs=600
        )
        assert batch["message_ids"] == []
        assert batch["pending_count"] == 5

    def test_deadline_flushes_partial_batch(self, store):
        store.save_messages(_make_messages(5), {"user_id": "u1"})
        future = datetime.now(timezone.utc) + timedelta(seconds=1000)
        batch = compute_next_batch(
            store, {"user_id": "u1"}, batch_size=10, deadline_secs=600, now=future
        )
        assert batch["seqs"] == [1, 2, 3, 4, 5]

    def test_respects_cursor(self, store):
        filters = {"user_id": "u1"}
        store.save_messages(_make_messages(15), filters)
        store.advance_cursor("u1", "", "", "vector", 10)
        batch = compute_next_batch(
            store, filters, batch_size=10, deadline_secs=600
        )
        assert batch["pending_count"] == 5
        # Below batch size and fresh -> empty even though 5 pending.
        assert batch["message_ids"] == []


class TestFlushScope:
    @staticmethod
    def _run(coro):
        import asyncio
        return asyncio.run(coro)

    def test_flushes_partial_batches(self, store):
        filters = {"user_id": "u1"}
        saved = store.save_messages(_make_messages(23), filters)
        calls = []

        async def extract(scope, ids):
            calls.append(list(ids))

        result = self._run(flush_scope(store, filters, batch_size=10, extract_batch=extract))
        assert result["batches"] == 3  # 10 + 10 + 3 (partial final chunk)
        assert result["extracted_count"] == 23
        assert result["last_processed_seq"] == 23
        assert [len(c) for c in calls] == [10, 10, 3]
        # Extracted IDs match the stored messages in order.
        all_ids = [m["message_id"] for m in saved]
        assert [i for c in calls for i in c] == all_ids
        assert store.get_cursor("u1", "", "", "vector") == 23

    def test_empty_scope_is_noop(self, store):
        async def extract(scope, ids):
            raise AssertionError("must not be called")

        result = self._run(flush_scope(
            store, {"user_id": "nobody"}, batch_size=10, extract_batch=extract
        ))
        assert result == {"batches": 0, "extracted_count": 0, "last_processed_seq": 0}

    def test_extraction_failure_keeps_cursor(self, store):
        filters = {"user_id": "u1"}
        store.save_messages(_make_messages(15), filters)
        calls = []

        async def extract(scope, ids):
            calls.append(list(ids))
            if len(calls) == 2:
                raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            self._run(flush_scope(store, filters, batch_size=10, extract_batch=extract))
        # First batch committed; failed batch stays pending.
        assert store.get_cursor("u1", "", "", "vector") == 10
        assert store.count_pending_messages(filters, after_seq=10) == 5

    def test_concurrent_cursor_move_raises(self, store):
        filters = {"user_id": "u1"}
        store.save_messages(_make_messages(12), filters)

        async def extract(scope, ids):
            # Simulate the scheduler advancing the cursor mid-flush.
            store.advance_cursor("u1", "", "", "vector", 99)

        with pytest.raises(FlushConflictError):
            self._run(flush_scope(store, filters, batch_size=10, extract_batch=extract))
