"""Poison-batch guard tests (2026-09-23, plan: docs/internal-notes/20260922-batch-dedup-query-oversize-fix-plan.md).

Covers the three layers of the fix for the 2026-09-22 incident (a dedup query
over the embedding model's token limit deterministically 400s and poisons the
cursor retry loop):

1. memory_add._truncate_dedup_query — oversize batch query is truncated,
   undersize query is byte-identical (LOCOMO anchor safety).
2. config.resolve_embedding_max_tokens — model-name table, vendor-prefix
   normalization, explicit override precedence, unknown-model 512 fallback.
3. batching.process_scope_batch — circuit breaker: failures below the
   threshold retry without advancing; at the threshold the batch is skipped
   (cursor advances, error logged); success resets the counter.

Run: cd <repo root> && python -m pytest tests/test_poison_batch_guard.py -q
"""

import asyncio
import importlib
import json
import logging
import os
import tempfile

import pytest

import neatmem.config as config
from neatmem.batching import process_scope_batch
from neatmem.memory_add import _truncate_dedup_query
from neatmem.storage.message.sqlite import SQLiteMessageStore


# --- 1. dedup query truncation ---

class TestTruncateDedupQuery:
    def test_undersize_query_byte_identical(self):
        text = json.dumps([{"role": "user", "content": "短消息"}] * 10, ensure_ascii=False)
        assert len(text) <= config.EMBEDDING_MAX_TOKENS
        assert _truncate_dedup_query(text) == text

    def test_oversize_query_truncated_with_warning(self, caplog):
        text = json.dumps(
            [{"role": "user", "content": "长" * config.EMBEDDING_MAX_TOKENS}] * 3,
            ensure_ascii=False,
        )
        assert len(text) > config.EMBEDDING_MAX_TOKENS
        with caplog.at_level(logging.WARNING):
            out = _truncate_dedup_query(text)
        assert out == text[: config.EMBEDDING_MAX_TOKENS]
        assert "dedup query oversize" in caplog.text

    def test_boundary_not_truncated(self):
        text = "x" * config.EMBEDDING_MAX_TOKENS
        assert _truncate_dedup_query(text) == text


# --- 2. model-name -> max tokens resolution ---

class TestResolveEmbeddingMaxTokens:
    @pytest.mark.parametrize(
        "model,expected",
        [
            ("BAAI/bge-m3", 8192),
            ("bge-m3", 8192),
            ("siliconflow/BAAI/bge-m3", 8192),
            ("BAAI/bge-large-zh-v1.5", 512),
            ("netease-youdao/bce-embedding-base_v1", 512),
            ("text-embedding-ada-002", 8191),
            ("text-embedding-3-small", 8191),
            ("text-embedding-v3", 8192),
            ("text-embedding-v1", 2048),
            ("Qwen/Qwen3-Embedding-8B", 32768),
        ],
    )
    def test_table_hits(self, model, expected):
        assert config.resolve_embedding_max_tokens(model) == expected

    def test_unknown_model_falls_back_512_with_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            assert config.resolve_embedding_max_tokens("acme/future-embed-9") == 512
        assert "Unknown embedding model" in caplog.text

    def test_explicit_override_wins(self):
        assert config.resolve_embedding_max_tokens("BAAI/bge-m3", override=4096) == 4096
        assert config.resolve_embedding_max_tokens("acme/future-embed-9", override=1024) == 1024

    def test_config_module_wiring(self, monkeypatch, tmp_path):
        """Module-level EMBEDDING_MAX_TOKENS resolves from EMBEDDER_MODEL env,
        with explicit EMBEDDING_MAX_TOKENS taking precedence. Env is stubbed so
        the repo-root .env cannot leak in (same pattern as test_embedder_config)."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("NEATMEM_DIR", str(tmp_path / "data"))
        monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
        monkeypatch.setenv("EMBEDDER_MODEL", "BAAI/bge-large-zh-v1.5")
        monkeypatch.delenv("EMBEDDING_MAX_TOKENS", raising=False)
        assert importlib.reload(config).EMBEDDING_MAX_TOKENS == 512
        monkeypatch.setenv("EMBEDDING_MAX_TOKENS", "4096")
        assert importlib.reload(config).EMBEDDING_MAX_TOKENS == 4096
        monkeypatch.delenv("EMBEDDING_MAX_TOKENS", raising=False)
        monkeypatch.delenv("EMBEDDER_MODEL", raising=False)
        importlib.reload(config)


# --- 3. scheduler circuit breaker ---

def _make_scope(store, user_id="u1", agent_id="", run_id="r1"):
    filters = {"user_id": user_id, "agent_id": agent_id, "run_id": run_id}
    store.save_messages(
        [{"role": "user", "content": f"m{i}"} for i in range(10)], filters
    )
    return {"user_id": user_id, "agent_id": agent_id, "run_id": run_id}


class TestCircuitBreaker:
    @pytest.fixture
    def store(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            yield SQLiteMessageStore(path, extract_last_k=10)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def _run(self, store, scope, extract_batch, failures, max_failures=3):
        asyncio.run(
            process_scope_batch(
                store,
                scope,
                extract_batch=extract_batch,
                consecutive_failures=failures,
                batch_size=10,
                deadline_secs=600,
                max_consecutive_failures=max_failures,
            )
        )

    def _cursor(self, store, scope):
        return store.get_cursor(scope["user_id"], scope["agent_id"], scope["run_id"], "vector")

    def test_failure_below_threshold_retries_without_advancing(self, store):
        scope = _make_scope(store)

        async def boom(scope, ids, req_id):
            raise RuntimeError("deterministic 400")

        failures = {}
        self._run(store, scope, boom, failures)
        assert self._cursor(store, scope) == 0
        assert failures[("u1", "", "r1")] == 1

    def test_skip_at_threshold_advances_cursor(self, store, caplog):
        scope = _make_scope(store)

        async def boom(scope, ids, req_id):
            raise RuntimeError("deterministic 400")

        failures = {}
        with caplog.at_level(logging.ERROR):
            for _ in range(3):  # max_failures=3
                self._run(store, scope, boom, failures)
        # Cursor advanced past the poisoned batch; messages retained for replay.
        assert self._cursor(store, scope) == 10
        assert store.count_pending_messages(scope, 0) == 10
        assert failures[("u1", "", "r1")] == 0
        assert "poison batch skipped" in caplog.text

    def test_success_resets_counter(self, store):
        scope = _make_scope(store)
        calls = {"n": 0}

        async def flaky(scope, ids, req_id):
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("transient")

        failures = {}
        self._run(store, scope, flaky, failures)  # fail 1
        self._run(store, scope, flaky, failures)  # fail 2
        self._run(store, scope, flaky, failures)  # success
        assert self._cursor(store, scope) == 10
        assert ("u1", "", "r1") not in failures
