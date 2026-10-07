"""EXCLUDE_MEMORY_IDS_FILE 服务端候选池排除（补位语义）单测。

语义（20261005 计划 §5.6，2026-10-07 实施）：
- flag 未设置：零副作用（不读文件、不过滤、不放大 over-fetch）
- flag 设置：合并阶段排除指定 id（dense 各路 + bm25），rerank/top_k 截断之前
- over-fetch 放大 ×2 保持候选池满额：排除不缩小 internal_limit
"""

import os

import pytest

import neatmem.memory_search as ms


class _Hit:
    def __init__(self, rid, score, payload=None):
        self.id = rid
        self.score = score
        self.payload = payload or {"memory": f"text-{rid}"}


class _FakeVectorStore:
    """按请求的 top_k 返回头部 N 条，模拟 dense 检索。"""

    def __init__(self, ids):
        self.ids = ids
        self.requested_limits = []

    def search(self, query, vectors, top_k, filters):
        self.requested_limits.append(top_k)
        return [_Hit(rid, score=1.0 - i * 0.01) for i, rid in enumerate(self.ids[:top_k])]


class _FakeEmbed:
    def embed(self, text, mode):
        return [0.1]


class _FakeMemory:
    def __init__(self, ids):
        self.embedding_model = _FakeEmbed()
        self.vector_store = _FakeVectorStore(ids)


@pytest.fixture
def exclude_file(tmp_path):
    p = tmp_path / "exclude.txt"
    p.write_text("m3\nm5\n")
    return str(p)


@pytest.fixture(autouse=True)
def _clear_cache_and_env(monkeypatch):
    ms._exclude_ids_cache.clear()
    monkeypatch.delenv("EXCLUDE_MEMORY_IDS_FILE", raising=False)
    monkeypatch.delenv("INTERNAL_LIMIT", raising=False)
    yield
    ms._exclude_ids_cache.clear()


def _search(memory, top_k=3):
    return ms.search_memories(
        memory, query="q", filters={}, top_k=top_k,
        entity_extractor=None, entity_store=None,
        use_entity=False, use_bm25=False,
    )


def test_flag_off_zero_side_effect():
    mem = _FakeMemory(["m1", "m2", "m3", "m4", "m5", "m6"])
    out = _search(mem, top_k=3)
    assert [r["id"] for r in out["results"]] == ["m1", "m2", "m3"]
    # 未放大 over-fetch：max(3*4, 60) = 60
    assert mem.vector_store.requested_limits == [60]


def test_exclude_filters_and_backfills(monkeypatch, exclude_file):
    monkeypatch.setenv("EXCLUDE_MEMORY_IDS_FILE", exclude_file)
    ids = ["m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8"]
    mem = _FakeMemory(ids)
    out = _search(mem, top_k=3)
    got = [r["id"] for r in out["results"]]
    assert "m3" not in got and "m5" not in got
    # 补位：top_k 数量不变，后排候选顶上来
    assert got == ["m1", "m2", "m4"]
    assert len(out["results"]) == 3


def test_exclude_amplifies_overfetch(monkeypatch, exclude_file):
    monkeypatch.setenv("EXCLUDE_MEMORY_IDS_FILE", exclude_file)
    mem = _FakeMemory(["m1", "m2", "m3", "m4"])
    _search(mem, top_k=3)
    # 排除激活时 over-fetch ×2：60 → 120
    assert mem.vector_store.requested_limits == [120]


def test_exclude_missing_file_loud(monkeypatch):
    monkeypatch.setenv("EXCLUDE_MEMORY_IDS_FILE", "/nonexistent/ids.txt")
    mem = _FakeMemory(["m1"])
    with pytest.raises(OSError):
        _search(mem, top_k=1)


def test_exclude_bm25_hits(monkeypatch, exclude_file):
    from neatmem.signals.bm25.base import BM25SearchResult

    class _FakeBM25:
        def search(self, query, filters, top_k):
            return [BM25SearchResult(memory_id="m3", score=0.9),
                    BM25SearchResult(memory_id="m2", score=0.5)]

    monkeypatch.setenv("EXCLUDE_MEMORY_IDS_FILE", exclude_file)
    mem = _FakeMemory(["m1", "m2", "m3", "m4"])
    out = ms.search_memories(
        mem, query="q", filters={}, top_k=3,
        entity_extractor=None, entity_store=None,
        use_entity=False, use_bm25=True, bm25_index=_FakeBM25(),
    )
    got = [r["id"] for r in out["results"]]
    assert "m3" not in got


def test_exclude_ids_param_takes_precedence(monkeypatch, exclude_file):
    # 生产形态：参数传入淘汰集（ActivityStore.list_evicted_ids()），
    # 优先于 EXCLUDE_MEMORY_IDS_FILE。
    monkeypatch.setenv("EXCLUDE_MEMORY_IDS_FILE", exclude_file)  # excludes m3,m5
    ids = ["m1", "m2", "m3", "m4", "m5", "m6"]
    mem = _FakeMemory(ids)
    out = ms.search_memories(
        mem, query="q", filters={}, top_k=4,
        entity_extractor=None, entity_store=None,
        use_entity=False, use_bm25=False,
        exclude_ids={"m2"},  # param wins over the file's m3/m5
    )
    got = [r["id"] for r in out["results"]]
    assert "m2" not in got
    assert "m3" in got and "m5" in got
    assert len(got) == 4
