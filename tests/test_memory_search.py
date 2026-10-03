"""Unit tests for neatmem.memory_search multi-query merge (plan §3.1, R4/R6-d)."""

from types import SimpleNamespace

from neatmem.memory_search import search_memories


class FakeEmbedding:
    def embed(self, query, action):
        return [0.1, 0.2]


class FakeVectorStore:
    """Returns per-query fixture hits: {query: [(id, score), ...]}."""

    def __init__(self, fixtures):
        self.fixtures = fixtures
        self.searched_queries = []

    def search(self, query, vectors, top_k, filters):
        self.searched_queries.append(query)
        return [
            SimpleNamespace(id=rid, score=score, payload={"data": f"mem-{rid}"})
            for rid, score in self.fixtures.get(query, [])
        ]


class FakeMemory:
    def __init__(self, fixtures):
        self.embedding_model = FakeEmbedding()
        self.vector_store = FakeVectorStore(fixtures)


def _search(memory, query, extra_queries=None, bm25_index=None):
    return search_memories(
        memory=memory,
        query=query,
        filters={"user_id": "u"},
        top_k=10,
        threshold=0.0,
        use_entity=False,
        use_bm25=bm25_index is not None,
        bm25_index=bm25_index,
        extra_queries=extra_queries,
    )


def test_single_query_unchanged_shape():
    mem = FakeMemory({"q": [("a", 0.9), ("b", 0.8)]})
    result = _search(mem, "q")
    assert [r["id"] for r in result["results"]] == ["a", "b"]
    assert result["total_candidates"] == 2
    assert result["query_sources"] == {"a": "q", "b": "q"}


def test_merge_takes_max_score_and_provenance_tracks_winner():
    mem = FakeMemory({
        "q": [("a", 0.7), ("b", 0.9)],
        "exp1": [("a", 0.95), ("c", 0.6)],
    })
    result = _search(mem, "q", extra_queries=["exp1"])
    by_id = {r["id"]: r for r in result["results"]}
    assert by_id["a"]["score"] == 0.95
    assert result["query_sources"]["a"] == "exp1"   # expansion won the max
    assert result["query_sources"]["b"] == "q"
    assert result["query_sources"]["c"] == "exp1"
    assert result["total_candidates"] == 3


def test_duplicate_expansion_not_researched():
    mem = FakeMemory({"q": [("a", 0.9)]})
    result = _search(mem, "q", extra_queries=["q", " q ", "", None])
    assert mem.vector_store.searched_queries == ["q"]
    assert result["total_candidates"] == 1


def test_merged_results_are_superset_of_single_query_top_hits():
    """R4 safety net: merging expansions must never evict a memory the
    original query alone would have surfaced (same threshold)."""
    fixtures = {
        "q": [("a", 0.9), ("b", 0.85), ("c", 0.8)],
        "exp1": [("d", 0.95)],
        "exp2": [("e", 0.92)],
    }
    single = _search(FakeMemory(fixtures), "q")
    merged = _search(FakeMemory(fixtures), "q", extra_queries=["exp1", "exp2"])
    single_ids = {r["id"] for r in single["results"]}
    merged_ids = {r["id"] for r in merged["results"]}
    assert single_ids <= merged_ids
    assert merged_ids == {"a", "b", "c", "d", "e"}


def test_bm25_runs_on_original_query_only():
    class StubBM25:
        def __init__(self):
            self.queries = []

        def search(self, query, filters, top_k):
            self.queries.append(query)
            return []

    bm25 = StubBM25()
    mem = FakeMemory({"q": [("a", 0.9)], "exp1": [("b", 0.8)]})
    _search(mem, "q", extra_queries=["exp1"], bm25_index=bm25)
    assert bm25.queries == ["q"]


def test_extra_queries_none_equivalent_to_empty():
    fixtures = {"q": [("a", 0.9)]}
    r1 = _search(FakeMemory(fixtures), "q", extra_queries=None)
    r2 = _search(FakeMemory(fixtures), "q", extra_queries=[])
    assert [r["id"] for r in r1["results"]] == [r["id"] for r in r2["results"]]
