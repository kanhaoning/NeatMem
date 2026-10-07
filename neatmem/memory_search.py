"""Search orchestration: dense retrieval + entity boosting + optional rerank.

Calls `memory.vector_store.search()` directly (memory is a
neatmem.memory_store.MemoryStore) so BM25 and entity signals stay under
NeatMem's own control instead of being fused opaquely inside the store.

Multi-query (query rewrite/expansion, plan §3.1): ``extra_queries`` carries
the rewriter output; every query runs dense retrieval at the same
internal_limit (concurrently), hits merge by max(score) with per-hit
provenance (``query_sources``; R6-d — the observation log needs to know
which query surfaced each memory). BM25 and entity boosting still run once
on the original query only.
"""
import os
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional

from neatmem.signals.bm25.scoring import build_bm25_score_map
from neatmem.signals.entity.base import Entity
from neatmem.signals.entity.boosting import apply_entity_boost, compute_entity_boosts
from neatmem.storage.entity.base import AbstractEntityStore

logger = logging.getLogger(__name__)


def _format_candidate(cand: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a vector-store result into the NeatMem API format."""
    payload = cand.get("payload", {}) or {}
    created_at = payload.get("created_at")
    if not created_at:
        created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    # mem0 flattens metadata into the payload top level, so the event-time
    # timestamp lives at payload["timestamp"], never inside a nested dict.
    # Surface it back into the returned metadata for clients (same-session
    # delay, eval answer prompts).
    metadata = dict(payload.get("metadata") or {})
    if "timestamp" not in metadata and payload.get("timestamp"):
        metadata["timestamp"] = payload["timestamp"]
    return {
        "id": str(cand.get("id", "")),
        "memory": payload.get("data", payload.get("memory", "")),
        "hash": payload.get("hash", ""),
        "metadata": metadata,
        "score": float(cand.get("score", 0.0)),
        "created_at": created_at,
        "updated_at": payload.get("updated_at", None),
        "user_id": payload.get("user_id", None),
        "agent_id": payload.get("agent_id", None),
        "app_id": payload.get("app_id", None),
        "run_id": payload.get("run_id", None),
    }


def _dense_search(memory, query: str, internal_limit: int, filters: Dict[str, Any]):
    """One dense retrieval round: embed + vector-store search."""
    embedding = memory.embedding_model.embed(query, "search")
    return memory.vector_store.search(
        query=query,
        vectors=embedding,
        top_k=internal_limit,
        filters=filters,
    )


# --- 淘汰排除（memory 反馈，计划 20261005 §5.6）---
# 两种供给方式：
# 1. ``exclude_ids`` 参数（生产形态）：调用方（main.py 搜索端点）从
#    ActivityStore.list_evicted_ids() 取当前淘汰集传入，逐请求新鲜。
# 2. EXCLUDE_MEMORY_IDS_FILE 环境变量（评测/实验形态）：每行一个 memory id。
# 参数优先；两者都未给时零副作用（连文件都不读）。
# 排除发生在候选池合并阶段（rerank 和 top_k 截断之前），并放大 over-fetch
# 保持候选池满额，即"检索层无视淘汰记忆、internal_limit 恒定且全部非淘汰"
# 的补位语义。
_exclude_ids_cache: Dict[str, set] = {}


def _excluded_memory_ids():
    path = os.environ.get("EXCLUDE_MEMORY_IDS_FILE") or None
    if not path:
        return None
    if path not in _exclude_ids_cache:
        with open(path, encoding="utf-8") as f:
            _exclude_ids_cache[path] = {line.strip() for line in f if line.strip()}
        logger.info(f"[exclude-ids] loaded {len(_exclude_ids_cache[path])} ids from {path}")
    return _exclude_ids_cache[path]


def search_memories(
    memory,
    query: str,
    filters: Dict[str, Any],
    top_k: int = 10,
    threshold: float = 0.1,
    entity_extractor=None,
    entity_store: Optional[AbstractEntityStore] = None,
    rerank_fn: Optional[Callable[[str, List[Dict[str, Any]], int], List[Dict[str, Any]]]] = None,
    use_entity: bool = True,
    use_bm25: bool = True,
    bm25_index=None,
    extra_queries: Optional[List[str]] = None,
    keyword_query: Optional[str] = None,
    exclude_ids: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Search memories via dense retrieval + self-managed entity boosting.

    Args:
        memory: mem0 Memory instance (used for embedding and vector_store).
        query: search query (dense primary; the rewriter's final_query when
            query rewrite is on).
        filters: mem0-compatible filters, e.g. {"user_id": "alice"}.
        top_k: final number of results.
        threshold: minimum semantic score.
        entity_extractor: AbstractEntityExtractor instance.
        entity_store: AbstractEntityStore instance.
        rerank_fn: optional rerank callable(query, candidates, top_k) -> reranked.
        use_entity: whether to apply entity boosting.
        extra_queries: optional rewriter expansions; each runs dense retrieval
            at the same internal_limit and merges into the candidate pool by
            max(score). BM25/entity boosting stay on the original query only.
        keyword_query: when the rewriter rephrased the query, the caller
            passes the user's original text here so BM25/entity signals stay
            anchored to what the user actually typed. Defaults to ``query``.
        exclude_ids: memory ids excluded at the candidate-pool stage (eviction
            filter; production form is ActivityStore.list_evicted_ids()).
            Falls back to EXCLUDE_MEMORY_IDS_FILE when not given.

    Returns:
        dict with {"results": [...], "total_candidates": int,
        "entity_boosted_count": int, "query_sources": {memory_id: query}}
    """
    # 1. Dense retrieval: over-fetch like mem0 does internally. Multi-query:
    #    dedup([query, *extra_queries]) each search once, concurrently.
    _env_limit = os.environ.get("INTERNAL_LIMIT")
    internal_limit = int(_env_limit) if _env_limit else max(top_k * 4, 60)

    excluded_ids = set(exclude_ids) if exclude_ids is not None else _excluded_memory_ids()
    if excluded_ids:
        # Over-fetch headroom so the merged pool stays at internal_limit after
        # exclusion filtering (evicted ids must not shrink the candidate pool).
        internal_limit *= 2

    queries = [query]
    if extra_queries:
        seen = {query.strip()}
        for eq in extra_queries:
            eq = (eq or "").strip()
            if eq and eq not in seen:
                seen.add(eq)
                queries.append(eq)

    def _run(q: str):
        return q, _dense_search(memory, q, internal_limit, filters)

    if len(queries) == 1:
        hit_sets = [_run(queries[0])]
    else:
        with ThreadPoolExecutor(max_workers=len(queries)) as pool:
            hit_sets = list(pool.map(_run, queries))

    # Merge by max(score), tracking which query produced each hit's best score.
    best: Dict[str, Dict[str, Any]] = {}
    query_sources: Dict[str, str] = {}
    for q, results in hit_sets:
        for r in results:
            rid = str(r.id)
            if excluded_ids and rid in excluded_ids:
                continue
            if rid not in best or r.score > best[rid]["score"]:
                best[rid] = {
                    "id": rid,
                    "score": r.score,
                    "payload": r.payload if hasattr(r, "payload") else {},
                }
                query_sources[rid] = q
    semantic_candidates = list(best.values())

    # 2. BM25 keyword search (original user query when the rewriter rephrased).
    keyword_query = keyword_query or query
    bm25_scores: Dict[str, float] = {}
    if use_bm25 and bm25_index is not None:
        bm25_hits = bm25_index.search(keyword_query, filters=filters, top_k=internal_limit)
        if excluded_ids:
            bm25_hits = [h for h in bm25_hits if str(h.memory_id) not in excluded_ids]
        if bm25_hits:
            from neatmem.utils.spacy.lemmatization import lemmatize_for_bm25
            lemmatized_query = lemmatize_for_bm25(keyword_query)
            bm25_scores = build_bm25_score_map(bm25_hits, keyword_query, lemmatized_query)

    # 3. Entity extraction and boosting (same anchor as BM25).
    entity_boosts: Dict[str, float] = {}
    if use_entity and entity_extractor and entity_store:
        query_entities: List[Entity] = entity_extractor.extract(keyword_query)
        if query_entities:
            entity_boosts = compute_entity_boosts(
                query_entities=query_entities,
                filters=filters,
                entity_store=entity_store,
                embed_fn=memory.embedding_model.embed,
            )

    # 3. Fuse and truncate.
    boosted = apply_entity_boost(
        semantic_candidates, entity_boosts, threshold,
        bm25_scores=bm25_scores,
    )
    results = boosted[:top_k]

    entity_boosted_count = sum(
        1 for c in results if c.get("entity_boost", 0.0) > 0
    )

    _diag_log_path = os.environ.get("ENTITY_DIAG_LOG")
    if _diag_log_path and entity_boosts:
        _write_search_diag_log(
            _diag_log_path,
            query=query,
            semantic_candidates=semantic_candidates,
            entity_boosts=entity_boosts,
            results=results,
        )

    # 4. Format to NeatMem output.
    formatted = [_format_candidate(c) for c in results]

    # 5. Optional rerank.
    if rerank_fn:
        formatted = rerank_fn(query, formatted, top_k=top_k)

    return {
        "results": formatted,
        "total_candidates": len(semantic_candidates),
        "entity_boosted_count": entity_boosted_count,
        "query_sources": query_sources,
    }


def _write_search_diag_log(
    path: str,
    query: str,
    semantic_candidates: List[Dict[str, Any]],
    entity_boosts: Dict[str, float],
    results: List[Dict[str, Any]],
) -> None:
    """Append search-side diag: boosted memory dense scores + topk dense scores."""
    import json
    import statistics

    boosted_ids = set(entity_boosts.keys())
    boosted_dense = [
        c.get("score", 0.0) for c in semantic_candidates
        if str(c.get("id")) in boosted_ids
    ]
    topk_dense = [c.get("semantic_score", c.get("score", 0.0)) for c in results]

    def _stats(xs):
        if not xs:
            return None
        xs_sorted = sorted(xs)
        return {
            "min": min(xs),
            "median": statistics.median(xs),
            "p90": xs_sorted[int(len(xs_sorted) * 0.9)],
            "max": max(xs),
        }

    record = {
        "phase": "search",
        "query": query[:200],
        "total_candidates": len(semantic_candidates),
        "total_boosted": len(boosted_ids),
        "boosted_memories_in_topk": sum(
            1 for c in results if c.get("entity_boost", 0.0) > 0
        ),
        "boosted_dense_scores": _stats(boosted_dense),
        "topk_dense_scores": _stats(topk_dense),
    }
    with open(path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
