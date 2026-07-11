"""Search orchestration: dense retrieval + entity boosting + optional rerank.

Bypasses `memory.search()` because mem0's search always fuses BM25 and entity
signals internally. This module calls `memory.vector_store.search()` directly.
"""
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from neatmem.signals.bm25.scoring import build_bm25_score_map
from neatmem.signals.entity.base import Entity
from neatmem.signals.entity.boosting import apply_entity_boost, compute_entity_boosts
from neatmem.storage.entity.base import AbstractEntityStore


def _format_candidate(cand: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a vector-store result into the NeatMem API format."""
    payload = cand.get("payload", {}) or {}
    created_at = payload.get("created_at")
    if not created_at:
        created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return {
        "id": str(cand.get("id", "")),
        "memory": payload.get("data", payload.get("memory", "")),
        "hash": payload.get("hash", ""),
        "metadata": payload.get("metadata", {}),
        "score": float(cand.get("score", 0.0)),
        "created_at": created_at,
        "updated_at": payload.get("updated_at", None),
        "user_id": payload.get("user_id", None),
        "agent_id": payload.get("agent_id", None),
        "app_id": payload.get("app_id", None),
        "run_id": payload.get("run_id", None),
    }


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
) -> Dict[str, Any]:
    """Search memories via dense retrieval + self-managed entity boosting.

    Args:
        memory: mem0 Memory instance (used for embedding and vector_store).
        query: search query.
        filters: mem0-compatible filters, e.g. {"user_id": "alice"}.
        top_k: final number of results.
        threshold: minimum semantic score.
        entity_extractor: AbstractEntityExtractor instance.
        entity_store: AbstractEntityStore instance.
        rerank_fn: optional rerank callable(query, candidates, top_k) -> reranked.
        use_entity: whether to apply entity boosting.

    Returns:
        dict with {"results": [...], "total_candidates": int, "entity_boosted_count": int}
    """
    # 1. Dense retrieval: over-fetch like mem0 does internally.
    internal_limit = max(top_k * 4, 60)
    query_embedding = memory.embedding_model.embed(query, "search")
    raw_results = memory.vector_store.search(
        query=query,
        vectors=query_embedding,
        top_k=internal_limit,
        filters=filters,
    )

    semantic_candidates = []
    for r in raw_results:
        semantic_candidates.append({
            "id": str(r.id),
            "score": r.score,
            "payload": r.payload if hasattr(r, "payload") else {},
        })

    # 2. BM25 keyword search.
    bm25_scores: Dict[str, float] = {}
    if use_bm25 and bm25_index is not None:
        bm25_hits = bm25_index.search(query, filters=filters, top_k=internal_limit)
        if bm25_hits:
            from mem0.utils.lemmatization import lemmatize_for_bm25
            lemmatized_query = lemmatize_for_bm25(query)
            bm25_scores = build_bm25_score_map(bm25_hits, query, lemmatized_query)

    # 3. Entity extraction and boosting.
    entity_boosts: Dict[str, float] = {}
    if use_entity and entity_extractor and entity_store:
        query_entities: List[Entity] = entity_extractor.extract(query)
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

    # 4. Format to NeatMem output.
    formatted = [_format_candidate(c) for c in results]

    # 5. Optional rerank.
    if rerank_fn:
        formatted = rerank_fn(query, formatted, top_k=top_k)

    return {
        "results": formatted,
        "total_candidates": len(semantic_candidates),
        "entity_boosted_count": entity_boosted_count,
    }
