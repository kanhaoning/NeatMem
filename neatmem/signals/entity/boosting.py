"""Entity boosting logic: replicate mem0's _compute_entity_boosts and score_and_rank."""
import os
from typing import Any, Callable, Dict, List, Optional

from neatmem.signals.entity.base import Entity
from neatmem.storage.entity.base import AbstractEntityStore


def deduplicate_entities(entities: List[Entity], max_entities: int = 8) -> List[Entity]:
    """De-duplicate entities by lower-cased text and cap the count.

    Mirrors mem0's behavior in _compute_entity_boosts.
    """
    seen: set = set()
    unique: List[Entity] = []
    for e in entities:
        key = e.text.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(e)
        if len(unique) >= max_entities:
            break
    return unique


def compute_entity_boosts(
    query_entities: List[Entity],
    filters: Dict[str, Any],
    entity_store: AbstractEntityStore,
    embed_fn: Callable[[str, str], List[float]],
    similarity_threshold: Optional[float] = None,
    entity_boost_weight: Optional[float] = None,
) -> Dict[str, float]:
    """Compute per-memory-id entity boost scores.

    Replicates mem0.utils.scoring._compute_entity_boosts.
    """
    if similarity_threshold is None:
        similarity_threshold = float(os.environ.get("ENTITY_SIMILARITY_THRESHOLD", "0.5"))
    if entity_boost_weight is None:
        entity_boost_weight = float(os.environ.get("ENTITY_BOOST_WEIGHT", "0.5"))
    deduped = deduplicate_entities(query_entities, max_entities=8)
    if not deduped:
        return {}

    search_filters = {
        k: v
        for k, v in filters.items()
        if k in ("user_id", "agent_id", "run_id") and v
    }

    memory_boosts: Dict[str, float] = {}
    for entity in deduped:
        embedding = embed_fn(entity.text, "search")
        matches = entity_store.find_matching_by_vector(
            query=entity.text,
            vectors=embedding,
            top_k=500,
            filters=search_filters,
        )
        for match in matches:
            if match.score is None or match.score < similarity_threshold:
                continue
            num_linked = max(len(match.linked_memory_ids), 1)
            memory_count_weight = 1.0 / (1.0 + 0.001 * ((num_linked - 1) ** 2))
            boost = match.score * entity_boost_weight * memory_count_weight
            for memory_id in match.linked_memory_ids:
                if memory_id:
                    mid = str(memory_id)
                    memory_boosts[mid] = max(memory_boosts.get(mid, 0.0), boost)
    _diag_log_path = os.environ.get("ENTITY_DIAG_LOG")
    if _diag_log_path:
        _write_entity_diag_log(
            _diag_log_path,
            query_entities=query_entities,
            deduped=deduped,
            entity_store=entity_store,
            embed_fn=embed_fn,
            search_filters=search_filters,
            similarity_threshold=similarity_threshold,
            entity_boost_weight=entity_boost_weight,
            memory_boosts=memory_boosts,
        )
    return memory_boosts


def _write_entity_diag_log(
    path: str,
    query_entities: List[Entity],
    deduped: List[Entity],
    entity_store: AbstractEntityStore,
    embed_fn: Callable[[str, str], List[float]],
    search_filters: Dict[str, Any],
    similarity_threshold: float,
    entity_boost_weight: float,
    memory_boosts: Dict[str, float],
) -> None:
    """Append one JSON line per query with entity boost diagnostics."""
    import json
    import statistics

    per_entity = []
    for entity in deduped:
        embedding = embed_fn(entity.text, "search")
        matches = entity_store.find_matching_by_vector(
            query=entity.text,
            vectors=embedding,
            top_k=500,
            filters=search_filters,
        )
        scores = [m.score for m in matches if m.score is not None]
        above = [s for s in scores if s >= similarity_threshold]
        per_entity.append({
            "entity_text": entity.text,
            "matched_records_total": len(scores),
            "matched_records_above_threshold": len(above),
            "cosine_min": min(scores) if scores else None,
            "cosine_median": statistics.median(scores) if scores else None,
            "cosine_p90": sorted(scores)[int(len(scores) * 0.9)] if scores else None,
            "cosine_max": max(scores) if scores else None,
        })

    boost_values = list(memory_boosts.values())
    record = {
        "phase": "compute",
        "query_entities": [e.text for e in query_entities],
        "num_query_entities": len(query_entities),
        "num_deduped": len(deduped),
        "similarity_threshold": similarity_threshold,
        "entity_boost_weight": entity_boost_weight,
        "per_entity": per_entity,
        "total_boosted_memories_unique": len(memory_boosts),
        "boost_values_min": min(boost_values) if boost_values else None,
        "boost_values_median": statistics.median(boost_values) if boost_values else None,
        "boost_values_max": max(boost_values) if boost_values else None,
    }
    with open(path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def apply_entity_boost(
    candidates: List[Dict[str, Any]],
    entity_boosts: Dict[str, float],
    threshold: float = 0.1,
    bm25_scores: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """Fuse semantic scores with BM25 and entity boosts.

    Replicates mem0.utils.scoring.score_and_rank (mem0-202606/mem0/utils/scoring.py:60-121).
    Formula: combined = (semantic + bm25 + entity) / max_possible
    max_possible: 1.0 (dense) + 1.0 (bm25 if active) + 0.5 (entity if active)
    """
    has_bm25 = bool(bm25_scores)
    has_entity = bool(entity_boosts)
    max_possible = 1.0
    if has_bm25:
        max_possible += 1.0
    if has_entity:
        max_possible += 0.5  # ENTITY_BOOST_WEIGHT, same as mem0

    scored: List[Dict[str, Any]] = []
    for cand in candidates:
        mem_id = str(cand.get("id"))
        semantic_score = cand.get("score", 0.0)
        if semantic_score < threshold:
            continue
        boost = entity_boosts.get(mem_id, 0.0) if has_entity else 0.0
        bm25 = bm25_scores.get(mem_id, 0.0) if has_bm25 else 0.0
        combined = min((semantic_score + bm25 + boost) / max_possible, 1.0)
        new_cand = dict(cand)
        new_cand["score"] = combined
        new_cand["semantic_score"] = semantic_score
        new_cand["entity_boost"] = boost
        new_cand["bm25_score"] = bm25
        scored.append(new_cand)
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored
