"""BM25 scoring utilities."""

import math
from typing import Dict, List

from neatmem.signals.bm25.base import BM25SearchResult


def get_bm25_params(query: str, *, lemmatized: str = "") -> tuple:
    """Return (midpoint, steepness) for sigmoid normalization based on query length.

    Mirrors mem0.utils.scoring.get_bm25_params.
    """
    num_terms = len(lemmatized.split()) if lemmatized else 1
    if num_terms <= 3:
        return 5.0, 0.7
    elif num_terms <= 6:
        return 7.0, 0.6
    elif num_terms <= 9:
        return 9.0, 0.5
    elif num_terms <= 15:
        return 10.0, 0.5
    else:
        return 12.0, 0.5


def normalize_bm25(raw_score: float, midpoint: float, steepness: float) -> float:
    """Normalize raw BM25 score to [0, 1] using logistic sigmoid."""
    return 1.0 / (1.0 + math.exp(-steepness * (raw_score - midpoint)))


def build_bm25_score_map(
    bm25_results: List[BM25SearchResult],
    query: str,
    lemmatized_query: str,
) -> Dict[str, float]:
    """Convert raw BM25 search results to normalized memory_id -> score map."""
    midpoint, steepness = get_bm25_params(query, lemmatized=lemmatized_query)
    scores = {}
    for r in bm25_results:
        if r.score and r.score > 0:
            scores[r.memory_id] = normalize_bm25(r.score, midpoint, steepness)
    return scores
