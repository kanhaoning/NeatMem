"""BM25 index factory."""

import os
from typing import Optional

from neatmem.signals.bm25.base import AbstractBM25Index


def create_bm25_index(
    backend: str,
    vector_store=None,
    collection_name: str = "mem0",
) -> AbstractBM25Index:
    """Create a BM25 index backend.

    Args:
        backend: "qdrant_sparse" or "none".
        vector_store: mem0 Qdrant vector store instance (required for qdrant_sparse).
        collection_name: Qdrant collection name.

    Returns:
        AbstractBM25Index instance.
    """
    if backend == "none":
        from neatmem.signals.bm25.noop import NoopBM25Index
        return NoopBM25Index()

    if backend == "qdrant_sparse":
        if vector_store is None:
            raise ValueError("vector_store is required for qdrant_sparse backend")
        from neatmem.signals.bm25.qdrant_sparse import QdrantSparseBM25Index
        return QdrantSparseBM25Index(
            vector_store=vector_store,
            collection_name=collection_name,
        )

    raise ValueError(f"Unknown BM25 backend: {backend}")


def create_bm25_index_from_env(vector_store=None, collection_name: str = "mem0") -> AbstractBM25Index:
    """Create BM25 index from environment variables."""
    backend = os.environ.get("BM25_BACKEND", "qdrant_sparse")
    return create_bm25_index(backend, vector_store, collection_name)
