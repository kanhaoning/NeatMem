"""Entity store factory."""
from typing import Any, Optional

from neatmem.storage.entity.base import AbstractEntityStore
from neatmem.storage.entity.qdrant import QdrantEntityStore


def create_entity_store(
    backend: str = "qdrant",
    *,
    qdrant_client: Optional[Any] = None,
    collection_name: str = "mem0_entities",
    vector_size: int = 1024,
    **kwargs: Any,
) -> AbstractEntityStore:
    if backend == "qdrant":
        return QdrantEntityStore(
            qdrant_client=qdrant_client,
            collection_name=collection_name,
            vector_size=vector_size,
        )
    raise ValueError(f"Unsupported entity store backend: {backend}")
