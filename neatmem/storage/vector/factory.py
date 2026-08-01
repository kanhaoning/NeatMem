"""Vector store factory.

Only Qdrant is supported. The factory exists so that adding another backend
later is a matter of registering a new class here, and so wiring code follows
the same factory pattern as neatmem.storage.entity / neatmem.storage.message.
"""

from typing import Optional

from neatmem.storage.vector.base import VectorStoreBase
from neatmem.storage.vector.qdrant import Qdrant


def create_vector_store(
    backend: str = "qdrant",
    *,
    collection_name: str,
    embedding_model_dims: int,
    client=None,
    host: Optional[str] = None,
    port: Optional[int] = None,
    path: Optional[str] = None,
    url: Optional[str] = None,
    api_key: Optional[str] = None,
    on_disk: bool = False,
) -> VectorStoreBase:
    if backend == "qdrant":
        return Qdrant(
            collection_name=collection_name,
            embedding_model_dims=embedding_model_dims,
            client=client,
            host=host,
            port=port,
            path=path,
            url=url,
            api_key=api_key,
            on_disk=on_disk,
        )
    raise ValueError(f"Unsupported vector store backend: {backend!r} (only 'qdrant' is supported)")
