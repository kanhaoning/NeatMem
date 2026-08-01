"""Minimal vector store interface for NeatMem storage backends.

Kept intentionally small: it covers exactly the operations NeatMem performs
(insert/search/delete/update/get/list/reset), exposes the raw backend client
for the self-managed signals (BM25/entity stores share the Qdrant client),
and requires a `build_filter` translator so signals can reuse the store's
filter syntax.
"""

from abc import ABC, abstractmethod


class VectorStoreBase(ABC):
    # Raw backend client (e.g. qdrant_client.QdrantClient). Signals that need
    # direct access (BM25 sparse writes, entity store) use this attribute.
    client = None

    @abstractmethod
    def create_col(self, vector_size, on_disk, distance=None):
        """Create the underlying collection/index if it does not exist."""

    @abstractmethod
    def insert(self, vectors, payloads=None, ids=None):
        """Insert vectors into a collection."""

    @abstractmethod
    def search(self, query, vectors, top_k=5, filters=None):
        """Search for similar vectors."""

    @abstractmethod
    def delete(self, vector_id):
        """Delete a vector by ID."""

    @abstractmethod
    def update(self, vector_id, vector=None, payload=None):
        """Update a vector and its payload."""

    @abstractmethod
    def get(self, vector_id):
        """Retrieve a vector by ID."""

    @abstractmethod
    def list_cols(self):
        """List all collections."""

    @abstractmethod
    def delete_col(self):
        """Delete a collection."""

    @abstractmethod
    def col_info(self):
        """Get information about a collection."""

    @abstractmethod
    def list(self, filters=None, top_k=100):
        """List vectors in a collection."""

    @abstractmethod
    def reset(self):
        """Reset by deleting the collection and recreating it."""

    @abstractmethod
    def build_filter(self, filters: dict):
        """Translate a mem0-compatible filter dict into the backend's native
        filter object. Returns None for empty filters."""
