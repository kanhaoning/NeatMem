"""BM25 index backed by Qdrant sparse vector slot."""

import logging
from typing import Dict, List, Optional

from qdrant_client.models import PointVectors, SparseVector

from neatmem.utils.spacy.lemmatization import lemmatize_for_bm25
from neatmem.signals.bm25.base import AbstractBM25Index, BM25SearchResult

logger = logging.getLogger(__name__)


class QdrantSparseBM25Index(AbstractBM25Index):
    """BM25 index using Qdrant's sparse vector slot named 'bm25'.

    This backend reuses the same sparse vector slot that mem0 creates,
    but NeatMem controls when and how the sparse vector is written and queried.
    """

    def __init__(self, vector_store, collection_name: str = "mem0"):
        self.vector_store = vector_store
        self.client = vector_store.client
        self.collection_name = collection_name
        self._encoder = None
        self._has_bm25_slot: Optional[bool] = None

    def _get_encoder(self):
        """Lazy-load fastembed BM25 sparse encoder."""
        if self._encoder is None:
            try:
                from fastembed import SparseTextEmbedding
                self._encoder = SparseTextEmbedding(model_name="Qdrant/bm25")
                logger.info("BM25 encoder loaded (fastembed Qdrant/bm25)")
            except Exception as e:
                logger.error(f"Failed to load BM25 encoder: {e}")
                raise
        return self._encoder

    def _encode(self, text: str) -> SparseVector:
        """Encode text into a BM25 sparse vector."""
        lemmatized = lemmatize_for_bm25(text)
        results = list(self._get_encoder().embed([lemmatized]))
        sparse = results[0]
        return SparseVector(
            indices=sparse.indices.tolist(),
            values=sparse.values.tolist(),
        )

    def _check_slot(self) -> bool:
        """Check whether the collection has a 'bm25' sparse vector slot."""
        if self._has_bm25_slot is None:
            try:
                info = self.client.get_collection(self.collection_name)
                sparse_cfg = info.config.params.sparse_vectors
                self._has_bm25_slot = bool(sparse_cfg and "bm25" in sparse_cfg)
            except Exception as e:
                logger.warning(f"Failed to check bm25 slot: {e}")
                self._has_bm25_slot = False
        return self._has_bm25_slot

    def index_memory(
        self,
        memory_id: str,
        text: str,
        filters: Optional[Dict] = None,
    ) -> None:
        """Write or refresh the BM25 sparse vector for a memory."""
        if not self._check_slot():
            logger.warning(
                f"Collection {self.collection_name} has no 'bm25' sparse slot; "
                "skipping BM25 index update"
            )
            return

        sparse = self._encode(text)
        self.client.update_vectors(
            collection_name=self.collection_name,
            points=[PointVectors(id=memory_id, vector={"bm25": sparse})],
        )

    def search(
        self,
        query: str,
        filters: Optional[Dict] = None,
        top_k: int = 60,
    ) -> List[BM25SearchResult]:
        """Search memories by BM25 sparse vector."""
        if not self._check_slot():
            return []

        sparse = self._encode(query)
        query_filter = self.vector_store._create_filter(filters) if filters else None

        try:
            hits = self.client.query_points(
                collection_name=self.collection_name,
                query=sparse,
                using="bm25",
                query_filter=query_filter,
                limit=top_k,
            )
            return [
                BM25SearchResult(memory_id=str(h.id), score=h.score)
                for h in hits.points
            ]
        except Exception as e:
            logger.warning(f"BM25 search failed: {e}")
            return []

    def delete(self, memory_id: str) -> None:
        """Sparse vector deletion is handled automatically when the point is deleted."""
        return
