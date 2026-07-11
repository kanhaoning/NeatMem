"""No-op BM25 index for when BM25 is disabled."""

from typing import Dict, List, Optional

from neatmem.signals.bm25.base import AbstractBM25Index, BM25SearchResult


class NoopBM25Index(AbstractBM25Index):
    """BM25 index that does nothing. Used when ENABLE_BM25=false."""

    def index_memory(
        self,
        memory_id: str,
        text: str,
        filters: Optional[Dict] = None,
    ) -> None:
        return

    def search(
        self,
        query: str,
        filters: Optional[Dict] = None,
        top_k: int = 60,
    ) -> List[BM25SearchResult]:
        return []

    def delete(self, memory_id: str) -> None:
        return
