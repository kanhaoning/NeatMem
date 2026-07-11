"""BM25 index abstract interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class BM25SearchResult:
    memory_id: str
    score: float  # normalized BM25 score in [0, 1]


class AbstractBM25Index(ABC):
    """Abstract interface for BM25 keyword retrieval."""

    @abstractmethod
    def index_memory(
        self,
        memory_id: str,
        text: str,
        filters: Optional[Dict] = None,
    ) -> None:
        """Index or re-index a memory's text for BM25 retrieval."""
        ...

    @abstractmethod
    def search(
        self,
        query: str,
        filters: Optional[Dict] = None,
        top_k: int = 60,
    ) -> List[BM25SearchResult]:
        """Search memories by BM25 and return normalized scores."""
        ...

    @abstractmethod
    def delete(self, memory_id: str) -> None:
        """Remove a memory from the BM25 index if applicable."""
        ...
