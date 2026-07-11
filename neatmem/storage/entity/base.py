"""Entity storage abstractions."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class EntityRecord:
    entity_id: str
    entity_text: str
    entity_type: str
    scope: str
    linked_memory_ids: List[str]
    score: Optional[float] = None


class AbstractEntityStore(ABC):
    @abstractmethod
    def link_entities(
        self, entities: List[Any], memory_id: str, scope: str
    ) -> None:
        """Associate extracted entities with a memory id under a scope."""
        ...

    @abstractmethod
    def find_matching_by_vector(
        self,
        query: str,
        vectors: List[float],
        top_k: int,
        filters: Dict[str, Any],
    ) -> List[EntityRecord]:
        """Find stored entities similar to the query vector and filters."""
        ...

    @abstractmethod
    def delete_by_memory_id(self, memory_id: str, scope: str) -> None:
        """Remove a memory id from all entity links in the scope."""
        ...
