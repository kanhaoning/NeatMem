"""Entity extraction abstractions."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List


@dataclass
class Entity:
    type: str
    text: str


@dataclass
class EntityMatch:
    score: float
    linked_memory_ids: List[str]


class AbstractEntityExtractor(ABC):
    @abstractmethod
    def extract(self, text: str) -> List[Entity]:
        """Extract entities from text."""
        ...
