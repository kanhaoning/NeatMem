"""NER-based entity extractor: wraps mem0's extract_entities to guarantee parity."""
from typing import List

from mem0.utils.entity_extraction import extract_entities as mem0_extract_entities

from neatmem.signals.entity.base import AbstractEntityExtractor, Entity


class NEREntityExtractor(AbstractEntityExtractor):
    def extract(self, text: str) -> List[Entity]:
        if not text or not text.strip():
            return []
        raw = mem0_extract_entities(text)
        return [Entity(type=entity_type, text=entity_text) for entity_type, entity_text in raw]
