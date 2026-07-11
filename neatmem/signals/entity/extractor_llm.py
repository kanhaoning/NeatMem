"""LLM-based entity extractor (placeholder for stage two)."""
from typing import List

from neatmem.signals.entity.base import AbstractEntityExtractor, Entity


class LLMEntityExtractor(AbstractEntityExtractor):
    """Stub implementation. Not wired in parity validation."""

    def extract(self, text: str) -> List[Entity]:
        return []
