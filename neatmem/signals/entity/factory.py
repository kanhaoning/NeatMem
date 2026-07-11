"""Entity extractor factory."""
from typing import Any, Optional

from neatmem.signals.entity.base import AbstractEntityExtractor
from neatmem.signals.entity.extractor_ner import NEREntityExtractor


def create_entity_extractor(
    backend: str = "ner",
    **kwargs: Any,
) -> AbstractEntityExtractor:
    if backend == "ner":
        return NEREntityExtractor()
    raise ValueError(f"Unsupported entity extractor backend: {backend}")
