"""spaCy-based NLP utilities vendored from mem0 v2.0.0 (Apache-2.0).

See per-file headers for provenance and modification notes.
Do not change algorithm logic here — these files must stay byte-comparable
with the mem0 originals (modulo imports) to preserve LOCOMO parity.
"""
from neatmem.utils.spacy.entity_extraction import extract_entities
from neatmem.utils.spacy.lemmatization import lemmatize_for_bm25

__all__ = ["extract_entities", "lemmatize_for_bm25"]
