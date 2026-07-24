# This file is vendored from mem0 (https://github.com/mem0ai/mem0), v2.0.0.
# Copyright (c) Mem0
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Modifications: import paths rewritten from mem0.utils.* to relative imports;
# no algorithmic changes.

"""
BM25 lemmatization for consistent keyword matching.

Uses spaCy's lemmatizer for better handling of:
- Verb forms: attending/attends/attended -> attend
- Comparatives/superlatives: older/oldest -> old
- Plurals: memories -> memory
- Avoids over-stemming: organization != organize

Also includes original -ing forms alongside lemmas to handle cases
where spaCy's context-dependent lemmatization produces inconsistent
results (e.g., "meeting" as noun vs verb -> different lemmas).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def lemmatize_for_bm25(text: str) -> str:
    """Lemmatize text for BM25 matching.

    Returns space-joined lemmas for full-text search. Falls back to
    the original text if spaCy is unavailable.
    """
    from .spacy_models import get_nlp_lemma

    nlp = get_nlp_lemma()
    if nlp is None:
        return text

    doc = nlp(text.lower())
    tokens = []

    for token in doc:
        if token.is_punct or token.is_stop:
            continue

        lemma = token.lemma_
        if lemma.isalnum():
            tokens.append(lemma)

        # Also add original if it ends in -ing and differs from lemma.
        # This handles noun/verb ambiguity (meeting/meet, attending/attend).
        if token.text.endswith("ing") and token.text != lemma and token.text.isalnum():
            tokens.append(token.text)

    return " ".join(tokens)
