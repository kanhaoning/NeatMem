# Third-Party Licenses

## mem0

- Source: https://github.com/mem0ai/mem0
- Version: v2.0.0
- License: Apache License 2.0
- Copyright: (c) Mem0

Vendored files (verbatim copy with import path rewriting only, no algorithmic changes):
- `neatmem/utils/spacy/entity_extraction.py`
- `neatmem/utils/spacy/lemmatization.py`
- `neatmem/utils/spacy/spacy_models.py`
- `neatmem/utils/text_parsing.py` (`extract_json` + `remove_code_blocks`)
- `neatmem/storage/vector/qdrant.py` (ported from mem0 vector_stores/qdrant.py, telemetry stripped)
- `neatmem/storage/history.py` (ported from mem0 memory/storage.py SQLiteHistoryManager)
- `neatmem/embeddings.py` (ported from mem0 embeddings wrapper)
- `neatmem/memory_store.py` (ported from mem0 memory/main.py infer=False CRUD branch)

The full Apache License 2.0 text is available at:
http://www.apache.org/licenses/LICENSE-2.0
