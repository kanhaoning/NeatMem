# NeatMem

Lightweight local memory for agents, with cleaner deduplication, less memory pollution, and more relevant recall.

NeatMem is built for developers who want practical long-term memory. It focuses on keeping local agent memory clean: merging repeated facts, preventing AI suggestions, guesses, and tool noise from being saved as user facts, saving memories with enough context, and filtering irrelevant recalls.

!!! note "Status"
    v0.1-preview. NeatMem is usable for local development and mem0-compatible integrations, but APIs, packaging, and integrations may still change.

!!! success "Benchmark"
    90.80% accuracy on LOCOMO, fully reproducible locally (3-run mean; MiniMax-M3 answer + judge, SiliconFlow bge-m3 embedding). See the [evaluation guide](evaluation.md) for benchmark reproduction steps.

## Why NeatMem?

Agent memory is easy to start but hard to keep clean.

Common problems include:

- duplicate memories accumulating over time
- assistant suggestions being stored as user facts
- vague memories losing their original context
- semantically related memories not being merged
- irrelevant memories being recalled because of weak vector matches
- local agent tools needing a simple self-hosted memory backend

NeatMem focuses on one narrow goal:

> Local agent memory that stays clean, inspectable, and easy to tune.

## Features

- **LLM-assisted memory decisions**
  - Classifies each new memory as `add`, `none`, or `update` (listwise, single LLM call).
  - `DEDUP_RESOLVER` controls what happens on `update`: `skip` (keep both), `replace` (overwrite), `rewrite` (LLM merge), `edit` (LLM patch). Set `DEDUP_ENABLED=false` to turn dedup off.

- **Sequential memory updates**
  - Processes new memories one by one so each merge sees the latest stored version.
  - Helps avoid overwrite conflicts when several new facts update the same old memory.

- **Less memory pollution**
  - Avoids saving AI suggestions, guesses, or tool noise as user facts.
  - Tracks whether each memory came from the user, assistant, or tool output.

- **Memories with enough context**
  - Adds missing context from the same message batch when needed.
  - Example: “during development” can become “while developing a mem0-based memory module”.

- **More relevant recall**
  - Multi-signal retrieval: dense vector search + BM25 sparse matching + entity boosting.
  - Rerank filters and reorders candidates before injection into agent context — LLM (listwise/pointwise) or cross-encoder (hosted API or local model).

- **Lightweight local storage**
  - Runs with local Qdrant (embedded or server mode) by default.
  - Does not require Redis, a hosted memory service, or a full database stack.

- **Modular signal architecture**
  - Message store, BM25, and entity modules are decoupled under `neatmem/storage/` and `neatmem/signals/`.
  - Each signal can be toggled via environment variables (`ENABLE_BM25`, `ENABLE_ENTITY`, `ENABLE_GRAPH`).

- **Optional graph memory (opt-in)**
  - Entity-relation storage via KuzuDB, toggled by `ENABLE_GRAPH`.
  - Off by default.

- **OpenClaw and mem0-style integration**
  - Implements the core mem0-style memory endpoints needed for local agent workflows.
  - Designed to support OpenClaw platform-mode memory integration.

## Compatibility

NeatMem implements a mem0-compatible API subset for local agent memory workflows:

- add memory
- search memory
- list memories
- update memory
- delete memory
- health check

It is designed to work with OpenClaw's and Hermes' memory plugin flows and other mem0-style integrations. v0.1 does not aim to cover every mem0 SDK feature or mem0 hosted-platform behavior.

A remote client is provided for programmatic access:

```python
from neatmem import MemoryClient

client = MemoryClient(host="http://localhost:8790")  # requires `neatmem serve`

added = client.add("My name is Alex", user_id="default_user")
# {"results": [{"id": "...", "memory": "User's name is Alex", "event": "ADD"}]}

found = client.search("What is my name?", filters={"user_id": "default_user"})
print(found["results"][0]["memory"])  # -> "User's name is Alex"
```

Full method and parameter reference: [Python Client](client.md).

The client also provides server-side write batching (`add_messages`, `get_next_batch`, `mark_batch_processed`, `flush_messages`) and raw message history access (`client.messages` — `query`, `sessions`, `delete`, `reset`).

## Limitations

NeatMem is in active development. Current limitations:

- APIs and packaging may still change.
- No dashboard or GUI.
- No multi-tenant permission system.
- OpenClaw is the primary tested integration path.
- Prompt behavior is still being iterated and may vary across models.
- BM25 lemmatization is basic; bilingual (Chinese/English) tokenization needs improvement.

## Roadmap

- Bilingual multi-signal support (improved Chinese/English BM25 and entity extraction)
- Memory inspection and export/import tools
- Richer recall diagnostics

## License

MIT License.

## Acknowledgements

NeatMem is inspired by the mem0 project and mem0-style memory API patterns, and is designed to interoperate with OpenClaw memory plugin flows. Upstream license notices should be preserved where applicable.

Some utility functions in `neatmem/utils/spacy/` (`spacy_models.py`, `entity_extraction.py`, `lemmatization.py`) are vendored from mem0 v2.0.0 (Apache-2.0); see file headers for modification notes.
