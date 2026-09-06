# NeatMem

Lightweight local memory for agents — every dedup, update, and rerank decision inspectable and tunable.

3 dedup detectors × 4 update resolvers × 3 rerank modes · 60+ parameters · 6 prompts replaceable

!!! note "Status"
    Actively developed (v0.5.x). NeatMem is usable for local development and mem0-compatible client integrations, but APIs, packaging, and integrations may still change.

!!! success "Benchmark"
    90.8% accuracy on LoCoMo (MiniMax-M3 answer + judge, SiliconFlow bge-m3 embedding). See the [evaluation guide](evaluation.md) for benchmark reproduction steps.

## Why NeatMem?

Agent memory is easy to start but hard to keep clean.

Common problems include:

- duplicate memories accumulating over time
- assistant suggestions being stored as user facts
- semantically related memories not being merged
- irrelevant memories being recalled because of weak vector matches
- local agent tools needing a simple self-hosted memory backend

NeatMem keeps every memory decision inspectable and tunable:

- Every extraction, dedup, merge, and rerank decision runs through a prompt
  you can read and replace — 6 prompt slots, as plain text files.
- Every threshold and behavior switch is an explicit parameter — dedup
  strictness, merge strategy, recall depth, rerank mode — not a hidden
  model judgment.
- Every write logs what was added, merged, or skipped, so memory drift
  can be audited instead of discovered by accident.

## Features

- **Multi-target dedup & merge**
  - More thorough updates at no extra call cost: when one new fact affects several existing memories, all of them get updated in one pass — not just the closest match — leaving no stale or contradictory memory behind.
  - Detection mode, update behavior, and dedup itself are all switchable — see the [configuration reference](configuration.md).

- **Less memory pollution**
  - Avoids saving AI suggestions, guesses, or tool noise as user facts.
  - Tracks whether each memory came from the user, assistant, or tool output.

- **More relevant recall**
  - Multi-signal retrieval: dense vector search + BM25 keyword matching, with optional entity boosting.
  - Rerank filters and reorders candidates before injection into agent context — LLM (listwise/pointwise) or cross-encoder (hosted API or local model).

- **Lightweight local storage**
  - Runs with local Qdrant (embedded or server mode) by default.
  - Does not require Redis, a hosted memory service, or a full database stack.

- **Optional graph memory**
  - Entity-relation storage via KuzuDB. Off by default.

- **Agent integrations**
  - Works with OpenClaw and Hermes.
  - Python client API shaped like mem0's — point your existing mem0 client at the local server to migrate.

## Compatibility

NeatMem implements a mem0-compatible API subset for local agent memory workflows:

- add memory
- search memory
- list memories
- update memory
- delete memory
- health check

It is designed to work with OpenClaw's and Hermes' memory plugin flows and other mem0-style integrations.

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
- Prompt behavior may vary across models.
- BM25 lemmatization is basic; bilingual (Chinese/English) tokenization needs improvement.

## Roadmap

- Bilingual multi-signal support (improved Chinese/English BM25 and entity extraction)
- Memory inspection and export/import tools
- Richer recall diagnostics

## License

MIT License.

## Acknowledgements

Inspired by the mem0 project (Apache-2.0). Vendored-code notices are in the
respective file headers.
