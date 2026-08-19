# NeatMem

[![PyPI](https://img.shields.io/pypi/v/neatmem)](https://pypi.org/project/neatmem/)
[![Documentation](https://readthedocs.org/projects/neatmem/badge/?version=latest)](https://neatmem.readthedocs.io/en/latest/)

Lightweight local memory for agents, with cleaner deduplication, less memory pollution, and more relevant recall.

NeatMem is built for developers who want practical long-term memory. It focuses on keeping local agent memory clean: merging repeated facts, preventing AI suggestions, guesses, and tool noise from being saved as user facts, saving memories with enough context, and filtering irrelevant recalls.

> **Docs**: [neatmem.readthedocs.io](https://neatmem.readthedocs.io/en/latest/) — full quick start, configuration reference, custom prompts, API reference, and integration guides.

> Status: v0.1-preview. NeatMem is usable for local development and mem0-compatible integrations, but APIs, packaging, and integrations may still change.

> **Benchmark**: 90.80% accuracy on LOCOMO, fully reproducible locally (3-run mean; MiniMax-M3 answer + judge, SiliconFlow bge-m3 embedding). See the [evaluation guide](https://github.com/kanhaoning/NeatMem/blob/main/neatmem/evaluation/README.md) for benchmark reproduction steps.

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
  - LLM listwise rerank filters and reorders candidates before injection into agent context.

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

## How it works

### Add flow

```text
messages
  ↓
retrieve last-k messages as extraction context
  ↓
LLM memory extraction (with last-k context)
  ↓
context completion and source tracking
  ↓
sequential LLM-assisted memory decisions
  ├─ add    -> store as new memory
  ├─ none   -> skip (duplicate)
  └─ update -> merge per DEDUP_RESOLVER (skip/replace/rewrite/edit)
  ↓
write to vector store + BM25 index + entity store
```

### Search flow

```text
query
  ↓
dense vector search + BM25 sparse search + entity boosting
  ↓
LLM listwise rerank
  ↓
threshold filtering
  ↓
results
```

## Compatibility

NeatMem implements a mem0-compatible API subset for local agent memory workflows:

- add memory
- search memory
- list memories
- update memory
- delete memory
- health check

It is designed to work with OpenClaw's and Hermes' memory plugin flows and other mem0-style integrations. v0.1 does not aim to cover every mem0 SDK feature or mem0 hosted-platform behavior.

Runs on 10 LLM providers and 4 embedding providers (MiniMax, DeepSeek, Qwen, GLM, Kimi, Doubao, SiliconFlow, OpenAI, Gemini, OpenRouter) — endpoints and thinking-control parameters in [supported providers](https://neatmem.readthedocs.io/en/latest/providers/).

A remote client is provided for programmatic access:

```python
from neatmem import MemoryClient

client = MemoryClient(host="http://localhost:8790")  # requires `neatmem serve`

added = client.add("My name is Alex", user_id="default_user")
# {"results": [{"id": "...", "memory": "User's name is Alex", "event": "ADD"}]}

found = client.search("What is my name?", filters={"user_id": "default_user"})
print(found["results"][0]["memory"])  # -> "User's name is Alex"
```

The client also exposes NeatMem-specific extensions for server-side write batching (not part of the mem0 API): `add_messages`, `get_next_batch`, `mark_batch_processed`, `flush_messages`.

## Quick start

```bash
pip install neatmem

# Minimal .env (OpenAI-compatible LLM + SiliconFlow embedding)
curl -o .env https://raw.githubusercontent.com/kanhaoning/NeatMem/main/.env.example

neatmem serve   # listens on http://localhost:8790
```

For better BM25 keyword matching (searching "memory" also matches "memories"): `pip install "neatmem[nlp]" && python -m spacy download en_core_web_sm`. For source install and more, see the [full quick start](https://neatmem.readthedocs.io/en/latest/quickstart/).

## Configuration

NeatMem reads configuration from environment variables (a `.env` file in the working directory). Common settings — full table in the [configuration reference](https://neatmem.readthedocs.io/en/latest/configuration/):

| Variable | Required | Default | Description |
|---|---:|---|---|
| `LLM_PROVIDER` | no | - | LLM provider preset (`minimax`, `deepseek`, `dashscope`, …) — supplies the default base URL |
| `LLM_API_KEY` | yes | - | LLM API key (`OPENAI_API_KEY` accepted as fallback) |
| `LLM_MODEL` | yes | - | LLM model name (no default; server refuses to boot without it) |
| `EMBEDDING_PROVIDER` | no | `siliconflow` | `siliconflow`, `openai`, `dashscope`, or `xinference` |
| `EMBEDDING_API_KEY` | conditional | - | Required for hosted embedding providers |
| `EMBEDDING_MODEL` | no | `BAAI/bge-m3` | Embedding model name |
| `NEATMEM_PORT` | no | `8790` | Server port |
| `DEDUP_ENABLED` | no | `true` | Enable dedup on write |
| `DEDUP_RESOLVER` | no | `skip` | Duplicate resolution: `skip`, `replace`, `rewrite`, `edit` |

## Custom prompts

Every core prompt (extraction, dedup, merge rewrite, patch edit, rerank) can be replaced with a built-in variant id or your own prompt file — see the [custom prompts guide](https://neatmem.readthedocs.io/en/latest/custom-prompts/).

## OpenClaw integration

With the NeatMem server running at `http://localhost:8790`:

```bash
openclaw plugins install @neatmem/openclaw-neatmem
openclaw neatmem init
```

Then restart the gateway (`openclaw gateway restart`) to load the plugin.

`init` works with zero flags: it writes `apiKey=neatmem-local`, `baseUrl=http://localhost:8790`, and your OS username as `userId`, then validates against the server. Override with `--api-key`, `--user-id`, or `--base-url`.

Example OpenClaw configuration:

```json
{
  "plugins": {
    "slots": {
      "memory": "openclaw-neatmem"
    },
    "entries": {
      "openclaw-neatmem": {
        "enabled": true,
        "config": {
          "apiKey": "neatmem-local",
          "userId": "default_user",
          "baseUrl": "http://localhost:8790"
        }
      }
    }
  }
}
```

Then check:

```bash
openclaw neatmem status
```

The plugin id is `openclaw-neatmem`. It talks to NeatMem through the local mem0-compatible HTTP API. For full CLI/tool reference and building from source, see [openclaw/README.md](https://github.com/kanhaoning/NeatMem/blob/main/openclaw/README.md).

## Hermes integration

NeatMem includes a Hermes Agent memory provider under `hermes/`. With the NeatMem server running at `http://localhost:8790`:

```bash
hermes plugins install kanhaoning/NeatMem/hermes --enable
hermes config set memory.provider neatmem
```

The plugin registers four memory tools (`neatmem_search`, `neatmem_list`, `neatmem_update`, `neatmem_delete`) and recalls memories automatically on each turn. Each turn is forwarded to the server, which extracts memories in fixed-size batches; anything still pending is saved automatically when the session ends. Optional configuration via `~/.hermes/neatmem.json`:

```json
{
  "base_url": "http://localhost:8790",
  "user_id": "myname",
  "rerank": true
}
```

Verify: tell Hermes "remember that I prefer dark themes", then ask about it in a new session (pending messages are saved on session switch; extraction takes a few seconds). See [hermes/README.md](https://github.com/kanhaoning/NeatMem/blob/main/hermes/README.md) for the full configuration reference and troubleshooting.

## API reference

mem0-compatible endpoints for add, search, list, get, update, delete, and health check, plus a `/v1/messages/` endpoint family for server-side write batching — with curl examples in the [API reference](https://neatmem.readthedocs.io/en/latest/api/).

## Development probes

Memory quality iteration is done through `probe/`, which contains OpenClaw end-to-end probes and extraction simulation scripts. It is not a benchmark suite.

## Design notes

NeatMem is designed around a few constraints:

- keep the plugin layer thin
- keep the backend self-hosted and debuggable
- do not require Redis, an external scheduler, or a message queue
- prefer memory quality over feature breadth
- preserve compatibility with mem0-style APIs where possible

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
