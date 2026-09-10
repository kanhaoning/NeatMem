# NeatMem

[![PyPI](https://img.shields.io/pypi/v/neatmem)](https://pypi.org/project/neatmem/)
[![Documentation](https://readthedocs.org/projects/neatmem/badge/?version=latest)](https://neatmem.readthedocs.io/en/latest/)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/neatmem?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/neatmem)

Lightweight local memory for agents — every dedup, update, and rerank decision inspectable and tunable.

3 dedup detectors × 4 update resolvers × 3 rerank modes · 60+ parameters · 6 prompts replaceable

> **Docs**: [neatmem.readthedocs.io](https://neatmem.readthedocs.io/en/latest/) — full quick start, configuration reference, custom prompts, API reference, and integration guides.

> Status: actively developed (v0.5.x). NeatMem is usable for local development and mem0-compatible client integrations, but APIs, packaging, and integrations may still change.

> **Benchmark**: 90.8% accuracy on LoCoMo (MiniMax-M3 answer + judge, SiliconFlow bge-m3 embedding). See the [evaluation guide](https://neatmem.readthedocs.io/en/latest/evaluation/) for reproduction steps.

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
  - Detection mode, update behavior, and dedup itself are all switchable — see the [configuration reference](https://neatmem.readthedocs.io/en/latest/configuration/).

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
rerank (LLM listwise/pointwise or cross-encoder)
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

It is designed to work with OpenClaw's and Hermes' memory plugin flows and other mem0-style integrations. It does not aim to cover every mem0 SDK feature or hosted-platform behavior.

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

Full method and parameter reference: [Python Client](https://neatmem.readthedocs.io/en/latest/client/).

The client also provides server-side write batching (`add_messages`, `get_next_batch`, `mark_batch_processed`, `flush_messages`) and raw message history access (`client.messages` — `query`, `sessions`, `delete`, `reset`).

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
| `EMBEDDER_PROVIDER` | no | `siliconflow` | `siliconflow`, `openai`, `dashscope`, or `xinference` |
| `EMBEDDER_API_KEY` | conditional | - | Required for hosted embedding providers |
| `EMBEDDER_MODEL` | no | `BAAI/bge-m3` | Embedding model name |
| `NEATMEM_PORT` | no | `8790` | Server port |
| `DEDUP_ENABLED` | no | `true` | Enable dedup on write |
| `DEDUP_RESOLVER` | no | `rewrite` | Duplicate resolution: `skip`, `replace`, `rewrite`, `edit` |

## Custom prompts

Every core prompt (extraction, dedup, merge rewrite, group rewrite, patch edit, rerank) can be replaced with your own prompt file — see the [custom prompts guide](https://neatmem.readthedocs.io/en/latest/custom-prompts/).

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
