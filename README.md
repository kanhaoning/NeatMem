# NeatMem

Lightweight local memory for agents, with cleaner deduplication, less memory pollution, and more relevant recall.

NeatMem is built for developers who want practical long-term memory without adopting a full Memory OS or hosted memory service. It focuses on keeping local agent memory clean: merging repeated facts, preventing AI suggestions, guesses, and tool noise from being saved as user facts, saving memories with enough context, and filtering irrelevant recalls.

> Status: v0.1-preview. NeatMem is usable for local development and mem0-compatible integrations, but APIs, packaging, and integrations may still change.

> **Benchmark**: 91.01% accuracy on LOCOMO, fully reproducible locally (3-run mean; MiniMax-M3 answer + judge, SiliconFlow bge-m3 embedding). See the [evaluation guide](neatmem/evaluation/README.md) for benchmark reproduction steps.

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

It is not a full Memory OS and not an enterprise multi-tenant memory system.

## Features

- **LLM-assisted memory decisions**
  - Classifies each new memory as `add`, `none`, or `update` (listwise, single LLM call).
  - `DEDUP_MODE` controls behavior: `skip` (keep both), `replace` (overwrite), `rewrite` (LLM merge), `edit` (LLM patch).

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
  - Off by default; graph relations injection into answer prompt is experimental (`GRAPH_INJECT_RELATIONS`, known harmful on LOCOMO).

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

It is designed to work with OpenClaw's memory plugin flow and other mem0-style integrations. v0.1 does not aim to cover every mem0 SDK feature or mem0 hosted-platform behavior.


## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and configure your LLM and embedding provider.

Minimum configuration for OpenAI-compatible LLM providers:

```env
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://your-openai-compatible-endpoint/v1
LLM_MODEL=qwen-max-latest

EMBEDDING_PROVIDER=siliconflow
SILICONFLOW_API_KEY=your-siliconflow-api-key
```

### 3. Start the server

```bash
python -m neatmem.main
```

The server listens on:

```text
http://localhost:8790
```

To use a different port:

```bash
NEATMEM_PORT=9000 python -m neatmem.main
```

Check health:

```bash
curl http://localhost:8790/health
```

Expected response:

```json
{"status":"healthy","timestamp":"..."}
```

## Configuration

NeatMem reads configuration from `.env`.

| Variable | Required | Default | Description |
|---|---:|---|---|
| `NEATMEM_HOST` | no | `0.0.0.0` | Server bind host |
| `NEATMEM_PORT` | no | `8790` | Server port |
| `OPENAI_API_KEY` | yes | - | API key for OpenAI-compatible LLM provider |
| `OPENAI_BASE_URL` | yes | - | OpenAI-compatible API base URL |
| `LLM_MODEL` | no | `qwen-max-latest` | LLM model name |
| `EMBEDDING_PROVIDER` | no | `siliconflow` | `siliconflow` or `xinference` |
| `SILICONFLOW_API_KEY` | conditional | - | Required when `EMBEDDING_PROVIDER=siliconflow` |
| `XINFERENCE_SERVER_URL` | conditional | `http://localhost:9997` | Required when using Xinference |
| `XINFERENCE_MODEL_UID` | conditional | `bge-m3` | Xinference embedding model UID |
| `QDRANT_PATH` | no | `qdrant_db` | Local Qdrant storage path (embedded mode) |
| `QDRANT_HOST` | no | - | Qdrant server host (sets server mode; overrides `QDRANT_PATH`) |
| `QDRANT_PORT` | no | `6333` | Qdrant server port |
| `DEDUP_MODE` | no | `skip` | Dedup behavior: `off`, `skip`, `replace`, `rewrite`, `edit` |
| `ENABLE_BM25` | no | `true` | Enable BM25 sparse search signal |
| `ENABLE_ENTITY` | no | `false` | Enable entity extraction and boosting |
| `ENABLE_GRAPH` | no | `false` | Enable graph memory (KuzuDB entity-relation storage). Graph hooks are no-op when disabled |
| `KUZU_DB_PATH` | conditional | - | KuzuDB database file path. Required when `ENABLE_GRAPH=true` |
| `GRAPH_THRESHOLD` | no | `0.7` | Entity match threshold for graph retrieval |
| `GRAPH_SEARCH_TOP_K` | no | `5` | Max relations returned per speaker from graph search |
| `GRAPH_INJECT_RELATIONS` | no | `false` | Inject graph relations into answer prompt. Only effective when `ENABLE_GRAPH=true`. Experimental: -0.57pp on LOCOMO (2026-07-22), off by default |
| `GRAPH_EMBEDDING_MODEL` | no | `BAAI/bge-m3` | Embedding model for graph entities (defaults to main embedding model) |
| `GRAPH_EMBEDDING_DIMS` | no | `1024` | Embedding dimensions for graph entities |
| `GRAPH_EMBEDDING_BASE_URL` | no | `https://api.siliconflow.cn/v1` | Embedding API base URL for graph entities |
| `GRAPH_EMBEDDING_API_KEY` | no | - | Embedding API key for graph entities. Defaults to `SILICONFLOW_API_KEY` |
| `LLM_RERANK` | no | `true` | Enable LLM listwise rerank for recall |
| `RERANK_MODE` | no | `llm_listwise` | Rerank strategy |
| `RERANK_CANDS` | no | `20` | Head size for LLM listwise rerank: only top N candidates are reordered, the rest are appended in original order. Only effective when `LLM_RERANK=true` |
| `MERGE_STRATEGY` | no | `off` | Deprecated; use `DEDUP_MODE` instead |
| `DEDUP_THINKING` | no | `false` | Enable LLM thinking for dedup |
| `EDIT_THINKING` | no | `false` | Enable LLM thinking for edit mode (DEDUP_MODE=edit) |
| `HISTORY_DB_PATH` | no | `{QDRANT_PATH}/history.db` | SQLite message history database path |
| `EXTRACT_LAST_K_MESSAGES` | no | `10` | Number of recent messages fed to extraction as context |
| `MESSAGE_STORE_BACKEND` | no | `sqlite` | Message store backend: `sqlite` or `none` |
| `ENTITY_EXTRACTOR_BACKEND` | no | `ner` | Entity extractor: `ner` or `llm` |
| `ENTITY_STORE_BACKEND` | no | `qdrant` | Entity store backend |
| `RERANKER_MODEL_PATH` | no | - | Optional local Sentence-Transformers reranker |
| `RERANKER_DEVICE` | no | `cpu` | Reranker device |
| `RERANKER_BATCH_SIZE` | no | `32` | Reranker batch size |
| `RERANKER_TOP_K` | no | `5` | Reranker top-k |
| `HF_ENDPOINT` | no | `https://hf-mirror.com` | HuggingFace mirror endpoint |

## OpenClaw integration

NeatMem includes an OpenClaw plugin under `openclaw/`. Build it and install it as a linked local plugin during development:

```bash
cd /path/to/NeatMem/openclaw
npm install
npm run build

cd /path/to/NeatMem
openclaw plugins install ./openclaw --link
```

After changing plugin TypeScript source, rebuild before reinstalling or restarting OpenClaw.

The plugin id is `openclaw-neatmem`. It talks to NeatMem through the local mem0-compatible HTTP API.

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
          "mode": "platform",
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
openclaw mem0 status
```

The CLI command remains `openclaw mem0` for compatibility, but the active plugin id should be `openclaw-neatmem` and the backend should point to `http://localhost:8790`.

## API examples

### Health check

```bash
curl http://localhost:8790/health
```

### Add memory

```bash
curl -X POST http://localhost:8790/v1/memories/ \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "My name is Alex and I work on agent memory systems."},
      {"role": "assistant", "content": "Nice to meet you, Alex."}
    ],
    "user_id": "default_user",
    "infer": true
  }'
```

### Search memory

```bash
curl -X POST http://localhost:8790/v2/memories/search/ \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is Alex working on?",
    "filters": {"user_id": "default_user"},
    "top_k": 10,
    "threshold": 0.1
  }'
```

### List memories

```bash
curl -X POST http://localhost:8790/v2/memories/ \
  -H "Content-Type: application/json" \
  -d '{
    "filters": {"user_id": "default_user"},
    "page": 1,
    "page_size": 100
  }'
```

### Get memory

```bash
curl http://localhost:8790/v1/memories/{memory_id}/
```

### Update memory

```bash
curl -X PUT http://localhost:8790/v1/memories/{memory_id}/ \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Alex works on local-first agent memory systems.",
    "metadata": {"source": "manual_update"}
  }'
```

### Delete memory

```bash
curl -X DELETE http://localhost:8790/v1/memories/{memory_id}/
```

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
  └─ update -> merge per DEDUP_MODE (skip/replace/rewrite/edit)
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

## Development probes

Memory quality iteration is done through `probe/`, which contains OpenClaw end-to-end probes and extraction simulation scripts. It is not a benchmark suite.

## Design notes

NeatMem is designed around a few constraints:

- keep the plugin layer thin
- keep the backend self-hosted and debuggable
- do not require Redis or a background scheduler
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
- PyPI package publication
- Memory inspection and export/import tools
- Richer recall diagnostics

## License

MIT License.

## Acknowledgements

NeatMem is inspired by the mem0 project and mem0-style memory API patterns, and is designed to interoperate with OpenClaw memory plugin flows. Upstream license notices should be preserved where applicable.

Some utility functions in `neatmem/utils/spacy/` (`spacy_models.py`, `entity_extraction.py`, `lemmatization.py`) are vendored from mem0 v2.0.0 (Apache-2.0); see file headers for modification notes.
