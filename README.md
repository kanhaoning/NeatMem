# NeatMem

Lightweight local memory for agents, with cleaner deduplication, less memory pollution, and more relevant recall.

NeatMem is built for developers who want practical long-term memory without adopting a full Memory OS, heavy knowledge graph platform, or hosted memory service. It focuses on keeping local agent memory clean: merging repeated facts, preventing AI suggestions, guesses, and tool noise from being saved as user facts, saving memories with enough context, and filtering irrelevant recalls.

> Status: v0.1-preview. NeatMem is usable for local development and OpenClaw-style mem0-compatible integrations, but APIs, packaging, and integrations may still change.

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

It is not a full Memory OS, not a heavy knowledge graph platform, and not an enterprise multi-tenant memory system.

## Features

- **LLM-assisted memory decisions**
  - Classifies each new memory as `redundant`, `relevant`, or `independent`.
  - Replaces duplicates, merges related facts, and keeps unrelated memories separate.

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
  - Optional LLM rerank filters irrelevant vector-search hits before they are injected back into the agent context.

- **Lightweight local storage**
  - Runs with a simple local vector store by default.
  - Does not require Neo4j, Redis, a hosted memory service, or a full database stack.

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

## What NeatMem is not

NeatMem intentionally avoids platform-level complexity in the first version.

It is not:

- a full Memory OS
- a full knowledge graph platform
- a multi-tenant enterprise memory platform
- a dashboard product
- a replacement for all mem0 features
- a benchmark suite

The first version focuses on memory quality and local debuggability.

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
python main.py
```

The server listens on:

```text
http://localhost:8790
```

To use a different port:

```bash
NEATMEM_PORT=9000 python main.py
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
| `LLM_RERANK` | no | `true` | Enable LLM-based recall filtering |
| `RERANKER_MODEL_PATH` | no | - | Optional local Sentence-Transformers reranker |
| `RERANKER_DEVICE` | no | `cpu` | Reranker device |
| `RERANKER_BATCH_SIZE` | no | `32` | Reranker batch size |
| `RERANKER_TOP_K` | no | `5` | Reranker top-k |
| `HF_ENDPOINT` | no | `https://hf-mirror.com` | HuggingFace mirror endpoint |

## OpenClaw integration

NeatMem includes an OpenClaw plugin under `openclaw/`. Build it and install it as a linked local plugin during development:

```bash
cd /path/to/NeatMem/openclaw
pnpm install
pnpm run build

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
retrieve existing memories
  ↓
LLM memory extraction
  ↓
context completion and source tracking
  ↓
sequential LLM-assisted memory decisions
  ├─ redundant   → replace old memory
  ├─ relevant    → merge with latest old memory
  └─ independent → add as new memory
  ↓
write through the mem0-style memory API
```

### Search flow

```text
query
  ↓
vector search
  ↓
optional LLM recall filtering
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
- do not require Neo4j, Redis, or a background scheduler
- prefer memory quality over feature breadth
- preserve compatibility with mem0-style APIs where possible

## Limitations

Current preview limitations:

- API shape may still change.
- Packaging is not finalized.
- No dashboard.
- No enterprise multi-tenant permission system.
- No required graph database or full knowledge graph stack in v0.1.
- OpenClaw integration is the primary tested integration path.
- Prompt behavior is still being iterated and may vary across models.

## Roadmap

Near-term:

- finalize dependency list and package layout
- document OpenClaw setup with a full working example
- promote probe cases for memory quality regression
- add minimal public smoke tests
- optional Python package entrypoint

Later:

- standalone OpenClaw plugin package
- benchmark suite
- import/export tools
- memory inspection tools
- richer recall diagnostics

## License

MIT License.

## Acknowledgements

NeatMem is inspired by the mem0 project and mem0-style memory API patterns, and is designed to interoperate with OpenClaw memory plugin flows. Upstream license notices should be preserved where applicable.
