# @neatmem/openclaw-neatmem

OpenClaw plugin for using NeatMem as a local, controllable long-term memory backend.

This plugin keeps the OpenClaw side thin: it handles recall injection, capture forwarding, memory tools, and CLI commands, while NeatMem does extraction, deduplication, attribution handling, and recall filtering.

## Quick Start

```bash
cd /path/to/NeatMem/openclaw
pnpm install
pnpm run build

cd /path/to/NeatMem
openclaw plugins install ./openclaw --link
```

After changing plugin TypeScript source, rebuild before reinstalling or restarting OpenClaw.

### Platform mode with local NeatMem

Start the NeatMem backend at `http://localhost:8790`, then configure the plugin:

```bash
openclaw neatmem init --api-key neatmem-local --user-id <your-user-id>
```

Or configure manually in `openclaw.json`:

```json5
"openclaw-neatmem": {
  "enabled": true,
  "config": {
    "mode": "platform",
    "apiKey": "neatmem-local",
    "userId": "alice",
    "baseUrl": "http://localhost:8790"
  }
}
```

### Open-Source (Self-hosted)

No NeatMem key needed. Requires `OPENAI_API_KEY` for default embeddings and LLM. Vectors are stored locally in SQLite at `~/.mem0/vector_store.db` — no external database required.

Defaults: `text-embedding-3-small` for embeddings, `gpt-5.4` for fact extraction.

```json5
"openclaw-neatmem": {
  "enabled": true,
  "config": {
    "mode": "open-source",
    "userId": "alice"
  }
}
```

Customize the embedder, vector store, or LLM via the `oss` block:

```json5
"config": {
  "mode": "open-source",
  "userId": "alice",
  "oss": {
    "embedder": { "provider": "openai", "config": { "model": "text-embedding-3-small" } },
    "vectorStore": { "provider": "qdrant", "config": { "host": "localhost", "port": 6333 } },
    "llm": { "provider": "openai", "config": { "model": "gpt-5.4" } }
  }
}
```

All `oss` fields are optional.

## How It Works

**Auto-Recall** — Before the agent responds, the plugin searches NeatMem for relevant memories and injects them into context.

**Auto-Capture** — After the agent responds, the conversation is filtered through a noise-removal pipeline and sent to NeatMem. New facts get stored, stale ones updated, duplicates merged.

Both run silently. No prompting, no manual calls required.

### Memory Scopes

- **Session (short-term)** — Scoped to the current conversation via `run_id`. Recalled alongside long-term memories.
- **User (long-term)** — Persistent across all sessions. Default for `memory_add`.

### Multi-Agent Isolation

Each agent gets its own memory namespace automatically via session key routing (`agent:<name>:<uuid>` maps to `userId:agent:<name>`). Single-agent setups are unaffected.

## Agent Tools

Eight tools are registered for agent use:

| Tool | Description |
| ---- | ----------- |
| `memory_search` | Search by natural language query. Supports `scope` (`session`, `long-term`, `all`), `categories`, `filters`, and `agentId`. |
| `memory_add` | Store facts. Accepts `text` or `facts` array, `category`, `importance`, `longTerm`, `metadata`. |
| `memory_get` | Retrieve a single memory by ID. |
| `memory_list` | List all memories. Filter by `userId`, `agentId`, `scope`. |
| `memory_update` | Update a memory's text in place. Preserves history. |
| `memory_delete` | Delete by `memoryId`, `query` (search-and-delete), or `all: true` (requires `confirm: true`). |
| `memory_event_list` | List recent background processing events. Platform mode only. |
| `memory_event_status` | Get status of a specific event by ID. Platform mode only. |

## CLI

All commands: `openclaw neatmem <command>`.

```bash
# Memory operations
openclaw neatmem add "User prefers TypeScript over JavaScript"
openclaw neatmem search "what languages does the user know"
openclaw neatmem search "preferences" --scope long-term
openclaw neatmem get <memory_id>
openclaw neatmem list --user-id alice --top-k 20
openclaw neatmem update <memory_id> "Updated preference text"
openclaw neatmem delete <memory_id>
openclaw neatmem delete --all --user-id alice --confirm
openclaw neatmem import memories.json

# Management
openclaw neatmem init
openclaw neatmem init --api-key <key> --user-id alice
openclaw neatmem status
openclaw neatmem config show
openclaw neatmem config get api_key
openclaw neatmem config set user_id alice

# Events (platform only)
openclaw neatmem event list
openclaw neatmem event status <event_id>

# Memory consolidation
openclaw neatmem dream
openclaw neatmem dream --dry-run
```

## Configuration Reference

### General

| Key | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `mode` | `"platform"` \| `"open-source"` | `"platform"` | Backend mode |
| `userId` | `string` | OS username | User identifier. All memories scoped to this value. |
| `autoRecall` | `boolean` | `true` | Inject relevant memories before each turn |
| `autoCapture` | `boolean` | `true` | Extract and store facts after each turn |
| `topK` | `number` | `5` | Max memories returned per recall |
| `searchThreshold` | `number` | `0.1` | Minimum similarity score (0-1) |

### Platform Mode

| Key | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `apiKey` | `string` | — | **Required.** NeatMem API key (supports `${NEATMEM_API_KEY}`) |
| `customInstructions` | `string` | *(built-in)* | Custom extraction rules |
| `customCategories` | `object` | *(12 defaults)* | Category name to description map |

### Open-Source Mode

All fields optional. Defaults: `text-embedding-3-small` embeddings, local SQLite vector store (`~/.mem0/vector_store.db`), `gpt-5.4` LLM.

| Key | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `customPrompt` | `string` | *(built-in)* | Extraction prompt |
| `oss.embedder.provider` | `string` | `"openai"` | Embedding provider |
| `oss.embedder.config` | `object` | — | Provider config (`apiKey`, `model`, `baseURL`) |
| `oss.vectorStore.provider` | `string` | `"memory"` | Vector store provider (see list above) |
| `oss.vectorStore.config` | `object` | — | Provider config (`host`, `port`, `collectionName`, `dbPath`) |
| `oss.llm.provider` | `string` | `"openai"` | LLM provider |
| `oss.llm.config` | `object` | — | Provider config (`apiKey`, `model`, `baseURL`) |
| `oss.historyDbPath` | `string` | — | SQLite path for edit history |

## License

[Apache 2.0](LICENSE)
