# @neatmem/openclaw-neatmem

OpenClaw plugin for using NeatMem as a local, controllable long-term memory backend.

This plugin keeps the OpenClaw side thin: it handles recall injection, capture forwarding, memory tools, and CLI commands, while NeatMem does extraction, deduplication, attribution handling, and recall filtering.

## Quick Start

Prerequisite: a running NeatMem server (`pip install neatmem && neatmem serve`, listens on `http://localhost:8790` by default).

```bash
openclaw plugins install @neatmem/openclaw-neatmem
openclaw neatmem init
```

Then restart the gateway (`openclaw gateway restart`) to load the plugin.

`init` works with zero flags: it writes `apiKey=neatmem-local`, `baseUrl=http://localhost:8790`, and your OS username as `userId`, then validates against the server. Override with `--api-key`, `--user-id`, or `--base-url`.

Or configure manually in `openclaw.json`:

```json5
"openclaw-neatmem": {
  "enabled": true,
  "config": {
    "apiKey": "neatmem-local",
    "userId": "alice",
    "baseUrl": "http://localhost:8790"
  }
}
```

### Development (from source)

```bash
cd /path/to/NeatMem/openclaw
pnpm install
pnpm run build

cd /path/to/NeatMem
openclaw plugins install ./openclaw --link
```

After changing plugin TypeScript source, rebuild before reinstalling or restarting OpenClaw.

## How It Works

**Auto-Recall** — Before the agent responds, the plugin searches NeatMem for relevant memories and injects them into context.

**Auto-Capture** — After the agent responds, the conversation is filtered through a noise-removal pipeline and sent to NeatMem. New facts get stored, stale ones updated, duplicates merged.

Both run silently. No prompting, no manual calls required.

### Memory Scopes

- **Session (short-term)** — Scoped to the current conversation via `run_id`. Recalled alongside long-term memories.
- **User (long-term)** — Persistent across all sessions.

### Multi-Agent Isolation

Each agent gets its own memory namespace automatically via session key routing (`agent:<name>:<uuid>` maps to `userId:agent:<name>`). Single-agent setups are unaffected.

## Agent Tools

Five tools are registered for agent use:

| Tool | Description |
| ---- | ----------- |
| `memory_search` | Search by natural language query. Supports `scope` (`session`, `long-term`, `all`), `categories`, `filters`, and `agentId`. |
| `memory_get` | Retrieve a single memory by ID. |
| `memory_list` | List all memories. Filter by `userId`, `agentId`, `scope`. |
| `memory_update` | Update a memory's text in place. Preserves history. |
| `memory_delete` | Delete by `memoryId`, `query` (search-and-delete), or `all: true` (requires `confirm: true`). |

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
openclaw neatmem init --api-key <key> --user-id alice --base-url http://192.168.1.10:8790
openclaw neatmem status
openclaw neatmem config show
openclaw neatmem config get api_key
openclaw neatmem config set user_id alice
```

## Configuration Reference

| Key | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `apiKey` | `string` | — | **Required.** NeatMem API key (supports `${NEATMEM_API_KEY}`) |
| `baseUrl` | `string` | `http://localhost:8790` | NeatMem server URL |
| `userId` | `string` | OS username | User identifier. All memories scoped to this value. |
| `autoRecall` | `boolean` | `true` | Inject relevant memories before each turn |
| `autoCapture` | `boolean` | `true` | Extract and store facts after each turn |
| `topK` | `number` | `5` | Max memories returned per recall |
| `searchThreshold` | `number` | `0.1` | Minimum similarity score (0-1) |
| `customInstructions` | `string` | *(built-in)* | Custom extraction rules |
| `customCategories` | `object` | *(12 defaults)* | Category hints sent to the extraction model |
| `recall` | `object` | — | Recall tuning: `threshold`, `rerank`, `keywordSearch`, `filterMemories` |

## License

[Apache 2.0](LICENSE)

Derived from [mem0](https://github.com/mem0ai/mem0)'s OpenClaw plugin (Apache 2.0), independently maintained since v1.0.6.
