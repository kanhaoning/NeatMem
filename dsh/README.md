# @neatmem/dsh-neatmem

DeepSeek Harness (dsh) plugin for using NeatMem as a long-term memory backend.

The plugin is a thin REST client: it handles recall injection, turn capture, and memory tools, while the NeatMem server does extraction, deduplication, batching, and recall filtering. No native dependencies — nothing to compile or approve at install time.

## Quick Start

Prerequisites:

- A running NeatMem server (`pip install neatmem && neatmem serve`, listens on `http://localhost:8790` by default)
- A dsh installation with a configured model route (verified against dsh `0.1.5-rc.2`)

```bash
dsh plugin --profile <name> add @neatmem/dsh-neatmem
```

Restart the profile to load the plugin. Verify the bundle composed:

```bash
dsh --profile <name> --dump-config   # shows a `neatmem-dsh` row
```

The plugin works with zero configuration — defaults are `baseUrl=http://localhost:8790`, `userId=default`, recall/capture/tools all enabled. At startup it pings the server once; if unreachable it warns and degrades (chats keep working, no recall/capture) until the server is back.

### Configuration

Override any field per profile in `$DSH_HOME/profiles/<name>/cordis.patch.yml`:

```yaml
- id: neatmem-dsh
  config:
    baseUrl: http://localhost:8790
    userId: myname
    topK: 5
```

| Field | Default | Purpose |
|---|---|---|
| `enabled` | `true` | Master switch |
| `baseUrl` | `http://localhost:8790` | NeatMem server URL |
| `apiKey` | `''` | Optional Token auth header |
| `userId` | `default` | Memory namespace (`user_id`) |
| `agentId` | `''` | Optional agent-preset scope (`agent_id`) |
| `recallEnabled` / `captureEnabled` / `toolsEnabled` | `true` | Feature switches |
| `recallTimeoutMs` | `3000` | Auto-recall budget (hard-capped at 3s, fail-open) |
| `topK` | `5` | Max memories injected per turn |
| `searchThreshold` | `0.3` | Minimum relevance score |
| `contextMaxChars` / `toolResultMaxChars` | `6000` / `1200` | Bounded rendering |

Invalid configuration fails at load time; an unreachable server only downgrades memory, never the chat.

## How It Works

**Auto-recall** — on each direct-user turn's first step, the plugin runs one bounded search (`rerank` off, fast path) and injects hits as a source-labelled user message (`plugin/neatmem-dsh/recall`), reconstructable from the session log. Plugin/tool messages never trigger recall.

**Auto-capture** — each finished turn (user + assistant text) is forwarded via `POST /v1/messages/add/` (store-first; the server's scheduler batches extraction). On session close, a bounded flush (`/v1/messages/flush/`) forces extraction of anything pending. Capture runs on a serial per-session queue and never blocks a turn.

**Memory tools** — the agent can call `memory_search` (exposes `rerank` for accuracy), `memory_get`, `memory_list`, `memory_update`, `memory_delete`. There is intentionally no `memory_add`: capture is automatic.

Scoping: `run_id` maps to the dsh session id (enables `sessionScope` in tools), `user_id` is your configured namespace.

## Development (from source)

```bash
cd /path/to/NeatMem/dsh
pnpm install
pnpm run build

cd /path/to/deepseek-harness
pnpm dsh plugin --profile <name> add /path/to/NeatMem/dsh
```

Rebuild after changing TypeScript source, then restart the profile.

### Publishing

`npm publish` (prepublishOnly rebuilds `dist`; the tarball ships only `dist` + `cordis.patch.yml`).
