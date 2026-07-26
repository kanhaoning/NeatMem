# NeatMem Memory Provider for Hermes Agent

Use a self-hosted [NeatMem](https://github.com/kanhaoning/NeatMem) server as the long-term memory backend for [Hermes Agent](https://github.com/NousResearch/hermes-agent).

NeatMem provides server-side LLM fact extraction, deduplication, entity boosting, BM25 hybrid search and optional LLM rerank. This plugin is a thin transport layer — all memory intelligence happens in the NeatMem server.

## Prerequisites

A running NeatMem server (default `http://localhost:8790`):

```bash
pip install neatmem   # or run from source
neatmem serve
curl http://localhost:8790/v1/ping/   # → {"status":"ok",...}
```

See the [main repo](https://github.com/kanhaoning/NeatMem) for server setup (LLM/embedder configuration).

## Install

```bash
hermes plugins install kanhaoning/NeatMem/hermes --enable
hermes config set memory.provider neatmem
```

Without `--enable`, the installer asks whether to enable the plugin (or run `hermes plugins enable neatmem` later). Verify with `hermes plugins list` — `neatmem` should show as `enabled`.

No restart needed for CLI/TUI (each launch loads plugins fresh); only a running gateway needs `hermes gateway restart`.

Manage the plugin with `hermes plugins update neatmem` / `disable` / `remove`.

## Configuration

Defaults work out of the box (`localhost:8790`, user `hermes-user`). To customize, create `~/.hermes/neatmem.json`:

```json
{
  "base_url": "http://localhost:8790",
  "user_id": "myname",
  "agent_id": "hermes",
  "rerank": true
}
```

| Key | Default | Description |
|---|---|---|
| `base_url` | `http://localhost:8790` | NeatMem server URL |
| `user_id` | `hermes-user` | Canonical user id; memories are scoped and recalled per user |
| `agent_id` | `hermes` | Agent id attached to writes |
| `rerank` | `true` | Default LLM rerank for the `neatmem_search` tool (prefetch always uses rerank=false for speed) |

Environment variables `NEATMEM_BASE_URL` / `NEATMEM_USER_ID` / `NEATMEM_AGENT_ID` / `NEATMEM_API_KEY` are read as fallback defaults (the JSON file takes precedence).

To disable Hermes' built-in curated memory (MEMORY.md/USER.md) and use NeatMem exclusively:

```bash
hermes config set memory.memory_enabled false
hermes config set memory.user_profile_enabled false
```

## How it works

- **Write**: after each turn, the user/assistant pair is queued and sent to `POST /v1/memories/` with `infer=True` (server-side extraction + dedup). Writes go through a serial queue — no turn is silently dropped.
- **Recall**: on each turn a background prefetch searches with `rerank=false` (fast path) and injects up to 10 hits into the context. The model can also call tools: `neatmem_search` (rerank on by default), `neatmem_add`, `neatmem_list`, `neatmem_update`, `neatmem_delete`.

## Agent Tools

| Tool | Description |
|---|---|
| `neatmem_search` | Semantic search; `top_k` (max 50), `rerank` (default true) |
| `neatmem_add` | Store a fact verbatim (no extraction) |
| `neatmem_list` | Paginated full list (max 200/page) |
| `neatmem_update` | Replace memory text by ID |
| `neatmem_delete` | Delete by ID |

## Verify

1. Tell Hermes: *"Remember that I prefer dark themes."*
2. Check the server: `curl -X POST http://localhost:8790/v2/memories/ -H 'Content-Type: application/json' -d '{"filters":{"user_id":"hermes-user"}}'`
3. In a new session, ask about it — the model should recall via `neatmem_search`.

## Troubleshooting

- **Provider not listed** (`hermes plugins list`): check `~/.hermes/plugins/neatmem/` exists with `plugin.yaml`, `__init__.py`, `_backend.py`.
- **Searches/writes fail**: confirm the server is reachable (`curl $base_url/v1/ping/`). The plugin pauses calls for 120s after 5 consecutive failures (circuit breaker) — check `hermes logs` for warnings.
- **No auto-injected memories**: the prefetch hot path waits only 1.5s; on slow setups injection may be skipped. The `neatmem_search` tool is the reliable path.
- **GitHub clone timed out after 60 seconds**: the installer shallow-clones with a hardcoded 60s timeout; the download is only ~4 MB, so just retry. `hermes plugins update` uses the same path — same advice.
