# NeatMem Plugin for Claude Code

Use a self-hosted [NeatMem](https://github.com/kanhaoning/NeatMem) server as the long-term memory backend for [Claude Code](https://claude.com/claude-code).

Extraction, deduplication and retrieval all run in the NeatMem server; the plugin only delivers session content to it and brings relevant memories back.

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
claude plugin marketplace add kanhaoning/NeatMem
claude plugin install neatmem@neatmem
```

The interactive `/plugin` menu performs the same two steps. Plugins load at session start, so open a new session after installing.

Manage with `claude plugin update neatmem` / `uninstall neatmem` / `marketplace remove neatmem`. Updates apply on the next session.

## Configuration

Defaults work out of the box (server `localhost:8790`, your OS account as user). Adjust in the `/plugin` configuration page for the plugin, or via environment variables in `~/.claude/settings.json` (environment wins over form values):

```json
{
  "env": {
    "NEATMEM_API_URL": "http://localhost:8790"
  }
}
```

| Option | Environment | Default | Description |
|---|---|---|---|
| `api_url` | `NEATMEM_API_URL` | `http://127.0.0.1:8790` | NeatMem server URL |
| `api_key` | `NEATMEM_API_KEY` | empty | Only when the server sits behind an authenticating proxy; form values are stored in the system credential store |
| `user_id` | `NEATMEM_USER_ID` | OS account | Stable ID to share memories across machines |
| `search_scope` | — | `repo` | `repo`: repository-shared plus your own memories; `mine`: only yours |
| `top_k` | — | 3 | Max memories returned per manual search |
| `max_context_chars` | — | 4000 | Max injected memory characters per search |
| — | `NEATMEM_CODE_RECENT_MEMORY_DELAY_SECONDS` | 1800 | Local override for the server policy below |
| — | `NEATMEM_CODE_SEARCH_TIMEOUT` | 5 | HTTP timeout (seconds) for automatic prompt-search calls |

`NEATMEM_ENABLED=0 claude` disables capture and injection for a single session (MCP tools stay available). `/neatmem:pause` disables persistently across sessions until `/neatmem:unpause`.

Auto-injection behavior (when to inject, minimum prompt length, recent-memory delay) is configured on the server (`INJECT_TIMING`, `MIN_QUERY_CHARS`, `RECENT_MEMORY_DELAY_SECONDS`); all clients apply it when a session starts. The delay keeps memories produced by the current session out of automatic injection for their first 30 minutes — their content is still in the conversation, so injecting them adds nothing. Memories from before the last compact are exempt (compaction dropped them from context), and explicit search (`/neatmem:search`, MCP) is never delayed.

## How it works

- **Write**: hooks capture prompts, tool results and responses into a local queue. The server extracts memories in batches; extraction is also forced when the session ends and before context compaction, so the next session can already search them.
- **Recall**: every user prompt triggers a search and injects the hits as context (messages shorter than `MIN_QUERY_CHARS` are skipped). Claude can also search at any time through the `search_memories` MCP tool or `/neatmem:search`.
- **Scope**: memories are tagged with your user ID, the repository, and the session. Default search covers the current repository's shared memories plus your own.

## Commands

| Command | Description |
|---|---|
| `/neatmem:search` | Search memories interactively |
| `/neatmem:status` | Show capture state, scope, server reachability and flush counts |
| `/neatmem:resume` | Summarize memories relevant to the current task |
| `/neatmem:pause` | Stop capturing and injecting (persists across sessions) |
| `/neatmem:unpause` | Resume capturing and injecting |

## Verify

1. In a project directory, tell Claude: *"Remember that my build command is make -j8."*
2. `/exit` — the session-end flush extracts the memory within seconds.
3. Start a new session in the same directory and ask about it. The prompt triggers a search and the answer should recall the fact.
4. Server-side check: `curl -X POST http://localhost:8790/v2/memories/ -H 'Content-Type: application/json' -d '{"filters":{"user_id":"<your-account>"}}'`

## Troubleshooting

- **Plugin not active**: plugins load at session start — open a new session. `claude plugin list` should show `neatmem@neatmem` as enabled.
- **No memories created**: run `/neatmem:status` and confirm the server is reachable (`curl $NEATMEM_API_URL/v1/ping/`). Capture resumes automatically once it is.
- **No injected context on a prompt**: prompts shorter than `MIN_QUERY_CHARS` (default 5) skip auto-injection; `/neatmem:search` or the MCP tool always works.
- **Memory saved in another project not found**: search always filters by the current repository. `search_scope=mine` only narrows "repository-shared plus yours" to "yours" — it does not search across projects.
- **Changed server-side recall settings not taking effect**: clients apply server settings when a session starts; restart the session.
