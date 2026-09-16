# Claude Code Integration

With the NeatMem server running at `http://localhost:8790`:

```bash
claude plugin marketplace add kanhaoning/NeatMem
claude plugin install neatmem@neatmem
```

Open a new session after installing — plugins load at session start. The plugin captures each session and extracts memories when the session ends and before context compaction; the first message of every new session searches and injects relevant memories, and Claude can search anytime via the `search_memories` MCP tool or `/neatmem:search`.

Defaults need no configuration (`localhost:8790`, your OS account as the memory user). To point at a different server, set `NEATMEM_API_URL` in the `env` section of `~/.claude/settings.json`, or use the `/plugin` configuration page.

Verify: say "remember that I prefer dark themes", `/exit`, then ask about it in a new session in the same directory. See [claude-code/README.md](https://github.com/kanhaoning/NeatMem/blob/main/claude-code/README.md) for the full configuration reference, command list and troubleshooting.
