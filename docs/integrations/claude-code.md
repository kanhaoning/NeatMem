# Claude Code Integration

With the NeatMem server running at `http://localhost:8790`:

```bash
claude plugin marketplace add kanhaoning/NeatMem
claude plugin install neatmem@neatmem
```

Open a new session after installing — plugins load at session start. The plugin captures each session and extracts memories when the session ends and before context compaction; every prompt searches and injects relevant memories, and Claude can also search anytime via the `search_memories` MCP tool or `/neatmem:search`.

Whenever memories are injected, the plugin prints a short notice right below your prompt so you can see what was recalled. For long autonomous tasks, you can also opt into mid-task injections (memories searched and injected while Claude works, not just at each prompt) by setting `NEATMEM_CODE_MIDTASK_REMINDER_ENABLED=1` before starting a session.

Defaults need no configuration (`localhost:8790`, your OS account as the memory user). To point at a different server, set `NEATMEM_API_URL` in the `env` section of `~/.claude/settings.json`, or use the `/plugin` configuration page.

Verify: say "remember that I prefer dark themes", `/exit`, then ask about it in a new session in the same directory. See [claude-code/README.md](https://github.com/kanhaoning/NeatMem/blob/main/claude-code/README.md) for the full configuration reference, command list and troubleshooting.
