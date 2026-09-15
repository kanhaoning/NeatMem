---
name: status
description: Show whether NeatMem memory is working in this repository, covering configuration, capture state, flush counts, and whether the NeatMem server is reachable. Use when the user asks whether memory is on, why a memory is missing, or anything looks broken.
disable-model-invocation: false
---

# Memory status

Run both commands and report the combined result in plain language:

```bash
python3 "${CLAUDE_PLUGIN_ROOT}/core/memory_cli.py" --harness "claude-code" --plugin-data-dir "${CLAUDE_PLUGIN_DATA}" status --json
python3 "${CLAUDE_PLUGIN_ROOT}/core/memory_cli.py" --harness "claude-code" --plugin-data-dir "${CLAUDE_PLUGIN_DATA}" doctor
```

Summarize, using only fields the JSON actually reports: whether capture is
active or paused, the user ID and repository scope (`repo_id`), the server
URL, the event/flush/retrieval counts (`flushes` is the number of completed
flushes, not a pending count), and the doctor check results. If doctor reports
the server is unreachable, say clearly that memories are NOT being created and
suggest checking that the NeatMem server is running and that the configured
`api_url` points at it.
