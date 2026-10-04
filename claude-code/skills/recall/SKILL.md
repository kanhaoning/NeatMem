---
name: recall
description: Show what NeatMem last injected into this conversation, including the full memory texts, scores, and the reason when nothing was injected. Use when the user asks what was recalled, why a memory was (or was not) injected, or wants details behind the one-line recall notice.
disable-model-invocation: false
---

# Last recall

Read the persisted recall file and report the result in plain language:

```
${CLAUDE_PLUGIN_DATA}/last_recall.json
```

(Use the Read tool on that path — no shell command needed.)

The JSON has: `ts` / `operation` (`prompt-search` is the automatic search
before a user prompt, `midtask-reminder` is a mid-task search during a long
tool run), `query`, `count`, `items` (each with `score` and the full `text`
of one injected memory), and `skipped` (the reason when nothing was
injected). Summarize these fields, quoting full memory texts when the user
asks about a specific one.

This file holds only the most recent recall — which may be the one triggered
by the user's `/neatmem:recall` prompt itself. When the user asks about an
earlier prompt ("what was injected for my previous question?"), read
`${CLAUDE_PLUGIN_DATA}/recall_history.jsonl` instead: one JSON object per
line, oldest first, same fields — walk back from the last line to find the
entry they mean. If that file is too large to read in full, read its tail.

If neither file exists, say that no memory search has run yet in this
session and note that memories only appear after the first prompt or tool
call. A `memory_cli.py recall [--limit N]` command also prints this data,
but prefer reading the files directly — it needs no command permission.
