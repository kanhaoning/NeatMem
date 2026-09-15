---
name: search
description: Search memories from earlier Claude Code sessions in this repository. Use it when earlier work may already explain the code, error, decision, or command you need, so you can avoid repeating file reads, searches, or experiments.
argument-hint: "[question] [--top-k number] [--scope repo|mine] [--run-id session-id]"
disable-model-invocation: true
---

# Search memories

Call `search_memories` with the user's question. Treat `--top-k`, `--scope`,
and `--run-id` as tool arguments instead of including them in the query.

Omit `top_k` to use the configured default. Omit `scope` to use the
configured default, normally `repo`: this repository's shared memory, which
everyone who works in it contributes to, plus your own memories. Pass `mine`
to narrow results to memories you saved.

Pass `run_id` with any scope to retrieve memories saved in a specific coding-agent
session. Omit `run_id` to search across sessions. It filters the memories returned;
it does not identify the session making the search request. Use a known session ID,
never invent one. Return the tool's result directly.
