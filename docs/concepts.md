# Concepts

## How it works

### Add flow

```text
messages
  ↓
retrieve last-k messages as extraction context
  ↓
LLM memory extraction (with last-k context)
  ↓
context completion and source tracking
  ↓
sequential LLM-assisted memory decisions
  ├─ add    -> store as new memory
  ├─ none   -> skip (duplicate)
  └─ update -> merge per DEDUP_MODE (skip/replace/rewrite/edit)
  ↓
write to vector store + BM25 index + entity store
```

### Search flow

```text
query
  ↓
dense vector search + BM25 sparse search + entity boosting
  ↓
LLM listwise rerank
  ↓
threshold filtering
  ↓
results
```

### Server-side write batching (queue mode)

Clients can either send messages for immediate extraction (`POST /v1/memories/`) or just forward them (`POST /v1/messages/add/`) and let the server batch extraction:

- an in-process scheduler extracts a batch once `MESSAGE_BATCH_SIZE` messages are pending, or once the oldest pending message exceeds `MESSAGE_BATCH_DEADLINE_SECS`
- a per-user cursor (`message_cursor` table) tracks how far extraction has progressed; a failed batch is retried on the next round, and a server restart resumes from the cursor — no message is extracted twice or skipped
- `POST /v1/messages/flush/` forces extraction of everything pending for a scope (used e.g. at session end)
- both paths share the same extraction pipeline, so batching changes scheduling, not memory semantics

## Development probes

Memory quality iteration is done through `probe/`, which contains OpenClaw end-to-end probes and extraction simulation scripts. It is not a benchmark suite.

## Design notes

NeatMem is designed around a few constraints:

- keep the plugin layer thin
- keep the backend self-hosted and debuggable
- do not require Redis, an external scheduler, or a message queue
- prefer memory quality over feature breadth
- preserve compatibility with mem0-style APIs where possible
