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

## Development probes

Memory quality iteration is done through `probe/`, which contains OpenClaw end-to-end probes and extraction simulation scripts. It is not a benchmark suite.

## Design notes

NeatMem is designed around a few constraints:

- keep the plugin layer thin
- keep the backend self-hosted and debuggable
- do not require Redis or a background scheduler
- prefer memory quality over feature breadth
- preserve compatibility with mem0-style APIs where possible
