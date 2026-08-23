# API Reference

## Health check

```bash
curl http://localhost:8790/health
```

## Add memory

```bash
curl -X POST http://localhost:8790/v1/memories/ \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "My name is Alex and I work on agent memory systems."},
      {"role": "assistant", "content": "Nice to meet you, Alex."}
    ],
    "user_id": "default_user",
    "infer": true
  }'
```

`POST /v1/memories/` also accepts an optional `message_ids` field (list of message IDs previously stored via `/v1/messages/add/`) instead of `messages`; the server then extracts from those stored messages. The two fields are mutually exclusive.

## Message batching (queue mode)

These endpoints add server-side write batching. With `MESSAGE_BATCHING_ENABLED=true` (the default), clients can forward raw messages as they happen and let the server extract memories in fixed-size batches: a batch runs once it reaches `MESSAGE_BATCH_SIZE` messages, or when the oldest pending message exceeds `MESSAGE_BATCH_DEADLINE_SECS`.

### Store messages without extraction

```bash
curl -X POST http://localhost:8790/v1/messages/add/ \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "My name is Alex."}],
    "user_id": "default_user"
  }'
# {"results": [{"message_id": "...", "seq": 101}], "count": 1}
```

### Get the next pending batch

```bash
curl -X POST http://localhost:8790/v1/messages/next-batch/ \
  -H "Content-Type: application/json" \
  -d '{"user_id": "default_user"}'
# {"message_ids": ["..."], "seqs": [101, 102], "pending_count": 15}
```

### Advance the extraction cursor

Call only after a batch was extracted successfully:

```bash
curl -X POST http://localhost:8790/v1/messages/mark-processed/ \
  -H "Content-Type: application/json" \
  -d '{"user_id": "default_user", "last_processed_seq": 110}'
# {"marked": true, "last_processed_seq": 110}
```

### Flush pending messages now

Extracts everything pending for the scope, ignoring the batch-size and deadline conditions:

```bash
curl -X POST http://localhost:8790/v1/messages/flush/ \
  -H "Content-Type: application/json" \
  -d '{"user_id": "default_user"}'
# {"batches": 2, "extracted_count": 4, "last_processed_seq": 110}
```

## Message history

Raw messages stored via `/v1/messages/add/` (or inline adds) can be inspected and managed directly. Client equivalents live under `client.messages` (`query`, `sessions`, `delete`, `reset`).

### Query stored messages

```bash
curl -X POST http://localhost:8790/v1/messages/query/ \
  -H "Content-Type: application/json" \
  -d '{"user_id": "default_user", "limit": 100, "order": "desc"}'
# {"messages": [...], "total": 231, "limit": 100, "offset": 0}
```

Optional filters: `agent_id`, `run_id`, `app_id`, `roles`, `content_like` (substring match), `after`/`before` (timestamp range), `offset`, `order` (`asc`/`desc`). At least one scope filter is required.

### List sessions

```bash
curl -X POST http://localhost:8790/v1/messages/sessions/ \
  -H "Content-Type: application/json" \
  -d '{"user_id": "default_user"}'
# {"sessions": [...], "total": 3}
```

### Delete stored messages

Deletes raw messages matching the given scope filters (at least one required). Does not touch extracted memories.

```bash
curl -X POST http://localhost:8790/v1/messages/delete/ \
  -H "Content-Type: application/json" \
  -d '{"user_id": "default_user", "run_id": "session-42"}'
# {"deleted": 18}
```

### Reset all stored messages

Deletes ALL raw messages and resets extraction cursors. Does not touch extracted memories.

```bash
curl -X POST http://localhost:8790/v1/messages/reset/
# {"reset": true}
```

## Search memory

```bash
curl -X POST http://localhost:8790/v2/memories/search/ \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is Alex working on?",
    "filters": {"user_id": "default_user"},
    "top_k": 10,
    "threshold": 0.1
  }'
```

## List memories

```bash
curl -X POST http://localhost:8790/v2/memories/ \
  -H "Content-Type: application/json" \
  -d '{
    "filters": {"user_id": "default_user"},
    "page": 1,
    "page_size": 100
  }'
```

## Get memory

```bash
curl http://localhost:8790/v1/memories/{memory_id}/
```

## Update memory

```bash
curl -X PUT http://localhost:8790/v1/memories/{memory_id}/ \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Alex works on local-first agent memory systems.",
    "metadata": {"source": "manual_update"}
  }'
```

## Delete memory

```bash
curl -X DELETE http://localhost:8790/v1/memories/{memory_id}/
```

## Python client

Prefer Python over raw HTTP? `neatmem.MemoryClient` wraps these endpoints —
see the [Python Client reference](client.md).
