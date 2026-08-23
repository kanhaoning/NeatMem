# Python Client

Reference for the `MemoryClient` class in the `neatmem` package — a remote
client for a running NeatMem server (`neatmem serve`). Method signatures and
return shapes match mem0's `MemoryClient`.

## Initialization

```python
from neatmem import MemoryClient

client = MemoryClient(host="http://localhost:8790")
```

Constructor: `MemoryClient(api_key=None, host=None, org_id=None, project_id=None, timeout=300.0)`.
If `host` is not provided, reads the `NEATMEM_URL` environment variable,
then falls back to `http://localhost:8790`. If `api_key` is provided (or
the `NEATMEM_API_KEY` env is set), it is sent as an `Authorization: Token`
header; the server currently ignores it.

Passing `org_id` or `project_id` (mem0 Platform parameters) raises
`NotImplementedError`.

---

## Memory methods

### add(messages, ...)

Extract and store memories from messages.

```python
client.add(
    [
        {"role": "user", "content": "I'm a vegetarian and allergic to nuts."},
        {"role": "assistant", "content": "Got it! I'll remember that."},
    ],
    user_id="alice",
)

# Raw text — skip LLM inference
client.add("User prefers dark mode.", user_id="alice", infer=False)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `messages` | str \| dict \| list[dict] | required | Message content; strings auto-convert to a user message |
| `user_id` | str | None | User identifier |
| `agent_id` | str | None | Agent identifier |
| `run_id` | str | None | Session/run identifier |
| `metadata` | dict | None | Custom key-value pairs |
| `infer` | bool | True | If False, store raw text without LLM inference |

At least one of `user_id` / `agent_id` / `run_id` is required (client-side
check). `timestamp` and `memory_type` raise `NotImplementedError`.

**Returns:** `dict` — `{"results": [...], "duplicates": [...], "merged": [...]}`;
each result item carries `id`, `memory`, `metadata`, `score`, `created_at`,
and an optional `event` (`ADD`/`UPDATE`/`DELETE`/`NONE`)

### search(query, ...)

Search memories by semantic similarity.

```python
results = client.search("dietary preferences", filters={"user_id": "alice"})
for mem in results.get("results", []):
    print(mem["memory"], mem["score"])
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `query` | str | required | Natural language search query |
| `filters` | dict | required | Must contain at least one of `user_id` / `agent_id` / `run_id` |
| `top_k` | int | 20 | Number of results |
| `threshold` | float | 0.1 | Minimum similarity score |
| `rerank` | bool | False | Apply the rerank engine configured on the server ([`RERANK_MODE`](configuration.md)) |

Entity ids go inside `filters`; top-level `user_id` / `agent_id` / `run_id` /
`app_id` kwargs are rejected. `explain=True` raises `NotImplementedError`.

**Returns:** `dict` — `{"results": [{"id", "memory", "score", "user_id", ...}]}`

### get(memory_id)

Retrieve a single memory by ID.

**Returns:** `dict` — the memory object.

### get_all(...)

List memories.

```python
client.get_all(filters={"user_id": "alice"}, top_k=50)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `filters` | dict | required | Must contain at least one of `user_id` / `agent_id` / `run_id` |
| `top_k` | int | 20 | Page size |

Entity ids go inside `filters`; top-level entity kwargs are rejected, same as
`search()`.

**Returns:** `dict` — `{"results": [...], "page": 1, "page_size": N}`

### update(memory_id, data, metadata=None)

Update a memory's text, and optionally its metadata.

**Returns:** `dict` — `{"message": "Memory updated successfully!"}`

### delete(memory_id)

Delete a memory by ID.

**Returns:** `dict` — `{"message": "Memory deleted successfully!"}`

### delete_all(...)

Delete all memories belonging to a scope.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `user_id` | str | None | User identifier |
| `agent_id` | str | None | Agent identifier |
| `run_id` | str | None | Session/run identifier |

At least one of the three is required. To delete everything on the server,
use `reset()`.

### history(memory_id)

Get the change history (ADD/UPDATE/DELETE events) of a memory.

**Returns:** `list` — bare list of history entries.

### reset()

Delete ALL memories on the server. Raw message history is reset separately
via `client.messages.reset()`.

---

## Additional methods

### Message batching

Store-only ingest plus server-side batch extraction (see
[Message batching](api.md#message-batching-queue-mode)):

```python
client.add_messages([{"role": "user", "content": "..."}], user_id="alice")
client.flush_messages(user_id="alice")  # extract everything pending now
```

| Method | Description |
|---|---|
| `add_messages(messages, user_id=..., agent_id=None, run_id=None, app_id=None)` | Store messages without extraction |
| `get_next_batch(user_id=..., agent_id=None, run_id=None, store="vector")` | Fetch the next batch due for extraction (message ids/seqs only, no content) |
| `mark_batch_processed(user_id=..., last_processed_seq=..., agent_id=None, run_id=None, store="vector")` | Advance the extraction cursor; call only after the batch was extracted |
| `flush_messages(user_id=..., agent_id=None, run_id=None, store="vector")` | Extract all pending messages for the scope synchronously |

`store` selects the storage backend; leave it at the default `vector`.

With the default [`MESSAGE_BATCHING_ENABLED`](configuration.md)`=true`, the
server schedules extraction itself — most users only need `add_messages()`
and `flush_messages()`.

### Raw message history

Stored raw messages (before extraction) are accessible under
`client.messages` (see [Message history](api.md#message-history)):

| Method | Description |
|---|---|
| `query(user_id=None, agent_id=None, run_id=None, app_id=None, roles=None, content_like=None, after=None, before=None, limit=100, offset=0, order="desc")` | Query stored messages; the server requires at least one scope filter |
| `sessions(user_id=None, agent_id=None, app_id=None, limit=100, offset=0)` | List sessions that have stored messages |
| `delete(user_id=None, agent_id=None, run_id=None, app_id=None)` | Delete stored messages matching at least one filter; does not touch extracted memories |
| `reset()` | Delete ALL stored messages and reset extraction cursors; does not touch extracted memories |

### Server utilities

| Method | Description |
|---|---|
| `ping()` | Health check |
| `delete_entity(entity_type, entity_id)` | Delete all memories of a user/agent/app/run, e.g. `delete_entity("user", "alice")` |
