# Configuration

NeatMem reads configuration from `.env`.

| Variable | Required | Default | Description |
|---|---:|---|---|
| `NEATMEM_HOST` | no | `0.0.0.0` | Server bind host |
| `NEATMEM_PORT` | no | `8790` | Server port |
| `NEATMEM_URL` | no | `http://localhost:8790` | Base URL used by `MemoryClient` |
| `NEATMEM_API_KEY` | no | - | API key sent as `Authorization: Token` header by `MemoryClient` (server ignores it) |
| `LLM_PROVIDER` | no | - | LLM provider preset: `deepseek`, `dashscope`, `zhipu`, `moonshot`, `volcengine`, `minimax`, `siliconflow`, `openai`, `gemini`, `openrouter` (aliases: `qwen`, `glm`, `kimi`, `doubao`). Supplies the default base URL and verified thinking-control parameters |
| `LLM_API_KEY` | yes | - | LLM API key (`OPENAI_API_KEY` accepted as fallback) |
| `OPENAI_BASE_URL` | no | provider preset | Explicit LLM base URL override (beats the `LLM_PROVIDER` preset) |
| `LLM_MODEL` | yes | - | LLM model name (no default; server refuses to boot without it) |
| `EMBEDDING_PROVIDER` | no | `siliconflow` | `siliconflow`, `openai`, `dashscope`, or `xinference` |
| `EMBEDDING_API_KEY` | conditional | - | Embedding API key (`SILICONFLOW_API_KEY` accepted as fallback). Required for hosted embedding providers |
| `SILICONFLOW_API_KEY` | conditional | - | Legacy name for `EMBEDDING_API_KEY` when `EMBEDDING_PROVIDER=siliconflow` |
| `EMBEDDING_MODEL` | no | `BAAI/bge-m3` | Embedding model name |
| `EMBEDDING_BASE_URL` | no | `https://api.siliconflow.cn/v1` | Embedding API base URL |
| `EMBEDDING_DIMS` | no | auto-detect | Embedding dimensions. When unset, auto-detected from a startup probe; set explicitly to enforce a dimension check at boot |
| `XINFERENCE_SERVER_URL` | conditional | `http://localhost:9997` | Required when using Xinference |
| `XINFERENCE_MODEL_UID` | conditional | `bge-m3` | Xinference embedding model UID |
| `NEATMEM_DIR` | no | `~/.neatmem` | Data root directory: default parent for all local data below. `MEM0_DIR` is honored as a legacy fallback |
| `QDRANT_PATH` | no | `{NEATMEM_DIR}/qdrant` | Local Qdrant storage path (embedded mode) |
| `QDRANT_HOST` | no | - | Qdrant server host (sets server mode; overrides `QDRANT_PATH`) |
| `QDRANT_PORT` | no | `6333` | Qdrant server port |
| `DEDUP_ENABLED` | no | `true` | Enable dedup on write (`false` = every memory is stored as new) |
| `DEDUP_RESOLVER` | no | `skip` | How to resolve a detected duplicate: `skip` (keep both), `replace` (overwrite old), `rewrite` (LLM merge), `edit` (LLM patch) |
| `DEDUP_DETECTOR` | no | `listwise` | How duplicates are detected: `listwise` (one LLM call judges the whole candidate batch) or `pointwise` (per-pair three-way classification) |
| `ENABLE_BM25` | no | `true` | Enable BM25 sparse search signal |
| `ENABLE_ENTITY` | no | `false` | Enable entity extraction and boosting |
| `ENABLE_GRAPH` | no | `false` | Enable graph memory (KuzuDB entity-relation storage). Graph hooks are no-op when disabled |
| `KUZU_DB_PATH` | conditional | - | KuzuDB database file path. Required when `ENABLE_GRAPH=true` |
| `GRAPH_THRESHOLD` | no | `0.7` | Entity match threshold for graph retrieval |
| `GRAPH_SEARCH_TOP_K` | no | `5` | Max relations returned per speaker from graph search |
| `GRAPH_EMBEDDING_MODEL` | no | `BAAI/bge-m3` | Embedding model for graph entities (defaults to main embedding model) |
| `GRAPH_EMBEDDING_DIMS` | no | `1024` | Embedding dimensions for graph entities |
| `GRAPH_EMBEDDING_BASE_URL` | no | `https://api.siliconflow.cn/v1` | Embedding API base URL for graph entities |
| `GRAPH_EMBEDDING_API_KEY` | no | - | Embedding API key for graph entities. Defaults to `SILICONFLOW_API_KEY` |
| `RERANK_MODE` | no | `llm` | Rerank engine: `llm`, `cross_encoder`, `off` |
| `LLM_RERANK_MODE` | no | `listwise` | LLM rerank mode: `listwise` (filter + rank in one call), `pointwise` (per-doc scoring). Only effective when `RERANK_MODE=llm` |
| `LLM_RERANK_CANDS` | no | `20` | Only the top N candidates are rescored; the rest are appended in original order |
| `LLM_RERANK_CAND_TEXT_LEN` | no | `120` | Candidate text truncation before sending to the LLM; `0` = no truncation |
| `LLM_RERANK_PROMPT` | no | - | Custom rerank prompt (built-in id or file path), applies to the active `LLM_RERANK_MODE` |
| `CROSS_ENCODER_PROVIDER` | no | `siliconflow` | Cross-encoder provider: `siliconflow` or `local`; `local` needs `pip install neatmem[local-reranker]` |
| `CROSS_ENCODER_MODE` | no | `pointwise` | Scoring mode. Only `pointwise` (per-doc scores) is implemented; selecting `listwise` fails at startup |
| `CROSS_ENCODER_MODEL` | no | preset | Scoring model (default follows the provider preset, e.g. `Qwen/Qwen3-Reranker-8B` on siliconflow) |
| `CROSS_ENCODER_BASE_URL` | no | preset | API base URL override |
| `CROSS_ENCODER_API_KEY` | no | - | API key; falls back to the provider's own key env (e.g. `SILICONFLOW_API_KEY`) |
| `CROSS_ENCODER_CANDS` | no | `100` | Only the top N candidates are rescored by the cross-encoder |
| `CROSS_ENCODER_CAND_TEXT_LEN` | no | `0` | Candidate truncation before scoring; `0` = no truncation (the model's own length limit applies) |
| `CROSS_ENCODER_REL_THRESHOLD` | no | `0` | `0` = sort + truncate only; >0 drops docs below ratio × top score |
| `CROSS_ENCODER_TIMEOUT` | no | `60` | HTTP timeout seconds (API providers) |
| `CROSS_ENCODER_DEVICE` | no | `auto` | `local` provider only: `auto`/`cuda`/`cpu` |
| `CROSS_ENCODER_BATCH_SIZE` | no | `32` | `local` provider only: scoring batch size |
| `RERANK_MAX_CONCURRENT` | no | `12` | Max concurrent rerank calls (protects against API rate limits) |
| `DEDUP_THINKING` | no | `false` | Enable LLM thinking for dedup |
| `EDIT_THINKING` | no | `false` | Enable LLM thinking for edit mode (`DEDUP_RESOLVER=edit`) |
| `HISTORY_DB_PATH` | no | `{NEATMEM_DIR}/messages.db` | SQLite message history database path |
| `MEMORY_HISTORY_DB_PATH` | no | `{NEATMEM_DIR}/history.db` | SQLite memory-change history (ADD/UPDATE/DELETE events) database path |
| `EXTRACT_LAST_K_MESSAGES` | no | `10` | Number of recent messages fed to extraction as context |
| `MESSAGE_STORE_BACKEND` | no | `sqlite` | Message store backend: `sqlite` or `none` |
| `MESSAGE_BATCHING_ENABLED` | no | `true` | Server-side write batching: messages forwarded to `/v1/messages/add/` are extracted in fixed-size batches by the in-process scheduler |
| `MESSAGE_BATCH_SIZE` | no | `10` | Messages per extraction batch |
| `MESSAGE_BATCH_DEADLINE_SECS` | no | `600` | Force a partial batch once the oldest pending message is older than this |
| `MESSAGE_BATCHING_CHECK_INTERVAL_SECS` | no | `30` | Batch scheduler check interval |
| `DEDUP_RECALL_THRESHOLD` | no | `0.40` | Vector similarity threshold for dedup candidate recall |
| `ENTITY_EXTRACTOR_BACKEND` | no | `ner` | Entity extractor: `ner` or `llm` |
| `ENTITY_STORE_BACKEND` | no | `qdrant` | Entity store backend |
| `HF_ENDPOINT` | no | `https://hf-mirror.com` | HuggingFace mirror endpoint |
