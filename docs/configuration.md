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
| `DEDUP_MODE` | no | `skip` | Dedup behavior: `off`, `skip`, `replace`, `rewrite`, `edit` |
| `ENABLE_BM25` | no | `true` | Enable BM25 sparse search signal |
| `ENABLE_ENTITY` | no | `false` | Enable entity extraction and boosting |
| `ENABLE_GRAPH` | no | `false` | Enable graph memory (KuzuDB entity-relation storage). Graph hooks are no-op when disabled |
| `KUZU_DB_PATH` | conditional | - | KuzuDB database file path. Required when `ENABLE_GRAPH=true` |
| `GRAPH_THRESHOLD` | no | `0.7` | Entity match threshold for graph retrieval |
| `GRAPH_SEARCH_TOP_K` | no | `5` | Max relations returned per speaker from graph search |
| `GRAPH_INJECT_RELATIONS` | no | `false` | Inject graph relations into answer prompt. Only effective when `ENABLE_GRAPH=true`. Experimental: -0.57pp on LOCOMO (2026-07-22), off by default |
| `GRAPH_EMBEDDING_MODEL` | no | `BAAI/bge-m3` | Embedding model for graph entities (defaults to main embedding model) |
| `GRAPH_EMBEDDING_DIMS` | no | `1024` | Embedding dimensions for graph entities |
| `GRAPH_EMBEDDING_BASE_URL` | no | `https://api.siliconflow.cn/v1` | Embedding API base URL for graph entities |
| `GRAPH_EMBEDDING_API_KEY` | no | - | Embedding API key for graph entities. Defaults to `SILICONFLOW_API_KEY` |
| `LLM_RERANK` | no | `true` | Enable LLM listwise rerank for recall |
| `RERANK_MODE` | no | `llm_listwise` | Rerank strategy |
| `RERANK_CANDS` | no | `20` | Head size for LLM listwise rerank: only top N candidates are reordered, the rest are appended in original order. Only effective when `LLM_RERANK=true` |
| `RERANK_MAX_CONCURRENT` | no | `4` | Max concurrent LLM rerank calls (protects against API rate limits) |
| `MERGE_STRATEGY` | no | `off` | Deprecated; use `DEDUP_MODE` instead |
| `DEDUP_THINKING` | no | `false` | Enable LLM thinking for dedup |
| `EDIT_THINKING` | no | `false` | Enable LLM thinking for edit mode (DEDUP_MODE=edit) |
| `HISTORY_DB_PATH` | no | `{NEATMEM_DIR}/messages.db` | SQLite message history database path |
| `MEMORY_HISTORY_DB_PATH` | no | `{NEATMEM_DIR}/history.db` | SQLite memory-change history (ADD/UPDATE/DELETE events) database path |
| `EXTRACT_LAST_K_MESSAGES` | no | `10` | Number of recent messages fed to extraction as context |
| `MESSAGE_STORE_BACKEND` | no | `sqlite` | Message store backend: `sqlite` or `none` |
| `ENTITY_EXTRACTOR_BACKEND` | no | `ner` | Entity extractor: `ner` or `llm` |
| `ENTITY_STORE_BACKEND` | no | `qdrant` | Entity store backend |
| `RERANKER_MODEL_PATH` | no | - | Optional local Sentence-Transformers reranker |
| `RERANKER_DEVICE` | no | `cpu` | Reranker device |
| `RERANKER_BATCH_SIZE` | no | `32` | Reranker batch size |
| `RERANKER_TOP_K` | no | `5` | Reranker top-k |
| `HF_ENDPOINT` | no | `https://hf-mirror.com` | HuggingFace mirror endpoint |
