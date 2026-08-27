# Changelog

## 0.4.0 — 2026-08-28

### Added

- **`listwise_multitarget` dedup detector**: same single LLM call per write as `listwise`, but judges each candidate independently, so one write can update several existing memories (`DEDUP_DETECTOR=listwise_multitarget` / `--dedup-detector listwise_multitarget`).
- **Dedup prompts are packaged as plain-text files** and the default is auto-paired from the `DEDUP_DETECTOR` + `DEDUP_RESOLVER` combination when `DEDUP_PROMPT` is unset. The resolved prompt file and its sha256 are logged at startup, and the evaluation manifest records them.

### Changed

- **Dedup defaults are now `DEDUP_DETECTOR=listwise_multitarget` + `DEDUP_RESOLVER=rewrite`** (were `listwise` + `skip`): detected duplicates are merged into the existing memory by default instead of coexisting with it.
- **`DEDUP_PROMPT` no longer accepts built-in ids** (`zh`/`en`) — pass a prompt file path, or leave it unset for auto-pairing (hard cutover, no compat layer). The built-in id mechanism is removed from all prompt env vars.

### Fixed

- `neatmem evaluate` no longer reports a resumed ingest as failed when the retry actually completes (the success check now reads the last `Successful: N / M` line in the appended ingest log).

## 0.3.0 — 2026-08-25

### Added

- **`neatmem evaluate` new flags**: `--project-name` (derives the output dir `runs/<name>`), `--answerer-model` / `--judge-model`, `--max-workers` umbrella concurrency (per-stage flag > `--max-workers` > built-in default), and `--predict-only` / `--evaluate-only` as shortcuts for stage subsets. See the evaluation guide.

### Changed

- **Embedding env family renamed to `EMBEDDER_*`** (hard cutover, no compat layer): `EMBEDDING_PROVIDER` / `EMBEDDING_MODEL` / `EMBEDDING_BASE_URL` / `EMBEDDING_API_KEY` / `EMBEDDING_DIMS`, the `GRAPH_EMBEDDING_*` group, and the `--embedding-*` serve flags. `SILICONFLOW_API_KEY` is still accepted as the key fallback.
- **`GRAPH_EMBEDDER_*` defaults now follow the main embedder config** instead of hardcoded SiliconFlow values; `GRAPH_EMBEDDER_API_KEY` actually reads its own env (it previously never did).
- Built-in eval stage concurrency unified to 4 (was 20/16/8); raise it with `--max-workers` when your quota allows.

### Fixed

- `neatmem evaluate` judge stage no longer re-runs on every resume: the resume check now compares judged-QA totals (the judged file is keyed per judged task, category 5 excluded) instead of key counts.

## 0.2.0 — 2026-08-23

### Added

- **`neatmem evaluate`**: one-command LOCOMO benchmark pipeline (qdrant → ingest → search+answer → judge → score), resumable per stage. See the evaluation guide on the docs site.
- **Server-side write batching (queue mode)**: the `/v1/messages/` endpoint family lets clients forward raw messages as they happen; the server extracts memories in fixed-size batches (`MESSAGE_BATCHING_*` settings), with a flush endpoint to force extraction at session boundaries.
- **Cross-encoder rerank engine**: `RERANK_MODE=llm|cross_encoder|off` selects the engine. The cross-encoder runs via a hosted SiliconFlow preset or locally through sentence-transformers (`pip install "neatmem[local-reranker]"`).
- **Pointwise LLM rerank mode**: `LLM_RERANK_MODE=listwise|pointwise`.
- **Pointwise dedup detector**: `DEDUP_DETECTOR=listwise|pointwise`.
- `MemoryClient` coverage for message batching and raw message history (`client.messages`), with a full Python Client reference on the docs site.
- OpenClaw plugin v2.0.0 (queue-mode write path); Hermes plugin flushes pending messages at session boundaries.

### Changed

- **Rerank configuration redesigned into two self-contained groups**: `LLM_RERANK_*` and `CROSS_ENCODER_*`. Removed environment variables: `LLM_RERANK` (bool), the old `RERANK_MODE` values `llm_listwise`/`llm_listwise_v2`, `RERANK_PROMPT`, `RERANK_CANDS`, `RERANK_CAND_TEXT_LEN`. The per-request `rerank` boolean now follows the configured engine. See the configuration reference.
- **Dedup configuration is now three axes** — `DEDUP_ENABLED` / `DEDUP_RESOLVER` (`skip`/`replace`/`rewrite`/`edit`) / `DEDUP_DETECTOR` — replacing the previous single mode variable.
- `RERANK_MAX_CONCURRENT` default 4 → 12.
- Default rewrite/edit prompts are packaged as plain-text files and can be replaced file-by-file (see the custom prompts guide).

### Fixed

- Answer-stage API timeouts during evaluation are retried with backoff instead of crashing the run.
- With rerank enabled, dense recall now widens to the reranker head size, so rerank candidates actually reach the reranker.

### Removed

- `neatmem evaluate --config` and the bundled strategy `.env` files (strategy is selected with serve flags instead).

## 0.1.1 — 2026-08-11

### Changed

- **spaCy is now optional.** Bare `pip install neatmem` boots and serves: without spaCy, BM25 keyword search falls back to raw-token matching (no lemmatization) with a startup warning; if the fastembed encoder is unavailable (not installed or model download fails), retrieval degrades to dense-only with a warning instead of failing requests. Install the `nlp` extra for full BM25 lemmatization.
- **`LLM_MODEL` no longer has a default.** The previous `qwen-max-latest` fallback was incoherent with the default base URL and would only ever produce a confusing API error; the server now refuses to boot with an explicit message instead. Set `LLM_PROVIDER` + `LLM_API_KEY` + `LLM_MODEL` (see `.env.example`).
- **Local data paths now root at `NEATMEM_DIR`** (default `~/.neatmem`):
  - `QDRANT_PATH` default: `./qdrant_db` (cwd-relative) → `{NEATMEM_DIR}/qdrant`
  - `HISTORY_DB_PATH` default: `{QDRANT_PATH}/history.db` → `{NEATMEM_DIR}/messages.db`
  - `MEMORY_HISTORY_DB_PATH` default: `{MEM0_DIR}/history.db` → `{NEATMEM_DIR}/history.db`
  - Priority per path: dedicated env var > `NEATMEM_DIR`-derived default. `MEM0_DIR` is honored as a legacy fallback for the root.
  - **Migration**: existing deployments that relied on the cwd-relative `./qdrant_db` default should either set `QDRANT_PATH` (and `HISTORY_DB_PATH`) explicitly to their current locations, or move the data into `~/.neatmem/`.
- **Default Qdrant collection renamed** `mem0` → `neatmem` (entity collection auto-derives as `neatmem_entities`). Existing embedded DBs keep their data under the old collection name; to retain it, re-ingest or copy the points into a `neatmem` collection before upgrading.

### Added

- Multi-provider LLM support (10 providers) and embedding providers (3), with per-provider thinking control. See `docs/` for the provider matrix.
