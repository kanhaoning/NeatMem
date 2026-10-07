# Changelog

## 0.8.0 — 2026-10-07

### Added

- **Memory usage feedback** (opt-in, `MEMORY_FEEDBACK_ENABLED`, default off). The server records what was actually injected and, offline, judges whether each injected memory was used by the answer — feeding per-memory counters (`inject_count` / `used_count`) in a new `activity.db`. Clients report injections via a `preceded_by_injection` field piggybacked on `/v1/messages/add/` messages (stripped before storage, never persisted as a message); search results are recorded as `search` events including hits that were never injected. Judgment runs offline via `neatmem feedback judge` (never on the serving hot path); `neatmem feedback status / evict / restore` inspect and manage the projection. Judge model/base-url/key/prompt are overridable via the `MEMORY_FEEDBACK_JUDGE_*` envs (default: follow the main LLM). Both flags off = byte-identical serving behavior.

- **Eviction gate** (opt-in, `MEMORY_FEEDBACK_EVICTION_ENABLED`, default off). Memories with `inject_count ≥ MEMORY_FEEDBACK_EVICTION_MIN_INJECTIONS` (default 10) and `used_count = 0` are marked `evicted` and excluded from recall at the candidate-pool stage with backfill (the pool stays full — eviction never shrinks result counts). Manual overrides (`neatmem feedback evict`) survive projection rebuilds via the `derived` / `manual` / `none` state split; `restore` clears either. LOCOMO 5-run gate: no causal harm (1/235 affected questions), net effect within the noise floor.

- **claude-code plugin: injection reporting for the feedback loop**. Prompt-search injections attach the actually-injected memory id set (after `unseen` dedup) to the recorded prompt event; mid-task reminder injections record a dedicated `injection` event. At upload, the first message after each injection point carries `preceded_by_injection` — the prompt message itself for prompt-search, the next assistant message for mid-task. Fail-open throughout: unmatched anchors are dropped, and older servers ignore the field. Plugin version 0.5.0 → 0.6.0.

## 0.7.1 — 2026-10-05

### Added

- **claude-code plugin: mid-task reminder injection** (opt-in, `NEATMEM_CODE_MIDTASK_REMINDER_ENABLED`, default off). During long tool runs (no user messages), the plugin estimates context growth from transcript bytes and, every `NEATMEM_CODE_MIDTASK_REMINDER_TOKENS` estimated tokens (default 8192), searches with the latest assistant plan text plus the current tool name as the query and injects up to `NEATMEM_CODE_MIDTASK_REMINDER_TOP_K` new memories (default 3) via PostToolUse `additionalContext` — giving mid-task steps access to relevant memories that only surface after the task started. The watermark resets on compact/clear, counts the injection itself toward the next window (bounded self-excitation), skips the search when the model produced no plan text in the window, and shares the plugin's fail-open path on any error. `unseen` dedup applies, so a memory is never injected twice in one session. The recent-memory delay applies on this channel too, so memories the current session produced moments ago cannot echo back as fresh injections (a delay-suppressed operation is logged when it filters).

- **claude-code plugin: recall notice** (default on; plugin option `recall_banner` or `NEATMEM_CODE_RECALL_BANNER=0` to disable). Injected memories travel via `additionalContext`, which Claude Code never renders in the interface — the user could not tell whether a recall happened. The plugin now also emits a one-line `systemMessage` (the only hook channel Claude Code renders inline in the transcript) whenever memories are injected, at both injection points: prompt search and watermark-triggered mid-task search (the latter tagged `for your current step`). Leading date prefixes are stripped from titles (our memories usually lead with a date), up to five titles are shown one per line (with a `+N more` marker beyond that), and truncation respects word boundaries (hard cut for CJK); the notice is purely additive and shares the injection's fail-open path.

### Fixed

- **Version reporting**: `neatmem.__version__` (and `server_info`) reported 0.6.4 in the 0.7.0 wheel; now correctly reports 0.7.1.

## 0.7.0 — 2026-10-03

### Added

- **Server-side query rewrite + expansion** (opt-in, `QUERY_REWRITE_ENABLED`, default off). Short context-dependent queries ("what about the second one?") are rewritten against the session's recent turns — located by a new top-level `run_id` on `/v2/memories/search/`, read from the server's own message store — and expanded into up to `QUERY_REWRITE_MAX_EXPANSIONS` alternative phrasings; the original, rephrased, and expanded queries are searched concurrently and their hits merged, so recall never drops below the verbatim baseline. Self-contained queries pass through untouched. Every failure (timeout, rate limit, unparseable output, rewrite-model down) fails open to the original query, and every call is logged as JSONL via `QUERY_REWRITE_LOG` for audit. The rewrite model defaults to `LLM_MODEL` and can be pointed elsewhere with `QUERY_REWRITE_MODEL` / `QUERY_REWRITE_BASE_URL` / `QUERY_REWRITE_API_KEY` / `QUERY_REWRITE_THINKING`; `QUERY_REWRITE_TIMEOUT` (default 3s) bounds the added latency and `QUERY_REWRITE_RETRIES` (default 0 — production behavior unchanged) opts into transient-failure retries for evaluation runs. The server advertises the feature in the `GET /v2/config/` client policy; the claude-code plugin honors it by still sending short prompts when rewrite is on (the server decides per prompt) and attaches `run_id` to search requests — both are ignored by older servers. LOCOMO 5-run gate on the production build: mean 90.17 vs anchor 90.58, all runs inside the anchor's observed range, rewrite fallback 0.51% — no regression.

### Fixed

- **Evaluation harness**: the LOCOMO answer path now retries 529/overloaded and connection errors in addition to 429/timeouts (aligned with the judge path), so transient upstream waves no longer kill multi-hour gate runs. A retried success is equivalent to a first-try success; scores are unaffected.

## 0.6.4 — 2026-10-01

### Added

- **claude-code plugin: per-turn forwarding** (opt-in, server policy `PER_TURN_FORWARD`, default off). When enabled, the Stop hook POSTs each turn's new messages to `/v1/messages/add/` as they happen (store-only; 2s budget, single attempt) instead of waiting for the next flush boundary — aligning with the hermes/openclaw clients and removing the up-to-session-end memory ingestion delay. Events uploaded this way are stamped `forwarded_at` locally so boundary flushes skip them (the local SQLite queue remains the fallback on any send failure), and SessionEnd/PreCompact still force-flush the server queue so an under-batch tail is extracted. Server-side extraction timing is unchanged (batch size/deadline). Emergency off-ramp: `NEATMEM_CODE_PER_TURN_FORWARD=0` overrides the server policy. Known interplay: with the new idempotent ingest below, several byte-identical assistant messages within one session collapse into one stored row.

### Removed

- **claude-code plugin: multi-harness machinery dropped** (mem0-fork residue). The plugin only ever served claude-code, so `configure_harness`/`harness_config`, the four `NEATMEM_PLUGIN_*` identity variables passed to the detached flush worker, and the `--harness` CLI flag are gone; the data-dir env fallback `NEATMEM_PLUGIN_DATA_DIR` is no longer read (use `NEATMEM_CODE_DATA_DIR`). Hook and flush-worker behavior is unchanged.

### Fixed

- **Idempotent message ingest**: `POST /v1/messages/add/` no longer stores duplicates when clients re-send identical messages (hook retries, pending-handoff replays, or two plugin versions mounted on the same session). `message_id` is now a deterministic SHA-256 over `app_id|user_id|agent_id|run_id|role|content` (`event_at` excluded — client-clock data), so the existing `UNIQUE` constraint actually fires; the response marks each entry `deduped: true/false` and adds a top-level `deduped_count`. Caveat: two byte-identical messages with the same role inside one run now collapse into one row — accepted trade-off for coding sessions. Cross-version formatting differences (e.g. a label prefix added by a newer plugin) are not caught; that remains the separate approximate-dedup topic.

## 0.6.3 — 2026-09-27

### Fixed

- **Poison-batch cursor stall**: a message batch whose whole-batch dedup query exceeded the embedding model's token limit (e.g. a 16KB compact summary) deterministically failed with a 400 and was retried forever, silently blocking the scope's cursor for days with zero extraction and no alert. Three-layer fix: (1) the dedup recall query is now truncated to `EMBEDDING_MAX_TOKENS` characters (extraction input is unaffected; under-size queries are byte-identical to before); (2) `EMBEDDING_MAX_TOKENS` resolves from an explicit env override, a built-in model-name table (bge-m3→8192, bge-large/base/small→512, OpenAI text-embedding-3/ada-002→8191, qwen text-embedding-v3/v4→8192, Qwen3-Embedding→32768, …), or a conservative 512 fallback with a warning for unknown models; (3) the batch scheduler now counts consecutive per-scope failures and skips a batch after `MESSAGE_BATCH_MAX_CONSECUTIVE_FAILURES` (default 10), advancing the cursor with an error log instead of retrying forever — skipped messages stay in the messages table and can be replayed by resetting the cursor.

## 0.6.2 — 2026-09-23

### Changed

- **Auto-injection defaults are now `INJECT_TIMING=every` and `MIN_QUERY_CHARS=5`** (were `first` and `20`), matching the configuration long-term deployments actually run. `first` had a structural flaw: when the session's first prompt was shorter than `MIN_QUERY_CHARS`, the whole session never searched. Deployments overriding these via env are unaffected. The claude-code plugin's fallback policy (used when the server is unreachable) is updated to the same defaults, and the docs' "first message triggers a search" wording now reads "every prompt".
- **Prompt-search HTTP timeout raised to 5s** (was hardcoded 2s) in the claude-code plugin — server-side LLM rerank plus cold starts could exceed 2s and silently return nothing. Override with `NEATMEM_CODE_SEARCH_TIMEOUT`.

## 0.6.1 — 2026-09-22

### Added

- **Recent-memory delay for automatic memory injection** (claude-code plugin): memories produced by the *current* session after the last compact and younger than `RECENT_MEMORY_DELAY_SECONDS` (default 1800, `0` disables) are excluded from automatic prompt injection, so a session does not immediately re-ingest its own just-written memories. Explicit search is unaffected. The server only serves the value via the `GET /v1/config/` client policy; enforcement lives in the plugin hook, which records a `delay-suppressed` operation when it filters.
- **Client-supplied event time (`event_at`) end to end**: clients may attach `event_at` per message on upload; the server stores it (new `messages.event_at` column, auto-migrated on existing databases; values >60s in the future are rejected to NULL) and stamps each extracted batch's memories with `metadata["timestamp"] = min(event_at)` (falling back to server receipt time when absent).

### Fixed

- **Search results now surface the memory event timestamp** in `metadata["timestamp"]`: mem0 flattens metadata into the payload top level, so `_format_candidate` always returned an empty metadata dict and evaluation answer prompts never received the per-memory date (the date-sorting/annotation logic was running on empty strings). This changes eval answer prompts, so the published LoCoMo anchor (90.75%) was re-measured before merging: baseline 90.43% → 90.51% with the fix (5 runs each, reference config; within the ±1-point gate).

## 0.6.0 — 2026-09-18

### Breaking

- **`HISTORY_DB_PATH` renamed to `MESSAGES_DB_PATH`**: the old name implied memory-change history, but the database actually stores raw chat messages (the memory-change history lives in `MEMORY_HISTORY_DB_PATH`). The old env var is no longer read — deployments setting `HISTORY_DB_PATH` must rename it. The CLI flag `--history-db-path` still works as a deprecated alias for the new `--messages-db-path` and prints a warning.

## 0.5.8 — 2026-09-10

### Changed

- **Remaining runtime logs translated to English**: batch scheduler, dedup pipeline (listwise/pointwise judgment, group merge, replace), extraction prefix, and the vector-store config summary — the log lines that 0.5.7 missed.

## 0.5.7 — 2026-09-09

### Changed

- **Runtime logs are now in English**: all logger/print text in the memory pipeline, server, and config summary (previously Chinese) matches the English-facing repo. The evaluate orchestrator's ingest log parsing (`writes: N`) was updated in the same change.

## 0.5.6 — 2026-09-09

### Added

- **`neatmem evaluate` shows stage progress on the console**: ingest/search/judge output is teed from the child processes — the log files still get the full content (unchanged), while the console shows a whitelist of progress lines (per-task completion, failures, judge `[done/total]` progress, summaries) instead of sitting silent for the whole ingest.

## 0.5.5 — 2026-09-08

### Fixed

- **`neatmem evaluate` preflight no longer imports spaCy**: the 0.5.4 lemmatization check now uses `importlib.util.find_spec` (presence-only), so machines with a broken torch/NumPy combo no longer get NumPy-ABI warning noise at startup.

## 0.5.4 — 2026-09-08

### Added

- **`neatmem evaluate` preflight warns when spaCy lemmatization is missing**: the published LoCoMo score was measured with `neatmem[nlp]` + `en_core_web_sm`; without it BM25 silently degrades to raw token matching, so the preflight now prints a warning (skipped when `ENABLE_BM25=false`).

## 0.5.3 — 2026-09-08

### Added

- **`neatmem evaluate` auto-downloads the qdrant server binary** on first run (pinned v1.17.0, ~29MB, cached under `~/.cache/neatmem/qdrant/`). When github.com is unreachable, `QDRANT_DOWNLOAD_BASE_URL` points the downloader at a mirror. Manual installation via `--qdrant-bin` / `QDRANT_BIN` / `PATH` still takes precedence.

## 0.5.2 — 2026-09-08

### Fixed

- **`neatmem evaluate --qdrant-bin` accepts relative paths**: the binary path is normalized to absolute at argument parsing, so `./qdrant` no longer breaks when the orchestrator spawns the server with a different working directory.
- **`neatmem evaluate` bridges `LLM_*` to `OPENAI_*`**: when only `LLM_API_KEY` (and `LLM_PROVIDER`) is configured, the answer/judge stages now inherit it automatically — the five `LLM_*`/`EMBEDDER_*` exports from the configuration reference are sufficient. The missing-key error message now points at the canonical `LLM_API_KEY` + `LLM_MODEL` setup.

### Changed

- Evaluation guide: prerequisite exports rewritten in `LLM_*` style (matching the server configuration docs), and the tested qdrant server version (v1.17.x, matching the pinned qdrant-client) is now stated.

## 0.5.1 — 2026-09-02

### Added

- **Group resolution for multi-target updates** (`DEDUP_DETECTOR=listwise_multitarget` + `DEDUP_RESOLVER=rewrite`): when one write is judged to update ≥2 existing memories, the resolver now fuses the new fact and all targets in a single merge call — the merged text is written to the highest-score target and the rest are deleted — instead of rewriting each target independently (which could leave near-duplicate memories). On a "No"/error answer it falls back to the per-target loop. Custom prompt via `REWRITE_GROUP_PROMPT` / `--rewrite-group-prompt` (default `rewrite_group_en.txt`).
- **Dedup detector raw-output logging**: every detector response is logged with `finish_reason` and the raw text before think-tag stripping (`DETECTOR RAW`) for debugging.

## 0.5.0 — 2026-08-30

### Added

- **`neatmem demo` command**: replay a small scenario through the memory pipeline and watch every decision — extraction results, dedup judgments with reasons, resolver actions, and the verbatim final store. Input is a case JSON file (path or `-` for stdin) or inline `--existing` / `--say` messages; `--reps N` repeats from a fresh in-memory store, `--output PATH` saves a markdown run record. Accepts the same serve flags as `neatmem serve`. See the demo guide.
- **`--dedup-recall-threshold` serve flag**: CLI passthrough for the existing `DEDUP_RECALL_THRESHOLD` env var (default 0.40), available on `serve` / `evaluate` / `demo`.
- **Multi-target dedup judgments now log the judge's `reason`** (truncated to 100 chars) alongside action and target.

### Fixed

- **Memories written verbatim (`infer=false`) now carry `attr_source` metadata** (default `user`, caller-supplied metadata wins), so they are visible to dedup candidate recall. Previously the recall filter excluded them entirely, so contradictory later facts were always stored as new duplicates instead of updating the verbatim memory.

## 0.4.0 — 2026-08-28

### Added

- **`listwise_multitarget` dedup detector**: same single LLM call per write as `listwise`, but judges each candidate independently, so one write can update several existing memories (`DEDUP_DETECTOR=listwise_multitarget` / `--dedup-detector listwise_multitarget`).
- **Dedup prompts are packaged as plain-text files** and the default is auto-paired from the `DEDUP_DETECTOR` + `DEDUP_RESOLVER` combination when `DEDUP_PROMPT` is unset. The resolved prompt file and its sha256 are logged at startup, and the evaluation manifest records them.

### Changed

- **Dedup defaults are now `DEDUP_DETECTOR=listwise_multitarget` + `DEDUP_RESOLVER=rewrite`** (were `listwise` + `skip`): detected duplicates are merged into the existing memory by default instead of coexisting with it.
- **`DEDUP_PROMPT` no longer accepts built-in ids** (`zh`/`en`) — pass a prompt file path, or leave it unset for auto-pairing. The built-in id mechanism is removed from all prompt env vars.

### Fixed

- `neatmem evaluate` no longer reports a resumed ingest as failed when the retry actually completes (the success check now reads the last `Successful: N / M` line in the appended ingest log).

## 0.3.0 — 2026-08-25

### Added

- **`neatmem evaluate` new flags**: `--project-name` (derives the output dir `runs/<name>`), `--answerer-model` / `--judge-model`, `--max-workers` umbrella concurrency (per-stage flag > `--max-workers` > built-in default), and `--predict-only` / `--evaluate-only` as shortcuts for stage subsets. See the evaluation guide.

### Changed

- **Embedding env family renamed to `EMBEDDER_*`**: `EMBEDDING_PROVIDER` / `EMBEDDING_MODEL` / `EMBEDDING_BASE_URL` / `EMBEDDING_API_KEY` / `EMBEDDING_DIMS`, the `GRAPH_EMBEDDING_*` group, and the `--embedding-*` serve flags. `SILICONFLOW_API_KEY` is still accepted as the key fallback.
- **`GRAPH_EMBEDDER_*` defaults now follow the main embedder config** instead of hardcoded SiliconFlow values; `GRAPH_EMBEDDER_API_KEY` now reads its own env.
- Built-in eval stage concurrency unified to 4 (was 20/16/8); raise it with `--max-workers` when your quota allows.

### Fixed

- `neatmem evaluate` judge stage no longer re-runs on every resume: the resume check now compares judged-QA totals (the judged file is keyed per judged task, category 5 excluded) instead of key counts.

## 0.2.0 — 2026-08-23

### Added

- **`neatmem evaluate`**: one-command LoCoMo benchmark pipeline (qdrant → ingest → search+answer → judge → score), resumable per stage. See the evaluation guide on the docs site.
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
- **`LLM_MODEL` no longer has a default.** The server now refuses to boot with an explicit message instead. Set `LLM_PROVIDER` + `LLM_API_KEY` + `LLM_MODEL` (see `.env.example`).
- **Local data paths now root at `NEATMEM_DIR`** (default `~/.neatmem`):
  - `QDRANT_PATH` default: `./qdrant_db` (cwd-relative) → `{NEATMEM_DIR}/qdrant`
  - `HISTORY_DB_PATH` default: `{QDRANT_PATH}/history.db` → `{NEATMEM_DIR}/messages.db`
  - `MEMORY_HISTORY_DB_PATH` default: `{MEM0_DIR}/history.db` → `{NEATMEM_DIR}/history.db`
  - Priority per path: dedicated env var > `NEATMEM_DIR`-derived default. `MEM0_DIR` is honored as a legacy fallback for the root.
  - **Migration**: existing deployments that relied on the cwd-relative `./qdrant_db` default should either set `QDRANT_PATH` (and `HISTORY_DB_PATH`) explicitly to their current locations, or move the data into `~/.neatmem/`.
- **Default Qdrant collection renamed** `mem0` → `neatmem` (entity collection auto-derives as `neatmem_entities`). Existing embedded DBs keep their data under the old collection name; to retain it, re-ingest or copy the points into a `neatmem` collection before upgrading.

### Added

- Multi-provider LLM support (10 providers) and embedding providers (3), with per-provider thinking control. See `docs/` for the provider matrix.
