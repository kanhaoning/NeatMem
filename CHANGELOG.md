# Changelog

## Unreleased

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
