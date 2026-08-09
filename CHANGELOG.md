# Changelog

## Unreleased

### Changed

- **Local data paths now root at `NEATMEM_DIR`** (default `~/.neatmem`):
  - `QDRANT_PATH` default: `./qdrant_db` (cwd-relative) → `{NEATMEM_DIR}/qdrant`
  - `HISTORY_DB_PATH` default: `{QDRANT_PATH}/history.db` → `{NEATMEM_DIR}/messages.db`
  - `MEMORY_HISTORY_DB_PATH` default: `{MEM0_DIR}/history.db` → `{NEATMEM_DIR}/history.db`
  - Priority per path: dedicated env var > `NEATMEM_DIR`-derived default. `MEM0_DIR` is honored as a legacy fallback for the root.
  - **Migration**: existing deployments that relied on the cwd-relative `./qdrant_db` default should either set `QDRANT_PATH` (and `HISTORY_DB_PATH`) explicitly to their current locations, or move the data into `~/.neatmem/`.
- **Default Qdrant collection renamed** `mem0` → `neatmem` (entity collection auto-derives as `neatmem_entities`). Existing embedded DBs keep their data under the old collection name; run `python scripts/migrate_mem0_to_neatmem.py <qdrant_path>` while the server is stopped to copy points into the new collections.

### Added

- Multi-provider LLM support (10 providers) and embedding providers (3), with per-provider thinking control. See `docs/` for the provider matrix.
