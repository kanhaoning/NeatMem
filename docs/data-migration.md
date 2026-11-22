# Data migration

NeatMem currently provides an offline export/import workflow for one embedded
Qdrant collection. Use this guide to move that collection to a separate Qdrant
server through a file, without calling an LLM or computing new embeddings.

The workflow is:

```text
Embedded Qdrant                 ZIP archive                 Qdrant server
on your machine     export →    points.jsonl    import →    on the same or
+ NeatMem stopped               + manifest.json               another machine
```

## Terms used in this guide

- **Embedded Qdrant**: Qdrant runs inside the NeatMem process and reads a local
  directory such as `~/.neatmem/qdrant`.
- **Qdrant server**: Qdrant runs as its own process, usually at
  `http://127.0.0.1:6333`.
- **Collection**: the Qdrant container holding points. NeatMem's main collection
  is normally named `neatmem`.
- **Archive**: the ZIP file produced by the export command. You can keep it on
  the same machine or copy it to another machine.

## What the archive contains—and what it does not

| Included | Not included |
|---|---|
| Point IDs | `messages.db` chat history |
| Payloads (metadata attached to points) | `history.db` memory-change history |
| Dense vectors | `activity.db` feedback activity |
| Sparse vectors, including `bm25` | NeatMem `.env` configuration |
| Collection configuration | Other collections unless exported separately |

The archive is a copy of **one Qdrant collection**, not a complete backup of a
NeatMem installation.

## Before you begin

1. **Stop NeatMem.** This prevents source changes and releases the embedded
   Qdrant storage lock.
2. **Back up your existing data.** The archive is not a replacement for a full
   backup or rollback plan.
3. **Start Qdrant server with empty, isolated storage.** The destination
   collection must not already exist.
4. **Use a server version compatible with your installed `qdrant-client`.**
   Keep vector precision and index settings unchanged during this migration.

Qdrant provides native binaries on its
[official releases page](https://github.com/qdrant/qdrant/releases); Docker is
not required. For example, start a downloaded binary with localhost-only access:

```bash
QDRANT__SERVICE__HOST=127.0.0.1 \
QDRANT__STORAGE__STORAGE_PATH=/path/to/qdrant/storage \
/path/to/qdrant
```

In another terminal, check that the server is ready:

```bash
curl -fsS http://127.0.0.1:6333/healthz
```

## Step 1: Export the embedded collection

```bash
neatmem migrate qdrant export \
  --source-path "$HOME/.neatmem/qdrant" \
  --collection neatmem \
  --output neatmem-qdrant-export.zip \
  --batch-size 100
```

Adjust `--source-path` if your `QDRANT_PATH` is different. `--batch-size`
controls how many points are read and written at a time.

The output file must not already exist. Successful output looks like:

```text
Exported 44613 points from collection 'neatmem' to neatmem-qdrant-export.zip
```

The ZIP contains two files:

- `points.jsonl`: one point per line, including its ID, metadata, and complete
  dense/sparse vectors;
- `manifest.json`: the collection name/configuration, point count, format
  version, and a SHA-256 checksum used to detect accidental corruption.

If export fails, a `.partial` file may remain. It is not a completed archive and
must not be imported.

## Step 2: Validate without writing to the server

```bash
neatmem migrate qdrant import \
  --input neatmem-qdrant-export.zip \
  --target-url http://127.0.0.1:6333 \
  --dry-run
```

Dry-run reads and validates every point and the checksum, connects to Qdrant,
and checks that the destination collection is absent. It does **not** create a
collection or write any points.

Successful output looks like:

```text
Archive validated: 44613 points for collection 'neatmem'; no data was written
```

## Step 3: Import the archive

```bash
neatmem migrate qdrant import \
  --input neatmem-qdrant-export.zip \
  --target-url http://127.0.0.1:6333 \
  --batch-size 100
```

The destination collection name comes from the archive. Import creates that
collection, writes points in acknowledged batches, and then checks:

- the total point count;
- dense and sparse collection configuration;
- up to ten sampled IDs, payloads, and vectors.

Cosine vectors are compared after normalization because embedded and server
Qdrant can return different normalized representations of the same vector.
Verification is sampled; it is not an independent comparison of every imported
point.

Successful output looks like:

```text
Imported and verified 44613 points into collection 'neatmem'
```

You can also inspect the collection through Qdrant's API:

```bash
curl -fsS http://127.0.0.1:6333/collections/neatmem
```

Look for `points_count` in the response.

## Step 4: Switch NeatMem to the imported collection

After you are satisfied with the import, edit NeatMem's `.env` yourself:

```dotenv
QDRANT_HOST=127.0.0.1
QDRANT_PORT=6333
```

Restart NeatMem and perform a normal memory search to confirm the application
works with the server.

The migration commands do **not** edit `.env`, restart NeatMem, or delete the
old embedded data. After NeatMem writes new memories to the server, the old
embedded data is no longer current; rollback then requires reverse
synchronization or an explicit decision to discard those new writes.

## Moving the file to another machine

Copy the ZIP to the destination machine using your normal secure file-transfer
method. Then run the same `--dry-run` and import commands there, changing
`--target-url` to that machine's Qdrant server.

The ZIP is portable data. The paths in your export command and Qdrant server
configuration are machine-specific and are not embedded in the archive.

## If something goes wrong

| Problem | What it means and what to do |
|---|---|
| Source is locked | NeatMem or another process is using the embedded directory. Stop it before exporting. |
| Output already exists | The exporter never overwrites it. Choose a new output path. |
| A `.partial` file remains | Export did not complete. Keep it for investigation if needed, but do not import it. |
| Destination collection already exists | Import refuses to merge or overwrite. Use an empty destination after inspecting any existing data. |
| Checksum mismatch | The archive changed or was copied incorrectly. Validate the transfer or export a fresh archive. |
| Qdrant is unavailable | Start the server and confirm `/healthz` before retrying. |
| Import fails after writing starts | The destination may be partial. Do not treat it as complete; inspect it and use a fresh destination after an explicit cleanup decision. |

Opening embedded Qdrant can use substantial startup memory even though export
writes points in batches. Export runs as a separate short-lived process, so its
memory is released when it exits. Qdrant server also needs its own memory;
measure both processes when evaluating the result.

## Security and current limits

- Archives may contain sensitive memories in plaintext. Protect file
  permissions and use secure transfer and storage.
- SHA-256 detects accidental corruption. It does not encrypt the archive or
  prove who created it.
- The commands currently support one offline collection export/import into a
  new destination collection.
- Resume, merging into an existing collection, online migration, other
  databases, other memory products, automatic model changes, and automatic
  NeatMem cutover are not currently supported.
