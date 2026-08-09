"""One-off migration: copy qdrant collections mem0 -> neatmem (and
mem0_entities -> neatmem_entities) inside a local embedded qdrant DB, plus
~/.mem0/history.db -> ~/.neatmem/history.db.

Run while the neatmem server is STOPPED (embedded qdrant holds a lock).
Old collections/dir are left untouched; delete them manually after verifying.

Usage: python migrate_mem0_to_neatmem.py [qdrant_path]
(default: qdrant_db, the pre-rename default location)
"""

import shutil
import sys
from pathlib import Path

from qdrant_client import QdrantClient

QDRANT_PATH = sys.argv[1] if len(sys.argv) > 1 else "qdrant_db"
PAIRS = [("mem0", "neatmem"), ("mem0_entities", "neatmem_entities")]


def migrate_collection(client: QdrantClient, src: str, dst: str) -> None:
    names = [c.name for c in client.get_collections().collections]
    if src not in names:
        print(f"[skip] {src}: not present")
        return
    if dst in names:
        print(f"[skip] {dst}: already exists ({client.get_collection(dst).points_count} points)")
        return
    info = client.get_collection(src)
    client.create_collection(
        collection_name=dst,
        vectors_config=info.config.params.vectors,
        sparse_vectors_config=info.config.params.sparse_vectors,
    )
    total = 0
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=src, limit=256, offset=offset,
            with_payload=True, with_vectors=True,
        )
        if not points:
            break
        client.upsert(collection_name=dst, points=points)
        total += len(points)
        if offset is None:
            break
    dst_count = client.get_collection(dst).points_count
    assert dst_count == info.points_count, f"count mismatch: {dst_count} != {info.points_count}"
    print(f"[ok] {src} -> {dst}: {total} points")


def migrate_history_db() -> None:
    src = Path.home() / ".mem0" / "history.db"
    dst = Path.home() / ".neatmem" / "history.db"
    if not src.exists():
        print("[skip] ~/.mem0/history.db not present")
        return
    if dst.exists():
        print("[skip] ~/.neatmem/history.db already exists")
        return
    dst.parent.mkdir(exist_ok=True)
    shutil.copy2(src, dst)
    print(f"[ok] {src} -> {dst}")


if __name__ == "__main__":
    client = QdrantClient(path=QDRANT_PATH)
    for src, dst in PAIRS:
        migrate_collection(client, src, dst)
    client.close()
    migrate_history_db()
