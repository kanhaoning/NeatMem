import json
import subprocess
import sys
import uuid
import zipfile
from pathlib import Path

import pytest
from qdrant_client import QdrantClient, models

from neatmem.migration import qdrant_archive
from neatmem.migration.qdrant_archive import (
    QdrantArchiveError,
    export_collection,
    import_collection,
)

COLLECTION = "archive_test"


def _make_source(path: Path, count: int = 7, named: bool = False) -> list[models.PointStruct]:
    client = QdrantClient(path=str(path))
    vectors = (
        {"text": models.VectorParams(size=4, distance=models.Distance.COSINE)}
        if named
        else models.VectorParams(size=4, distance=models.Distance.COSINE)
    )
    client.create_collection(
        COLLECTION,
        vectors_config=vectors,
        sparse_vectors_config={} if named else {
            "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)
        },
    )
    points = []
    for index in range(count):
        dense = [1.0 if position == index % 4 else 0.01 * (index + 1) for position in range(4)]
        vector = {"text": dense} if named else {
            "": dense,
            "bm25": models.SparseVector(indices=[index + 1, 100], values=[0.25, 0.75]),
        }
        points.append(models.PointStruct(
            id=index + 100 if index % 2 else str(uuid.UUID(int=index + 1)),
            vector=vector,
            payload={"user_id": "alice" if index % 2 else "bob", "position": index},
        ))
    client.upsert(COLLECTION, points=points, wait=True)
    client.close()
    return points


def _rewrite_archive(source: Path, target: Path, mutate) -> None:
    with zipfile.ZipFile(source) as original, zipfile.ZipFile(target, "w") as rewritten:
        for name in original.namelist():
            data = original.read(name)
            data = mutate(name, data)
            rewritten.writestr(name, data)


def test_export_contains_all_points_and_vectors(tmp_path):
    expected = _make_source(tmp_path / "embedded")
    archive_path = tmp_path / "export.zip"

    result = export_collection(tmp_path / "embedded", COLLECTION, archive_path, batch_size=2)

    assert result["points"] == len(expected)
    with zipfile.ZipFile(archive_path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        lines = archive.read("points.jsonl").splitlines()
    assert manifest["format_version"] == 1
    assert manifest["point_count"] == len(expected)
    assert len(manifest["points_sha256"]) == 64
    assert len(lines) == len(expected)
    records = [models.PointStruct.model_validate(json.loads(line)) for line in lines]
    assert {str(point.id) for point in records} == {str(point.id) for point in expected}
    for point in records:
        assert set(point.vector) == {"", "bm25"}
        assert len(point.vector[""]) == 4
        assert point.vector["bm25"].indices


def test_export_preserves_named_dense_vectors(tmp_path):
    _make_source(tmp_path / "embedded", count=3, named=True)
    archive_path = tmp_path / "named.zip"

    export_collection(tmp_path / "embedded", COLLECTION, archive_path, batch_size=2)

    with zipfile.ZipFile(archive_path) as archive:
        lines = archive.read("points.jsonl").splitlines()
    for line in lines:
        vector = json.loads(line)["vector"]
        assert set(vector) == {"text"}
        assert len(vector["text"]) == 4


def test_export_refuses_missing_source_collection_and_existing_output(tmp_path):
    _make_source(tmp_path / "embedded")
    with pytest.raises(QdrantArchiveError, match="source collection does not exist"):
        export_collection(tmp_path / "embedded", "missing", tmp_path / "missing.zip")
    with pytest.raises(QdrantArchiveError, match="does not exist"):
        export_collection(tmp_path / "absent", COLLECTION, tmp_path / "absent.zip")

    output = tmp_path / "existing.zip"
    output.write_bytes(b"keep me")
    with pytest.raises(QdrantArchiveError, match="output already exists"):
        export_collection(tmp_path / "embedded", COLLECTION, output)
    assert output.read_bytes() == b"keep me"
    with pytest.raises(QdrantArchiveError, match="positive integer"):
        export_collection(tmp_path / "embedded", COLLECTION, tmp_path / "new.zip", batch_size=0)


def test_export_source_lock_is_a_clean_error(tmp_path):
    _make_source(tmp_path / "embedded")
    client = QdrantClient(path=str(tmp_path / "embedded"))
    try:
        with pytest.raises(QdrantArchiveError, match="export failed"):
            export_collection(tmp_path / "embedded", COLLECTION, tmp_path / "locked.zip")
    finally:
        client.close()


def test_import_dry_run_rejects_checksum_mismatch(tmp_path):
    _make_source(tmp_path / "embedded")
    archive_path = tmp_path / "export.zip"
    export_collection(tmp_path / "embedded", COLLECTION, archive_path)

    def corrupt(name, data):
        if name != "points.jsonl":
            return data
        lines = data.splitlines()
        point = json.loads(lines[0])
        point["payload"]["position"] = 999
        lines[0] = json.dumps(point).encode()
        return b"\n".join(lines) + b"\n"

    corrupted = tmp_path / "corrupt.zip"
    _rewrite_archive(archive_path, corrupted, corrupt)
    with pytest.raises(QdrantArchiveError, match="checksum mismatch"):
        import_collection(corrupted, "http://127.0.0.1:1", dry_run=True)


def test_import_rejects_unsupported_format_version(tmp_path):
    _make_source(tmp_path / "embedded")
    archive_path = tmp_path / "export.zip"
    export_collection(tmp_path / "embedded", COLLECTION, archive_path)

    def change_version(name, data):
        if name != "manifest.json":
            return data
        manifest = json.loads(data)
        manifest["format_version"] = 999
        return json.dumps(manifest).encode()

    changed = tmp_path / "changed.zip"
    _rewrite_archive(archive_path, changed, change_version)
    with pytest.raises(QdrantArchiveError, match="format_version"):
        import_collection(changed, "http://127.0.0.1:1", dry_run=True)


class _FakeDestination:
    def __init__(self, exists=False):
        self.exists = exists
        self.closed = False
        self.write_called = False

    def collection_exists(self, name):
        return self.exists

    def close(self):
        self.closed = True

    def create_collection(self, *args, **kwargs):
        self.write_called = True
        raise AssertionError("dry-run must not create collections")

    def upsert(self, *args, **kwargs):
        self.write_called = True
        raise AssertionError("dry-run must not upsert")


def test_import_dry_run_checks_target_without_writes(tmp_path, monkeypatch):
    _make_source(tmp_path / "embedded")
    archive_path = tmp_path / "export.zip"
    export_collection(tmp_path / "embedded", COLLECTION, archive_path)
    fake = _FakeDestination()
    monkeypatch.setattr(qdrant_archive, "QdrantClient", lambda **kwargs: fake)

    result = import_collection(archive_path, "http://qdrant.invalid", dry_run=True)

    assert result == {"collection": COLLECTION, "points": 7, "dry_run": True}
    assert fake.closed and not fake.write_called


def test_import_dry_run_rejects_existing_destination(tmp_path, monkeypatch):
    _make_source(tmp_path / "embedded")
    archive_path = tmp_path / "export.zip"
    export_collection(tmp_path / "embedded", COLLECTION, archive_path)
    fake = _FakeDestination(exists=True)
    monkeypatch.setattr(qdrant_archive, "QdrantClient", lambda **kwargs: fake)

    with pytest.raises(QdrantArchiveError, match="already exists"):
        import_collection(archive_path, "http://qdrant.invalid", dry_run=True)
    assert fake.closed and not fake.write_called


def test_cli_help_and_invalid_batch_size():
    help_result = subprocess.run(
        [sys.executable, "-m", "neatmem.cli", "migrate", "qdrant", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert help_result.returncode == 0, help_result.stderr
    assert "export" in help_result.stdout and "import" in help_result.stdout

    invalid = subprocess.run(
        [sys.executable, "-m", "neatmem.cli", "migrate", "qdrant", "export",
         "--source-path", "x", "--collection", "x", "--output", "x.zip",
         "--batch-size", "0"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert invalid.returncode == 2
    assert "greater than zero" in invalid.stderr
