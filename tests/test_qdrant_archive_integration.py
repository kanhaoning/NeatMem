"""Opt-in native Qdrant server verification for the archive CLI."""

import json
import os
import subprocess
import sys
import uuid
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from qdrant_client import QdrantClient, models

from neatmem.evaluation.orchestrator import find_free_port, start_qdrant, stop

pytestmark = pytest.mark.skipif(
    os.environ.get("NEATMEM_QDRANT_INTEGRATION") != "1",
    reason="set NEATMEM_QDRANT_INTEGRATION=1 and QDRANT_BIN to run native Qdrant tests",
)

REPO_ROOT = Path(__file__).resolve().parents[1]
COLLECTION = "archive_integration"


def _cli(*args):
    result = subprocess.run(
        [sys.executable, "-m", "neatmem.cli", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result


def _start_server(root: Path):
    binary = os.environ.get("QDRANT_BIN")
    if not binary:
        pytest.skip("QDRANT_BIN is required for native Qdrant integration")
    (root / "logs").mkdir(parents=True, exist_ok=True)
    http_port = find_free_port()
    grpc_port = find_free_port()
    proc = start_qdrant(
        SimpleNamespace(qdrant_bin=binary),
        root,
        http_port,
        grpc_port,
        root / "storage",
    )
    return proc, f"http://127.0.0.1:{http_port}"


def _make_source(path: Path):
    client = QdrantClient(path=str(path))
    client.create_collection(
        COLLECTION,
        vectors_config=models.VectorParams(size=4, distance=models.Distance.COSINE),
        sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    points = []
    for index in range(7):
        points.append(models.PointStruct(
            id=index + 100 if index % 2 else str(uuid.UUID(int=index + 1000)),
            vector={
                "": [1.0 if position == index % 4 else 0.01 * (index + 1) for position in range(4)],
                "bm25": models.SparseVector(indices=[index + 1, 100], values=[0.25, 0.75]),
            },
            payload={"user_id": "alice" if index % 2 else "bob", "position": index},
        ))
    client.upsert(COLLECTION, points=points, wait=True)
    client.close()
    return points


def _id_key(value):
    return f"{type(value).__name__}:{value}"


def _normalized_unnamed_dense(vector):
    if isinstance(vector, dict):
        return {
            key: _normalized_unnamed_dense(value) if key == "" else value
            for key, value in vector.items()
        }
    if isinstance(vector, list) and vector and isinstance(vector[0], (int, float)):
        norm = sum(float(value) ** 2 for value in vector) ** 0.5
        return [float(value) / norm for value in vector] if norm else vector
    return vector


def _assert_close(expected, actual, path="vector"):
    if isinstance(expected, models.SparseVector):
        expected = expected.model_dump(exclude_none=True)
    if isinstance(actual, models.SparseVector):
        actual = actual.model_dump(exclude_none=True)
    if isinstance(expected, dict) and isinstance(actual, dict):
        assert expected.keys() == actual.keys(), path
        for key in expected:
            _assert_close(expected[key], actual[key], f"{path}.{key}")
        return
    if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
        assert len(expected) == len(actual), path
        for index, (left, right) in enumerate(zip(expected, actual)):
            _assert_close(left, right, f"{path}[{index}]")
        return
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        assert actual == pytest.approx(expected, rel=1e-6, abs=1e-6), path
        return
    assert expected == actual, path


def _all_points(client):
    records, offset = client.scroll(COLLECTION, limit=100, with_payload=True, with_vectors=True)
    assert offset is None
    return {_id_key(record.id): record for record in records}


def _dense_query(client):
    return client.query_points(
        COLLECTION,
        query=[1.0, 0.0, 0.0, 0.0],
        query_filter=models.Filter(must=[
            models.FieldCondition(key="user_id", match=models.MatchValue(value="bob"))
        ]),
        limit=5,
    ).points


def _bm25_query(client):
    return client.query_points(
        COLLECTION,
        query=models.SparseVector(indices=[1, 100], values=[1.0, 1.0]),
        using="bm25",
        limit=5,
    ).points


def test_native_cli_round_trip_search_and_restart(tmp_path):
    _make_source(tmp_path / "embedded")
    archive_path = tmp_path / "archive.zip"
    exported = _cli(
        "migrate", "qdrant", "export",
        "--source-path", str(tmp_path / "embedded"),
        "--collection", COLLECTION,
        "--output", str(archive_path),
        "--batch-size", "2",
    )
    assert exported.returncode == 0, exported.stderr
    with zipfile.ZipFile(archive_path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["point_count"] == 7
        assert len(archive.read("points.jsonl").splitlines()) == 7

    server_root = tmp_path / "native-server"
    proc, url = _start_server(server_root)
    second_proc = None
    try:
        dry_run = _cli(
            "migrate", "qdrant", "import", "--input", str(archive_path),
            "--target-url", url, "--dry-run",
        )
        assert dry_run.returncode == 0, dry_run.stderr
        probe = QdrantClient(url=url)
        assert not probe.collection_exists(COLLECTION)
        probe.close()

        imported = _cli(
            "migrate", "qdrant", "import", "--input", str(archive_path),
            "--target-url", url, "--batch-size", "3",
        )
        assert imported.returncode == 0, imported.stderr
        assert "Imported and verified 7 points" in imported.stdout

        source = QdrantClient(path=str(tmp_path / "embedded"))
        destination = QdrantClient(url=url)
        expected = _all_points(source)
        actual = _all_points(destination)
        assert actual.keys() == expected.keys()
        for key, expected_point in expected.items():
            actual_point = actual[key]
            assert actual_point.payload == expected_point.payload
            _assert_close(
                _normalized_unnamed_dense(expected_point.vector),
                _normalized_unnamed_dense(actual_point.vector),
            )

        source_dense = _dense_query(source)
        destination_dense = _dense_query(destination)
        assert [point.id for point in destination_dense] == [point.id for point in source_dense]
        source_bm25 = _bm25_query(source)
        destination_bm25 = _bm25_query(destination)
        assert destination_bm25
        assert source_bm25
        assert destination_bm25[0].id == source_bm25[0].id
        source.close()
        destination.close()

        duplicate = _cli(
            "migrate", "qdrant", "import", "--input", str(archive_path), "--target-url", url,
        )
        assert duplicate.returncode == 1
        assert "already exists" in duplicate.stderr

        stop(proc)
        proc = None
        second_proc, second_url = _start_server(server_root)
        restarted = QdrantClient(url=second_url)
        assert restarted.count(COLLECTION).count == 7
        assert len(_all_points(restarted)) == 7
        restarted.close()
        stop(second_proc)
        second_proc = None

        unavailable = _cli(
            "migrate", "qdrant", "import", "--input", str(archive_path),
            "--target-url", second_url,
        )
        assert unavailable.returncode == 1
        assert unavailable.stderr
    finally:
        stop(proc)
        stop(second_proc)
