"""Versioned, streaming Qdrant point archives.

The format is intentionally Qdrant-specific: a ZIP containing a JSON manifest
and one JSON-encoded PointStruct per line. It is not a general memory-product
exchange format.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import zipfile
from datetime import datetime, timezone
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any, Iterator

from qdrant_client import QdrantClient, models

from neatmem import __version__

FORMAT_KIND = "neatmem-qdrant-points"
FORMAT_VERSION = 1
MANIFEST_MEMBER = "manifest.json"
POINTS_MEMBER = "points.jsonl"
_VERIFY_SAMPLE_SIZE = 10


class QdrantArchiveError(RuntimeError):
    """An archive, endpoint, or migration precondition failed."""


def _positive_batch_size(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise QdrantArchiveError("batch_size must be a positive integer")


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _point_json(point: models.PointStruct) -> bytes:
    return _json_bytes(point.model_dump(mode="json", exclude_none=True))


def _payload_schema_json(info: models.CollectionInfo) -> dict[str, Any]:
    return {
        field_name: field_info.model_dump(mode="json", exclude_none=True)
        for field_name, field_info in info.payload_schema.items()
    }


def export_collection(
    source_path: str | os.PathLike[str],
    collection_name: str,
    output_path: str | os.PathLike[str],
    batch_size: int = 100,
) -> dict[str, Any]:
    """Export one embedded collection without accumulating all points in memory."""
    _positive_batch_size(batch_size)
    source = Path(source_path)
    output = Path(output_path)
    partial = output.with_name(output.name + ".partial")
    if not source.is_dir():
        raise QdrantArchiveError(f"embedded Qdrant directory does not exist: {source}")
    if not collection_name:
        raise QdrantArchiveError("collection name must not be empty")
    if output.exists():
        raise QdrantArchiveError(f"output already exists: {output}")
    if partial.exists():
        raise QdrantArchiveError(f"incomplete export already exists: {partial}")

    client: QdrantClient | None = None
    exported = 0
    try:
        client = QdrantClient(path=str(source))
        if not client.collection_exists(collection_name):
            raise QdrantArchiveError(f"source collection does not exist: {collection_name}")
        info = client.get_collection(collection_name)
        expected_before = client.count(collection_name).count
        output.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        offset = None

        with zipfile.ZipFile(
            partial, mode="x", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as archive:
            with archive.open(POINTS_MEMBER, mode="w", force_zip64=True) as points_file:
                while True:
                    records, offset = client.scroll(
                        collection_name,
                        offset=offset,
                        limit=batch_size,
                        with_payload=True,
                        with_vectors=True,
                    )
                    for record in records:
                        point = models.PointStruct(
                            id=record.id, vector=record.vector, payload=record.payload
                        )
                        line = _point_json(point)
                        points_file.write(line)
                        digest.update(line)
                        exported += 1
                    if offset is None:
                        break

            expected_after = client.count(collection_name).count
            if expected_before != expected_after or exported != expected_after:
                raise QdrantArchiveError(
                    "source changed or scroll was incomplete during export "
                    f"(before={expected_before}, after={expected_after}, exported={exported})"
                )
            manifest = {
                "format": FORMAT_KIND,
                "format_version": FORMAT_VERSION,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "producer": {
                    "name": "neatmem",
                    "version": __version__,
                    "qdrant_client": package_version("qdrant-client"),
                },
                "collection": {
                    "name": collection_name,
                    "config": info.config.model_dump(mode="json", exclude_none=True),
                    "payload_schema": _payload_schema_json(info),
                },
                "point_count": exported,
                "batch_size": batch_size,
                "points_member": POINTS_MEMBER,
                "points_sha256": digest.hexdigest(),
            }
            archive.writestr(MANIFEST_MEMBER, _json_bytes(manifest))

        partial.replace(output)
    except QdrantArchiveError:
        raise
    except Exception as exc:
        raise QdrantArchiveError(
            f"export failed for collection {collection_name!r}; "
            f"incomplete archive may remain at {partial}: {exc}"
        ) from exc
    finally:
        if client is not None:
            client.close()

    return {"collection": collection_name, "points": exported, "output": str(output)}


def _read_manifest(archive: zipfile.ZipFile) -> dict[str, Any]:
    names = archive.namelist()
    if len(names) != len(set(names)) or set(names) != {MANIFEST_MEMBER, POINTS_MEMBER}:
        raise QdrantArchiveError(
            "archive must contain exactly one manifest.json and one points.jsonl"
        )
    if archive.getinfo(MANIFEST_MEMBER).file_size > 16 * 1024 * 1024:
        raise QdrantArchiveError("archive manifest is unexpectedly large")
    try:
        manifest = json.loads(archive.read(MANIFEST_MEMBER))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QdrantArchiveError(f"invalid archive manifest: {exc}") from exc
    if not isinstance(manifest, dict):
        raise QdrantArchiveError("archive manifest must be a JSON object")
    if manifest.get("format") != FORMAT_KIND:
        raise QdrantArchiveError(f"unsupported archive format: {manifest.get('format')!r}")
    if manifest.get("format_version") != FORMAT_VERSION:
        raise QdrantArchiveError(
            f"unsupported archive format_version: {manifest.get('format_version')!r}"
        )
    if manifest.get("points_member") != POINTS_MEMBER:
        raise QdrantArchiveError("manifest points_member is invalid")
    point_count = manifest.get("point_count")
    if isinstance(point_count, bool) or not isinstance(point_count, int) or point_count < 0:
        raise QdrantArchiveError("manifest point_count must be a non-negative integer")
    digest = manifest.get("points_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise QdrantArchiveError("manifest points_sha256 is invalid")
    collection = manifest.get("collection")
    if not isinstance(collection, dict) or not collection.get("name"):
        raise QdrantArchiveError("manifest collection name is missing")
    try:
        models.CollectionConfig.model_validate(collection["config"])
        for field_info in collection.get("payload_schema", {}).values():
            models.PayloadIndexInfo.model_validate(field_info)
    except (KeyError, TypeError, ValueError) as exc:
        raise QdrantArchiveError(f"invalid collection metadata: {exc}") from exc
    return manifest


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not supported: {value}")


def _load_point(line: bytes, line_number: int) -> models.PointStruct:
    try:
        value = json.loads(line, parse_constant=_reject_json_constant)
        return models.PointStruct.model_validate(value)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise QdrantArchiveError(f"invalid point at points.jsonl:{line_number}: {exc}") from exc


def _scan_points(
    archive: zipfile.ZipFile, manifest: dict[str, Any]
) -> list[models.PointStruct]:
    """Validate every record and return only a small verification sample."""
    digest = hashlib.sha256()
    count = 0
    sample: list[models.PointStruct] = []
    with archive.open(POINTS_MEMBER, mode="r") as points_file:
        for line_number, line in enumerate(points_file, start=1):
            if not line.strip():
                raise QdrantArchiveError(f"empty point at points.jsonl:{line_number}")
            digest.update(line)
            point = _load_point(line, line_number)
            if len(sample) < _VERIFY_SAMPLE_SIZE:
                sample.append(point)
            count += 1
    if count != manifest["point_count"]:
        raise QdrantArchiveError(
            f"archive point count mismatch: manifest={manifest['point_count']}, actual={count}"
        )
    if digest.hexdigest() != manifest["points_sha256"]:
        raise QdrantArchiveError("archive points.jsonl checksum mismatch")
    return sample


def _iter_batches(
    archive: zipfile.ZipFile, batch_size: int
) -> Iterator[list[models.PointStruct]]:
    batch: list[models.PointStruct] = []
    with archive.open(POINTS_MEMBER, mode="r") as points_file:
        for line_number, line in enumerate(points_file, start=1):
            batch.append(_load_point(line, line_number))
            if len(batch) == batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


def _collection_config(manifest: dict[str, Any]) -> models.CollectionConfig:
    return models.CollectionConfig.model_validate(manifest["collection"]["config"])


def _create_destination(client: QdrantClient, manifest: dict[str, Any]) -> None:
    collection = manifest["collection"]
    config = _collection_config(manifest)
    params = config.params
    if params.sharding_method == models.ShardingMethod.CUSTOM:
        raise QdrantArchiveError("custom-sharded collections are not supported")

    def as_diff(value: Any, diff_type: Any) -> Any:
        if value is None:
            return None
        return diff_type.model_validate(value.model_dump(exclude_none=True))

    client.create_collection(
        collection_name=collection["name"],
        vectors_config=params.vectors,
        sparse_vectors_config=params.sparse_vectors,
        shard_number=params.shard_number,
        sharding_method=params.sharding_method,
        replication_factor=params.replication_factor,
        write_consistency_factor=params.write_consistency_factor,
        on_disk_payload=params.on_disk_payload,
        hnsw_config=as_diff(config.hnsw_config, models.HnswConfigDiff),
        optimizers_config=as_diff(config.optimizer_config, models.OptimizersConfigDiff),
        wal_config=as_diff(config.wal_config, models.WalConfigDiff),
        quantization_config=config.quantization_config,
        strict_mode_config=config.strict_mode_config,
        metadata=config.metadata,
    )
    for field_name, raw_info in collection.get("payload_schema", {}).items():
        field_info = models.PayloadIndexInfo.model_validate(raw_info)
        client.create_payload_index(
            collection["name"],
            field_name=field_name,
            field_schema=field_info.data_type if field_info.params is None else field_info.params,
        )


def _id_key(point_id: Any) -> str:
    return f"{type(point_id).__name__}:{point_id}"


def _compare_vector_values(expected: Any, actual: Any, path: str = "vector") -> None:
    if isinstance(expected, models.SparseVector):
        expected = expected.model_dump(exclude_none=True)
    if isinstance(actual, models.SparseVector):
        actual = actual.model_dump(exclude_none=True)
    if isinstance(expected, dict) and isinstance(actual, dict):
        if expected.keys() != actual.keys():
            raise QdrantArchiveError(f"verification failed at {path}: vector keys differ")
        for key in expected:
            _compare_vector_values(expected[key], actual[key], f"{path}.{key}")
        return
    if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
        if len(expected) != len(actual):
            raise QdrantArchiveError(f"verification failed at {path}: lengths differ")
        for index, (left, right) in enumerate(zip(expected, actual)):
            _compare_vector_values(left, right, f"{path}[{index}]")
        return
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if not math.isclose(float(expected), float(actual), rel_tol=1e-6, abs_tol=1e-6):
            raise QdrantArchiveError(f"verification failed at {path}: {expected} != {actual}")
        return
    if expected != actual:
        raise QdrantArchiveError(f"verification failed at {path}: {expected!r} != {actual!r}")


def _vector_config_json(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {key: item.model_dump(exclude_none=True) for key, item in value.items()}
    return value.model_dump(exclude_none=True)


def _normalized_for_verification(vector: Any, config: Any, name: str = "") -> Any:
    """Account for cosine normalization differences in local/remote retrieval."""
    if isinstance(vector, models.SparseVector) or vector is None:
        return vector
    if isinstance(vector, dict):
        if set(vector) == {"indices", "values"}:
            return vector
        return {
            key: _normalized_for_verification(value, config, key)
            for key, value in vector.items()
        }
    if isinstance(vector, (list, tuple)):
        if vector and isinstance(vector[0], (int, float)):
            params = config.get(name) if isinstance(config, dict) else config
            if params is not None and params.distance == models.Distance.COSINE:
                norm = math.sqrt(math.fsum(float(value) ** 2 for value in vector))
                if norm > 0:
                    return [float(value) / norm for value in vector]
            return vector
        return [_normalized_for_verification(value, config, name) for value in vector]
    return vector


def _verify_import(
    client: QdrantClient, manifest: dict[str, Any], sample: list[models.PointStruct]
) -> None:
    name = manifest["collection"]["name"]
    actual_count = client.count(name).count
    if actual_count != manifest["point_count"]:
        raise QdrantArchiveError(
            f"destination point count mismatch: expected={manifest['point_count']}, actual={actual_count}"
        )
    destination_info = client.get_collection(name)
    expected_vectors = _collection_config(manifest).params.vectors
    actual_vectors = destination_info.config.params.vectors
    if _vector_config_json(expected_vectors) != _vector_config_json(actual_vectors):
        raise QdrantArchiveError("destination dense vector configuration differs")
    expected_sparse = _collection_config(manifest).params.sparse_vectors
    actual_sparse = destination_info.config.params.sparse_vectors
    if not expected_sparse:
        if actual_sparse:
            raise QdrantArchiveError("destination unexpectedly has sparse vectors")
    elif not actual_sparse or {
        key: value.model_dump(exclude_none=True) for key, value in expected_sparse.items()
    } != {key: value.model_dump(exclude_none=True) for key, value in actual_sparse.items()}:
        raise QdrantArchiveError("destination sparse vector configuration differs")

    if not sample:
        return
    records = client.retrieve(
        name, ids=[point.id for point in sample], with_payload=True, with_vectors=True
    )
    actual_by_id = {_id_key(record.id): record for record in records}
    for expected in sample:
        actual = actual_by_id.get(_id_key(expected.id))
        if actual is None:
            raise QdrantArchiveError(f"verification point is missing: {expected.id}")
        if actual.payload != expected.payload:
            raise QdrantArchiveError(f"verification payload differs for point {expected.id}")
        _compare_vector_values(
            _normalized_for_verification(expected.vector, expected_vectors),
            _normalized_for_verification(actual.vector, expected_vectors),
        )


def import_collection(
    input_path: str | os.PathLike[str],
    target_url: str,
    batch_size: int = 100,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Import an archive into a new Qdrant server collection."""
    _positive_batch_size(batch_size)
    source = Path(input_path)
    if not source.is_file():
        raise QdrantArchiveError(f"archive does not exist: {source}")

    try:
        with zipfile.ZipFile(source, mode="r") as archive:
            manifest = _read_manifest(archive)
            sample = _scan_points(archive, manifest)
            if _collection_config(manifest).params.sharding_method == models.ShardingMethod.CUSTOM:
                raise QdrantArchiveError("custom-sharded collections are not supported")
            name = manifest["collection"]["name"]
            client = QdrantClient(url=target_url)
            imported = 0
            try:
                if client.collection_exists(name):
                    raise QdrantArchiveError(f"destination collection already exists: {name}")
                if dry_run:
                    return {
                        "collection": name,
                        "points": manifest["point_count"],
                        "dry_run": True,
                    }
                try:
                    _create_destination(client, manifest)
                    for batch in _iter_batches(archive, batch_size):
                        client.upsert(name, points=batch, wait=True)
                        imported += len(batch)
                    _verify_import(client, manifest, sample)
                except QdrantArchiveError:
                    raise
                except Exception as exc:
                    raise QdrantArchiveError(
                        f"import failed after {imported} points; destination collection "
                        f"{name!r} may be partial: {exc}"
                    ) from exc
            finally:
                client.close()
    except QdrantArchiveError:
        raise
    except (zipfile.BadZipFile, OSError) as exc:
        raise QdrantArchiveError(f"cannot read archive {source}: {exc}") from exc
    except Exception as exc:
        raise QdrantArchiveError(f"import preflight failed: {exc}") from exc

    return {"collection": name, "points": imported, "dry_run": False}
