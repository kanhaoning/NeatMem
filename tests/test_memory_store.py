"""Regression tests for the self-implemented storage layer (MemoryStore).

Covers the "minimal add+search per signal" contract:
  - qdrant collection always declares the `bm25` sparse slot
  - MemoryStore CRUD roundtrip (mem0-compatible shapes)
  - BM25 signal: index + keyword search hit
  - Entity signal: entity store write + boosted search path

Requires SILICONFLOW_API_KEY (repo .env provides it). Each test uses a fresh
embedded qdrant under tmp/ so nothing shared is touched.

Run:  cd <repo root> && python -m pytest tests/test_memory_store.py -x -q
"""

import os
import tempfile
import uuid

import pytest
from dotenv import load_dotenv

load_dotenv()

if not os.environ.get("SILICONFLOW_API_KEY"):
    pytest.skip("SILICONFLOW_API_KEY not set", allow_module_level=True)

from neatmem.embeddings import OpenAIEmbedder  # noqa: E402
from neatmem.memory_search import search_memories  # noqa: E402
from neatmem.memory_store import MemoryStore  # noqa: E402
from neatmem.signals.bm25.factory import create_bm25_index  # noqa: E402
from neatmem.signals.entity.factory import create_entity_extractor  # noqa: E402
from neatmem.storage.entity.factory import create_entity_store  # noqa: E402
from neatmem.storage.vector.factory import create_vector_store  # noqa: E402

TMP_ROOT = tempfile.mkdtemp(prefix="neatmem_test_")


def _make_store(tag: str) -> MemoryStore:
    workdir = os.path.join(TMP_ROOT, f"{tag}_{uuid.uuid4().hex[:8]}")
    os.makedirs(workdir, exist_ok=True)
    embedder = OpenAIEmbedder(
        model="BAAI/bge-m3",
        api_key=os.environ["SILICONFLOW_API_KEY"],
        base_url="https://api.siliconflow.cn/v1",
        expected_dims=1024,
    )
    vs = create_vector_store(
        "qdrant", collection_name="mem0", embedding_model_dims=1024,
        path=workdir, on_disk=False,
    )
    return MemoryStore(
        vector_store=vs,
        embedding_model=embedder,
        history_db_path=os.path.join(workdir, "mem_history.db"),
    )


@pytest.fixture(scope="module")
def store():
    s = _make_store("regression")
    yield s
    s.close()


def test_collection_declares_bm25_slot(store):
    info = store.vector_store.client.get_collection("mem0")
    sparse_cfg = info.config.params.sparse_vectors
    assert sparse_cfg and "bm25" in sparse_cfg


def test_crud_roundtrip(store):
    user = "pytest_user"
    res = store.add(
        messages=[
            {"role": "user", "content": "I enjoy playing the piano every evening.", "name": "Dana"},
            {"role": "system", "content": "skipped system message"},
            {"role": "assistant", "content": "That is a lovely habit."},
        ],
        user_id=user,
        metadata={"origin": "pytest"},
        infer=False,
    )
    assert len(res["results"]) == 2  # system message skipped
    item = res["results"][0]
    assert item["event"] == "ADD" and item["actor_id"] == "Dana" and item["role"] == "user"
    mid = item["id"]

    got = store.get(mid)
    assert got["memory"] == "I enjoy playing the piano every evening."
    assert got["user_id"] == user and got["actor_id"] == "Dana"
    assert got["metadata"] == {"origin": "pytest"}
    assert got["hash"] and got["created_at"]

    all_res = store.get_all(filters={"user_id": user}, top_k=100)
    assert len(all_res["results"]) == 2

    store.update(memory_id=mid, data="I enjoy playing the grand piano every evening.")
    got2 = store.get(mid)
    assert "grand piano" in got2["memory"]
    assert got2["created_at"] == got["created_at"]  # preserved on update

    hist = store.history(mid)
    assert [h["event"] for h in hist] == ["ADD", "UPDATE"]

    other = res["results"][1]["id"]
    store.delete(other)
    assert store.get(other) is None
    hist_del = store.history(other)
    assert hist_del[-1]["event"] == "DELETE" and hist_del[-1]["is_deleted"] is True

    store.delete_all(user_id=user)
    assert store.get_all(filters={"user_id": user}, top_k=100)["results"] == []


def test_bm25_signal_add_search(store):
    user = "pytest_bm25"
    res = store.add(
        messages=[{"role": "user", "content": "The ZebraFinch prototype uses a flux capacitor."}],
        user_id=user,
        infer=False,
    )
    mid = res["results"][0]["id"]

    bm25 = create_bm25_index("qdrant_sparse", vector_store=store.vector_store,
                             collection_name="mem0")
    bm25.index_memory(mid, "The ZebraFinch prototype uses a flux capacitor.")
    hits = bm25.search("ZebraFinch flux capacitor", filters={"user_id": user}, top_k=5)
    assert hits and hits[0].memory_id == mid

    sr = search_memories(
        memory=store, query="flux capacitor", filters={"user_id": user},
        top_k=5, use_bm25=True, use_entity=False, bm25_index=bm25,
    )
    assert any(r["id"] == mid for r in sr["results"])


def test_entity_signal_add_search(store):
    user = "pytest_entity"
    res = store.add(
        messages=[{"role": "user", "content": "Caroline visited the Riverside Gallery last Tuesday."}],
        user_id=user,
        infer=False,
    )
    mid = res["results"][0]["id"]

    extractor = create_entity_extractor("ner")
    entity_store = create_entity_store(
        "qdrant", qdrant_client=store.vector_store.client,
        collection_name="mem0_entities_pytest", vector_size=1024,
    )
    entities = extractor.extract("Caroline visited the Riverside Gallery last Tuesday.")
    assert entities, "NER extractor returned no entities"
    entity_store.link_entities(
        entities, memory_id=mid, scope=f"user_id={user}",
        embed_fn=store.embedding_model.embed,
    )

    sr = search_memories(
        memory=store, query="Where did Caroline go?", filters={"user_id": user},
        top_k=5, use_bm25=False, use_entity=True,
        entity_extractor=extractor, entity_store=entity_store,
    )
    assert any(r["id"] == mid for r in sr["results"])
