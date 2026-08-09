"""Lazy singleton factory for the KuzuGraphStore.

Builds the GraphAdapter (LLM client from resolved LLM_API_KEY/LLM_BASE_URL,
embedder = siliconflow bge-m3) and the KuzuGraphStore on first call. Kept
separate so memory_add.py and main.py share one store instance, and so the
kuzu import only happens when ENABLE_GRAPH=true (per plan: lazy import,
no-op when disabled).
"""

import logging
import os
from typing import Optional

from openai import OpenAI

from neatmem.config import (
    ENABLE_GRAPH,
    GRAPH_EMBEDDING_API_KEY,
    GRAPH_EMBEDDING_BASE_URL,
    GRAPH_EMBEDDING_DIMS,
    GRAPH_EMBEDDING_MODEL,
    GRAPH_THRESHOLD,
    KUZU_DB_PATH,
    LLM_API_KEY,
    LLM_BASE_URL,
)
from neatmem.signals.graph.adapter import GraphAdapter
from neatmem.storage.graph.kuzu_store import KuzuGraphStore

logger = logging.getLogger(__name__)

_graph_store: Optional[KuzuGraphStore] = None


def get_graph_store() -> KuzuGraphStore:
    """Return the process-wide KuzuGraphStore singleton.

    Builds it on first call. Caller must ensure ENABLE_GRAPH is true.
    """
    global _graph_store
    if _graph_store is not None:
        return _graph_store

    if not KUZU_DB_PATH:
        raise RuntimeError("KUZU_DB_PATH must be set when ENABLE_GRAPH=true")

    # LLM_MODEL is defined in main.py, not config.py — read from env directly.
    # main.py refuses to boot without LLM_MODEL, so it is always set here.
    llm_model = os.environ["LLM_MODEL"]

    # LLM client: same resolved key/base_url as the rest of NeatMem
    # (LLM_API_KEY/LLM_BASE_URL handle the provider presets and env fallback).
    # max_retries=5 to handle MiniMax 529 overloaded_error more robustly than
    # the OpenAI default of 2.
    llm_client = OpenAI(
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        max_retries=5,
    )
    # Embedder: siliconflow bge-m3, separate client (different base_url + key).
    embedder_client = OpenAI(
        api_key=GRAPH_EMBEDDING_API_KEY,
        base_url=GRAPH_EMBEDDING_BASE_URL,
    )

    adapter = GraphAdapter(
        llm_client=llm_client,
        llm_model=llm_model,
        embedder_client=embedder_client,
        embedder_model=GRAPH_EMBEDDING_MODEL,
        embedding_dims=GRAPH_EMBEDDING_DIMS,
    )
    _graph_store = KuzuGraphStore(
        db_path=KUZU_DB_PATH,
        adapter=adapter,
        threshold=GRAPH_THRESHOLD,
    )
    logger.info(
        "KuzuGraphStore initialized (db=%s, threshold=%s, dims=%s, llm=%s, embed=%s)",
        KUZU_DB_PATH, GRAPH_THRESHOLD, GRAPH_EMBEDDING_DIMS, llm_model, GRAPH_EMBEDDING_MODEL,
    )
    return _graph_store
