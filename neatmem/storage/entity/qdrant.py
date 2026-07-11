"""Qdrant-backed entity store.

Reads and writes the existing `mem0_entities` collection shape used by mem0:
- payload["data"]          -> entity text
- payload["entity_type"]   -> PROPER/COMPOUND/QUOTED/NOUN
- payload["linked_memory_ids"] -> list of memory ids
"""
import logging
import uuid
from typing import Any, Callable, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointIdsList,
    PointStruct,
    VectorParams,
)

from neatmem.signals.entity.base import Entity
from neatmem.storage.entity.base import AbstractEntityStore, EntityRecord

logger = logging.getLogger(__name__)


class QdrantEntityStore(AbstractEntityStore):
    def __init__(
        self,
        qdrant_client: Optional[QdrantClient] = None,
        collection_name: str = "mem0_entities",
        vector_size: int = 1024,
        path: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
    ):
        if qdrant_client is not None:
            self.client = qdrant_client
        elif host and port:
            self.client = QdrantClient(host=host, port=port)
        else:
            self.client = QdrantClient(path=path or "qdrant_db")

        self.collection_name = collection_name
        self.vector_size = vector_size
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        cols = self.client.get_collections()
        names = {c.name for c in cols.collections}
        if self.collection_name in names:
            return
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
        )
        logger.info("Created entity collection: %s", self.collection_name)

    @staticmethod
    def _build_filter(filters: Dict[str, Any]) -> Optional[Filter]:
        must = []
        for key, value in filters.items():
            if key in ("user_id", "agent_id", "run_id") and value:
                must.append(FieldCondition(key=key, match=MatchValue(value=value)))
        if not must:
            return None
        return Filter(must=must)

    def find_matching_by_vector(
        self,
        query: str,
        vectors: List[float],
        top_k: int,
        filters: Dict[str, Any],
    ) -> List[EntityRecord]:
        query_filter = self._build_filter(filters)
        hits = self.client.query_points(
            collection_name=self.collection_name,
            query=vectors,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )
        records = []
        for point in hits.points:
            payload = point.payload or {}
            records.append(
                EntityRecord(
                    entity_id=str(point.id),
                    entity_text=payload.get("data", ""),
                    entity_type=payload.get("entity_type", ""),
                    scope="",  # not stored as a single field by mem0
                    linked_memory_ids=payload.get("linked_memory_ids", []) or [],
                    score=point.score if hasattr(point, "score") else None,
                )
            )
        return records

    def link_entities(
        self,
        entities: List[Any],
        memory_id: str,
        scope: str,
        embed_fn: Optional[Callable[[str, str], List[float]]] = None,
    ) -> None:
        if not entities or not memory_id:
            return

        search_filters = self._parse_scope(scope)
        for entity in entities:
            entity_type = getattr(entity, "type", None)
            entity_text = getattr(entity, "text", None)
            if not isinstance(entity, Entity):
                # tolerate raw (type, text) tuples for parity with mem0
                if isinstance(entity, (tuple, list)) and len(entity) >= 2:
                    entity_type, entity_text = entity[0], entity[1]
            if not entity_text or not str(entity_text).strip():
                continue
            entity_text = str(entity_text).strip()

            existing = self._find_existing_entity(entity_text, search_filters, embed_fn=embed_fn)
            if existing is not None:
                point_id, payload = existing
                linked = list(payload.get("linked_memory_ids", []) or [])
                if memory_id not in linked:
                    linked.append(memory_id)
                    self.client.set_payload(
                        collection_name=self.collection_name,
                        payload={"linked_memory_ids": linked},
                        points=[point_id],
                    )
            else:
                if embed_fn is None:
                    logger.warning("Cannot insert new entity without embed_fn; skipping %s", entity_text)
                    continue
                entity_id = str(uuid.uuid4())
                vector = embed_fn(entity_text, "add")
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=[
                        PointStruct(
                            id=entity_id,
                            vector=vector,
                            payload={
                                "data": entity_text,
                                "entity_type": entity_type or "",
                                "linked_memory_ids": [memory_id],
                                **search_filters,
                            },
                        )
                    ],
                )

    def _parse_scope(self, scope: str) -> Dict[str, Any]:
        """Parse scope string like app_id=app1&user_id=alice into filter dict."""
        filters: Dict[str, Any] = {}
        if not scope:
            return filters
        for part in scope.split("&"):
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            if key in ("user_id", "agent_id", "run_id", "app_id") and value:
                filters[key] = value
        return filters

    def _find_existing_entity(
        self, entity_text: str, filters: Dict[str, Any], embed_fn: Optional[Callable] = None,
    ) -> Optional[tuple]:
        """Search for an entity with matching text under the filters.

        When embed_fn is provided, uses vector similarity (score >= 0.95) to
        match mem0 native _upsert_entity behavior. Falls back to exact text
        match when embed_fn is None.

        Returns (point_id, payload) or None.
        """
        query_filter = self._build_filter(filters)

        if embed_fn is not None:
            # Vector similarity search (matching mem0 native _upsert_entity)
            embedding = embed_fn(entity_text, "add")
            hits = self.client.query_points(
                collection_name=self.collection_name,
                query=embedding,
                query_filter=query_filter,
                limit=1,
                with_payload=True,
            )
            for point in hits.points:
                if point.score is not None and point.score >= 0.95:
                    return point.id, point.payload or {}
            return None

        # Fallback: exact text match
        must = [FieldCondition(key="data", match=MatchValue(value=entity_text))]
        if query_filter and query_filter.must:
            must.extend(query_filter.must)
        exact_filter = Filter(must=must)
        hits = self.client.query_points(
            collection_name=self.collection_name,
            query=None,
            query_filter=exact_filter,
            limit=10,
            with_payload=True,
        )
        for point in hits.points:
            payload = point.payload or {}
            if payload.get("data") == entity_text:
                return point.id, payload
        return None

    def delete_by_memory_id(self, memory_id: str, scope: str) -> None:
        if not memory_id:
            return
        search_filters = self._parse_scope(scope)
        query_filter = self._build_filter(search_filters)

        all_points, _next = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=query_filter,
            limit=10000,
            with_payload=True,
            with_vectors=False,
        )
        for point in all_points:
            payload = point.payload or {}
            linked = list(payload.get("linked_memory_ids", []) or [])
            if memory_id not in linked:
                continue
            linked.remove(memory_id)
            if linked:
                self.client.set_payload(
                    collection_name=self.collection_name,
                    payload={"linked_memory_ids": linked},
                    points=[point.id],
                )
            else:
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=PointIdsList(points=[point.id]),
                )
