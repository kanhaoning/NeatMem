# The CRUD semantics in this file are ported from mem0
# (https://github.com/mem0ai/mem0), v2.0.0 (mem0/memory/main.py, sync Memory
# class: _add_to_vector_store infer=False branch, get, get_all, update,
# delete, delete_all, history, reset, _create_memory, _update_memory,
# _delete_memory, and the entity-store linking helpers).
# Copyright (c) Mem0
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Modifications relative to the mem0 original:
#   - Telemetry (capture_event / MEM0_TELEMETRY) removed.
#   - LLM-dependent paths removed: infer=True extraction pipeline, procedural
#     memory, and vision message parsing. NeatMem runs its own extraction
#     pipeline (neatmem.memory_add) and always calls add(infer=False).
#     add(infer=True) and non-text message content raise NotImplementedError
#     explicitly instead of silently doing something different.
#   - mem0's Mem0ValidationError is replaced by plain ValueError (mem0's
#     validation helpers already raised ValueError subclasses).
#   - Entity linking uses NeatMem's vendored spaCy utilities
#     (neatmem.utils.spacy) and NeatMem's own Qdrant wrapper; behavior is
#     identical to mem0's (lazy `<collection>_entities` store sharing the
#     main store's client, link on update, unlink on update/delete).
#
# Attribute and method names (.vector_store, .embedding_model, .db,
# .collection_name, add/get/get_all/update/delete/delete_all/history/reset)
# intentionally mirror mem0's Memory class so existing call sites work
# unchanged.

import hashlib
import logging
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from neatmem.storage.history import SQLiteHistoryManager
from neatmem.storage.vector.qdrant import Qdrant
from neatmem.utils.spacy import extract_entities, lemmatize_for_bm25

logger = logging.getLogger(__name__)

# Entity ID parameters that must live inside `filters`, not as top-level kwargs
_ENTITY_PARAMS = {"user_id", "agent_id", "run_id"}

# Payload keys promoted to top level in get()/get_all() results
_PROMOTED_PAYLOAD_KEYS = ["user_id", "agent_id", "run_id", "actor_id", "role"]

# Payload keys that are not user metadata (core fields + promoted keys)
_CORE_AND_PROMOTED_KEYS = {
    "data",
    "hash",
    "created_at",
    "updated_at",
    "id",
    "text_lemmatized",
    "attributed_to",
    *_PROMOTED_PAYLOAD_KEYS,
}


def _reject_top_level_entity_params(kwargs: Dict[str, Any], method_name: str) -> None:
    """Reject top-level entity parameters - must use filters instead."""
    invalid_keys = _ENTITY_PARAMS & set(kwargs.keys())
    if invalid_keys:
        raise ValueError(
            f"Top-level entity parameters {invalid_keys} are not supported in {method_name}(). "
            f"Use filters={{'user_id': '...'}} instead."
        )


def _validate_and_trim_entity_id(value: Optional[str], name: str) -> Optional[str]:
    """
    Validates and normalizes an entity ID.
    - Trims leading/trailing whitespace
    - Rejects empty or whitespace-only strings
    - Rejects strings containing internal whitespace
    """
    if value is None:
        return None
    trimmed = value.strip()
    if trimmed == "":
        raise ValueError(
            f"Invalid {name}: cannot be empty or whitespace-only. Provide a valid identifier."
        )
    if any(c.isspace() for c in trimmed):
        raise ValueError(
            f"Invalid {name}: cannot contain whitespace. Provide a valid identifier without spaces."
        )
    return trimmed


def _validate_search_params(threshold: Optional[float] = None, top_k: Optional[int] = None) -> None:
    """Validates search parameters (threshold in [0, 1], top_k non-negative int)."""
    if threshold is not None:
        if not isinstance(threshold, (int, float)):
            raise ValueError("threshold must be a valid number")
        if threshold < 0 or threshold > 1:
            raise ValueError(
                f"Invalid threshold: {threshold}. Must be between 0 and 1 (inclusive)."
            )
    if top_k is not None:
        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise ValueError("top_k must be a valid integer")
        if top_k < 0:
            raise ValueError(
                f"Invalid top_k: {top_k}. Must be a non-negative integer."
            )


def _normalize_iso_timestamp_to_utc(timestamp: Optional[str]) -> Optional[str]:
    """Normalize timezone-aware ISO timestamps to UTC without rewriting naive values."""
    if not timestamp:
        return timestamp
    try:
        parsed = datetime.fromisoformat(timestamp)
    except ValueError:
        return timestamp
    if parsed.tzinfo is None:
        return timestamp
    return parsed.astimezone(timezone.utc).isoformat()


def _build_filters_and_metadata(
    *,  # Enforce keyword-only arguments
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    run_id: Optional[str] = None,
    actor_id: Optional[str] = None,  # For query-time filtering
    input_metadata: Optional[Dict[str, Any]] = None,
    input_filters: Optional[Dict[str, Any]] = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Constructs metadata for storage and filters for querying based on session
    and actor identifiers.

    Returns two dicts:
    1. `base_metadata_template`: template for metadata when storing new
       memories (session identifiers + input_metadata).
    2. `effective_query_filters`: filters for querying existing memories
       (session identifiers + input_filters + resolved actor_id).

    Actor filtering precedence: explicit `actor_id` arg -> `filters["actor_id"]`.
    """
    base_metadata_template = deepcopy(input_metadata) if input_metadata else {}
    effective_query_filters = deepcopy(input_filters) if input_filters else {}

    # ---------- validate and add all provided session ids ----------
    session_ids_provided = []

    user_id = _validate_and_trim_entity_id(user_id, "user_id")
    agent_id = _validate_and_trim_entity_id(agent_id, "agent_id")
    run_id = _validate_and_trim_entity_id(run_id, "run_id")

    if user_id:
        base_metadata_template["user_id"] = user_id
        effective_query_filters["user_id"] = user_id
        session_ids_provided.append("user_id")

    if agent_id:
        base_metadata_template["agent_id"] = agent_id
        effective_query_filters["agent_id"] = agent_id
        session_ids_provided.append("agent_id")

    if run_id:
        base_metadata_template["run_id"] = run_id
        effective_query_filters["run_id"] = run_id
        session_ids_provided.append("run_id")

    if not session_ids_provided:
        raise ValueError(
            "At least one of 'user_id', 'agent_id', or 'run_id' must be provided. "
            "Please provide at least one identifier to scope the memory operation."
        )

    # ---------- optional actor filter ----------
    resolved_actor_id = actor_id or effective_query_filters.get("actor_id")
    if resolved_actor_id:
        effective_query_filters["actor_id"] = resolved_actor_id

    return base_metadata_template, effective_query_filters


class MemoryStore:
    """mem0-compatible memory CRUD facade over NeatMem's own storage parts.

    Covers exactly the infer=False surface NeatMem uses: direct message
    writes, get/get_all/update/delete/delete_all/history/reset, plus mem0's
    entity-linking side effects on the update/delete paths. There is no LLM,
    no extraction pipeline, no reranker, and no telemetry here by design.
    """

    def __init__(
        self,
        *,
        vector_store: Qdrant,
        embedding_model,
        history_db_path: str,
    ):
        """
        Args:
            vector_store: Initialized NeatMem Qdrant vector store wrapper.
            embedding_model: Embedder with mem0's embed(text, memory_action)
                signature (see neatmem.embeddings).
            history_db_path: SQLite path for the memory-change history db.
        """
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.db = SQLiteHistoryManager(history_db_path)
        self.history_db_path = history_db_path
        self.collection_name = vector_store.collection_name

        # Entity store is initialized lazily on first use
        self._entity_store = None

    # ------------------------------------------------------------------
    # Entity store (lazy, shares the main store's Qdrant client)
    # ------------------------------------------------------------------

    @property
    def entity_store(self):
        """Lazily initialize entity store on first use."""
        if self._entity_store is None:
            self._entity_store = Qdrant(
                collection_name=f"{self.collection_name}_entities",
                embedding_model_dims=self.vector_store.embedding_model_dims,
                # Share the existing client to avoid RocksDB lock contention
                # when using embedded mode (path=...).
                client=self.vector_store.client,
                on_disk=self.vector_store.on_disk,
            )
        return self._entity_store

    def _upsert_entity(self, entity_text, entity_type, memory_id, filters):
        """Upsert an entity into the entity store, linking it to a memory."""
        try:
            entity_embedding = self.embedding_model.embed(entity_text, "add")
            search_filters = {k: v for k, v in filters.items() if k in ("user_id", "agent_id", "run_id") and v}

            existing = self.entity_store.search(
                query=entity_text,
                vectors=entity_embedding,
                top_k=1,
                filters=search_filters,
            )

            if existing and existing[0].score >= 0.95:
                # Update existing entity's linked_memory_ids
                match = existing[0]
                payload = match.payload or {}
                linked_ids = payload.get("linked_memory_ids", [])
                if memory_id not in linked_ids:
                    linked_ids.append(memory_id)
                    payload["linked_memory_ids"] = linked_ids
                    self.entity_store.update(
                        vector_id=match.id,
                        vector=None,
                        payload=payload,
                    )
            else:
                # Create new entity
                entity_id = str(uuid.uuid4())
                entity_payload = {
                    "data": entity_text,
                    "entity_type": entity_type,
                    "linked_memory_ids": [memory_id],
                    **{k: v for k, v in search_filters.items()},
                }
                self.entity_store.insert(
                    vectors=[entity_embedding],
                    ids=[entity_id],
                    payloads=[entity_payload],
                )
        except Exception as e:
            logger.warning(f"Entity upsert failed for '{entity_text}': {e}")

    def _remove_memory_from_entity_store(self, memory_id, filters):
        """Strip `memory_id` from every entity record scoped to `filters`.

        No-op if the entity store has never been initialized in this process.
        Errors on individual entities are swallowed at debug level; outer
        failures are swallowed at warning level so the primary delete/update
        path is never broken by entity cleanup.
        """
        if self._entity_store is None:
            return
        search_filters = {k: v for k, v in filters.items() if k in ("user_id", "agent_id", "run_id") and v}
        try:
            listed = self.entity_store.list(filters=search_filters, top_k=10000)
            rows = listed[0] if isinstance(listed, (list, tuple)) and listed and isinstance(listed[0], list) else listed
            for row in rows or []:
                try:
                    payload = getattr(row, "payload", None) or {}
                    linked = payload.get("linked_memory_ids", [])
                    if not isinstance(linked, list) or memory_id not in linked:
                        continue
                    remaining = [mid for mid in linked if mid != memory_id]
                    if not remaining:
                        try:
                            self.entity_store.delete(vector_id=row.id)
                        except Exception as e:
                            logger.debug(f"Entity delete failed for id={row.id}: {e}")
                    else:
                        entity_text = payload.get("data")
                        if not isinstance(entity_text, str) or not entity_text:
                            logger.debug(f"Entity id={row.id} missing 'data'; skipping update during cleanup")
                            continue
                        try:
                            vec = self.embedding_model.embed(entity_text, "update")
                        except Exception as e:
                            logger.debug(f"Entity re-embed failed for '{entity_text}': {e}")
                            continue
                        new_payload = {**payload, "linked_memory_ids": remaining}
                        try:
                            self.entity_store.update(
                                vector_id=row.id,
                                vector=vec,
                                payload=new_payload,
                            )
                        except Exception as e:
                            logger.debug(f"Entity update failed for id={row.id}: {e}")
                except Exception as e:
                    logger.debug(f"Entity cleanup error: {e}")
        except Exception as e:
            logger.warning(f"Entity store cleanup failed for memory_id={memory_id}: {e}")

    def _link_entities_for_memory(self, memory_id, text, filters):
        """Extract entities from `text` and link them to `memory_id` in the
        entity store, scoped to `filters`. Non-fatal on any failure.
        """
        try:
            entities = extract_entities(text)
            if not entities:
                return
            seen = set()
            for entity_type, entity_text in entities:
                key = entity_text.strip().lower()
                if not key or key in seen:
                    continue
                seen.add(key)
                try:
                    self._upsert_entity(entity_text, entity_type, memory_id, filters)
                except Exception as e:
                    logger.debug(f"Entity link failed for '{entity_text}': {e}")
        except Exception as e:
            logger.warning(f"Entity linking failed for memory_id={memory_id}: {e}")

    # ------------------------------------------------------------------
    # add (infer=False only)
    # ------------------------------------------------------------------

    def add(
        self,
        messages,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        infer: bool = True,
        memory_type: Optional[str] = None,
        prompt: Optional[str] = None,
    ):
        """
        Create a new memory. Only infer=False (direct storage) is supported;
        NeatMem's extraction pipeline lives in neatmem.memory_add.

        Returns:
            dict: `{"results": [{"id", "memory", "event": "ADD", "actor_id", "role"}]}`
        """
        if infer:
            raise NotImplementedError(
                "MemoryStore only supports infer=False (direct storage). "
                "Use NeatMem's extraction pipeline (neatmem.memory_add.add_memories) "
                "for infer=True."
            )
        if memory_type is not None:
            raise NotImplementedError(
                "MemoryStore does not support memory_type (procedural memory requires an LLM)."
            )

        processed_metadata, effective_filters = _build_filters_and_metadata(
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            input_metadata=metadata,
        )

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        elif isinstance(messages, dict):
            messages = [messages]
        elif not isinstance(messages, list):
            raise ValueError("messages must be str, dict, or list[dict]")

        for msg in messages:
            if not isinstance(msg.get("content"), str):
                raise NotImplementedError(
                    "MemoryStore only supports plain text message content "
                    "(vision/multipart content requires an LLM)."
                )

        returned_memories = []
        for message_dict in messages:
            if (
                not isinstance(message_dict, dict)
                or message_dict.get("role") is None
                or message_dict.get("content") is None
            ):
                logger.warning(f"Skipping invalid message format: {message_dict}")
                continue

            if message_dict["role"] == "system":
                continue

            per_msg_meta = deepcopy(processed_metadata)
            per_msg_meta["role"] = message_dict["role"]

            actor_name = message_dict.get("name")
            if actor_name:
                per_msg_meta["actor_id"] = actor_name

            msg_content = message_dict["content"]
            msg_embeddings = self.embedding_model.embed(msg_content, "add")
            mem_id = self._create_memory(msg_content, {msg_content: msg_embeddings}, per_msg_meta)

            returned_memories.append(
                {
                    "id": mem_id,
                    "memory": msg_content,
                    "event": "ADD",
                    "actor_id": actor_name if actor_name else None,
                    "role": message_dict["role"],
                }
            )
        return {"results": returned_memories}

    # ------------------------------------------------------------------
    # read
    # ------------------------------------------------------------------

    def get(self, memory_id):
        """
        Retrieve a memory by ID.

        Args:
            memory_id (str): ID of the memory to retrieve.

        Returns:
            dict: Retrieved memory.
        """
        memory = self.vector_store.get(vector_id=memory_id)
        if not memory:
            return None

        result_item = {
            "id": memory.id,
            "memory": memory.payload.get("data", ""),
            "hash": memory.payload.get("hash"),
            "metadata": None,
            "score": None,
            "created_at": memory.payload.get("created_at"),
            "updated_at": memory.payload.get("updated_at"),
        }

        for key in _PROMOTED_PAYLOAD_KEYS:
            if key in memory.payload:
                result_item[key] = memory.payload[key]

        additional_metadata = {k: v for k, v in memory.payload.items() if k not in _CORE_AND_PROMOTED_KEYS}
        if additional_metadata:
            result_item["metadata"] = additional_metadata

        return result_item

    def get_all(
        self,
        *,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 20,
        **kwargs,
    ):
        """
        List all memories.

        Args:
            filters (dict): Filter dict containing entity IDs and optional metadata filters.
                Must contain at least one of: user_id, agent_id, run_id.
            top_k (int, optional): The maximum number of memories to return. Defaults to 20.

        Returns:
            dict: `{"results": [...]}`
        """
        # Reject top-level entity params - must use filters instead
        _reject_top_level_entity_params(kwargs, "get_all")

        # Validate top_k
        _validate_search_params(top_k=top_k)

        # Validate and trim entity IDs in filters
        effective_filters = dict(filters) if filters else {}
        if "user_id" in effective_filters:
            effective_filters["user_id"] = _validate_and_trim_entity_id(
                effective_filters["user_id"], "user_id"
            )
        if "agent_id" in effective_filters:
            effective_filters["agent_id"] = _validate_and_trim_entity_id(
                effective_filters["agent_id"], "agent_id"
            )
        if "run_id" in effective_filters:
            effective_filters["run_id"] = _validate_and_trim_entity_id(
                effective_filters["run_id"], "run_id"
            )

        # Validate filters contains at least one entity ID
        if not any(key in effective_filters for key in ("user_id", "agent_id", "run_id")):
            raise ValueError(
                "filters must contain at least one of: user_id, agent_id, run_id. "
                "Example: filters={'user_id': 'u1'}"
            )

        limit = top_k

        all_memories_result = self._get_all_from_vector_store(effective_filters, limit)

        return {"results": all_memories_result}

    def _get_all_from_vector_store(self, filters, limit):
        memories_result = self.vector_store.list(filters=filters, top_k=limit)

        # Handle different vector store return formats by inspecting first element
        if isinstance(memories_result, (tuple, list)) and len(memories_result) > 0:
            first_element = memories_result[0]

            # If first element is a container, unwrap one level
            if isinstance(first_element, (list, tuple)):
                actual_memories = first_element
            else:
                # First element is a memory object, structure is already flat
                actual_memories = memories_result
        else:
            actual_memories = memories_result

        formatted_memories = []
        for mem in actual_memories:
            memory_item_dict = {
                "id": mem.id,
                "memory": mem.payload.get("data", ""),
                "hash": mem.payload.get("hash"),
                "metadata": None,
                "created_at": mem.payload.get("created_at"),
                "updated_at": mem.payload.get("updated_at"),
            }

            for key in _PROMOTED_PAYLOAD_KEYS:
                if key in mem.payload:
                    memory_item_dict[key] = mem.payload[key]

            additional_metadata = {k: v for k, v in mem.payload.items() if k not in _CORE_AND_PROMOTED_KEYS}
            if additional_metadata:
                memory_item_dict["metadata"] = additional_metadata

            formatted_memories.append(memory_item_dict)

        return formatted_memories

    # ------------------------------------------------------------------
    # update / delete
    # ------------------------------------------------------------------

    def update(self, memory_id, data, metadata: Optional[Dict[str, Any]] = None):
        """
        Update a memory by ID.

        Args:
            memory_id (str): ID of the memory to update.
            data (str): New content to update the memory with.
            metadata (dict, optional): Metadata to update with the memory. Defaults to None.

        Returns:
            dict: Success message indicating the memory was updated.
        """
        existing_embeddings = {data: self.embedding_model.embed(data, "update")}

        self._update_memory(memory_id, data, existing_embeddings, metadata)
        return {"message": "Memory updated successfully!"}

    def delete(self, memory_id):
        """
        Delete a memory by ID.

        Args:
            memory_id (str): ID of the memory to delete.
        """
        existing_memory = self.vector_store.get(vector_id=memory_id)
        if existing_memory is None:
            raise ValueError(f"Memory with id {memory_id} not found")

        self._delete_memory(memory_id, existing_memory)
        return {"message": "Memory deleted successfully!"}

    def delete_all(self, user_id: Optional[str] = None, agent_id: Optional[str] = None, run_id: Optional[str] = None):
        """
        Delete all memories.

        Args:
            user_id (str, optional): ID of the user to delete memories for. Defaults to None.
            agent_id (str, optional): ID of the agent to delete memories for. Defaults to None.
            run_id (str, optional): ID of the run to delete memories for. Defaults to None.
        """
        filters: Dict[str, Any] = {}
        if user_id:
            filters["user_id"] = user_id
        if agent_id:
            filters["agent_id"] = agent_id
        if run_id:
            filters["run_id"] = run_id

        if not filters:
            raise ValueError(
                "At least one filter is required to delete all memories. If you want to delete all memories, use the `reset()` method."
            )

        # delete all vector memories and reset the collections
        memories = self.vector_store.list(filters=filters)[0]
        for memory in memories:
            self._delete_memory(memory.id)

        logger.info(f"Deleted {len(memories)} memories")

        return {"message": "Memories deleted successfully!"}

    def history(self, memory_id):
        """
        Get the history of changes for a memory by ID.

        Args:
            memory_id (str): ID of the memory to get history for.

        Returns:
            list: List of changes for the memory.
        """
        return self.db.get_history(memory_id)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _create_memory(self, data, existing_embeddings, metadata=None):
        logger.debug(f"Creating memory with {data=}")
        if data in existing_embeddings:
            embeddings = existing_embeddings[data]
        else:
            embeddings = self.embedding_model.embed(data, memory_action="add")
        memory_id = str(uuid.uuid4())
        new_metadata = deepcopy(metadata) if metadata is not None else {}
        new_metadata["data"] = data
        new_metadata["hash"] = hashlib.md5(data.encode()).hexdigest()
        if "created_at" not in new_metadata:
            new_metadata["created_at"] = datetime.now(timezone.utc).isoformat()
        new_metadata["updated_at"] = new_metadata["created_at"]
        new_metadata["text_lemmatized"] = lemmatize_for_bm25(data)

        self.vector_store.insert(
            vectors=[embeddings],
            ids=[memory_id],
            payloads=[new_metadata],
        )
        self.db.add_history(
            memory_id,
            None,
            data,
            "ADD",
            created_at=new_metadata.get("created_at"),
            updated_at=new_metadata.get("updated_at"),
            actor_id=new_metadata.get("actor_id"),
            role=new_metadata.get("role"),
        )
        return memory_id

    def _update_memory(self, memory_id, data, existing_embeddings, metadata=None):
        logger.info(f"Updating memory with {data=}")

        try:
            existing_memory = self.vector_store.get(vector_id=memory_id)
        except Exception:
            logger.error(f"Error getting memory with ID {memory_id} during update.")
            raise ValueError(f"Error getting memory with ID {memory_id}. Please provide a valid 'memory_id'")

        if existing_memory is None:
            raise ValueError(f"Memory with id {memory_id} not found. Please provide a valid 'memory_id'")

        prev_value = existing_memory.payload.get("data")

        new_metadata = deepcopy(metadata) if metadata is not None else {}

        new_metadata["data"] = data
        new_metadata["hash"] = hashlib.md5(data.encode()).hexdigest()
        new_metadata["text_lemmatized"] = lemmatize_for_bm25(data)
        new_metadata["created_at"] = existing_memory.payload.get("created_at")
        new_metadata["updated_at"] = datetime.now(timezone.utc).isoformat()

        # Preserve session identifiers from existing memory only if not provided in new metadata
        if "user_id" not in new_metadata and "user_id" in existing_memory.payload:
            new_metadata["user_id"] = existing_memory.payload["user_id"]
        if "agent_id" not in new_metadata and "agent_id" in existing_memory.payload:
            new_metadata["agent_id"] = existing_memory.payload["agent_id"]
        if "run_id" not in new_metadata and "run_id" in existing_memory.payload:
            new_metadata["run_id"] = existing_memory.payload["run_id"]
        if "actor_id" in existing_memory.payload:
            new_metadata["actor_id"] = existing_memory.payload["actor_id"]
        if "role" not in new_metadata and "role" in existing_memory.payload:
            new_metadata["role"] = existing_memory.payload["role"]

        if data in existing_embeddings:
            embeddings = existing_embeddings[data]
        else:
            embeddings = self.embedding_model.embed(data, "update")

        self.vector_store.update(
            vector_id=memory_id,
            vector=embeddings,
            payload=new_metadata,
        )
        logger.info(f"Updating memory with ID {memory_id=} with {data=}")

        self.db.add_history(
            memory_id,
            prev_value,
            data,
            "UPDATE",
            created_at=new_metadata["created_at"],
            updated_at=new_metadata["updated_at"],
            actor_id=new_metadata.get("actor_id"),
            role=new_metadata.get("role"),
        )

        # Entity-store cleanup: strip this memory's id from old-text entities,
        # then re-extract entities from the new text and link them back.
        session_filters = {k: new_metadata[k] for k in ("user_id", "agent_id", "run_id") if new_metadata.get(k)}
        self._remove_memory_from_entity_store(memory_id, session_filters)
        self._link_entities_for_memory(memory_id, data, session_filters)

        return memory_id

    def _delete_memory(self, memory_id, existing_memory=None):
        logger.info(f"Deleting memory with {memory_id=}")
        if existing_memory is None:
            existing_memory = self.vector_store.get(vector_id=memory_id)
            if existing_memory is None:
                raise ValueError(f"Memory with id {memory_id} not found. Please provide a valid 'memory_id'")
        prev_value = existing_memory.payload.get("data", "")
        created_at = _normalize_iso_timestamp_to_utc(existing_memory.payload.get("created_at"))
        updated_at = datetime.now(timezone.utc).isoformat()
        payload = existing_memory.payload or {}
        session_filters = {k: payload[k] for k in ("user_id", "agent_id", "run_id") if payload.get(k)}
        self.vector_store.delete(vector_id=memory_id)
        self.db.add_history(
            memory_id,
            prev_value,
            None,
            "DELETE",
            created_at=created_at,
            updated_at=updated_at,
            actor_id=existing_memory.payload.get("actor_id"),
            role=existing_memory.payload.get("role"),
            is_deleted=1,
        )

        # Entity-store cleanup: strip this memory's id from any entity records
        # that linked to it. Non-fatal — the helper swallows errors.
        self._remove_memory_from_entity_store(memory_id, session_filters)

        return memory_id

    def reset(self):
        """
        Reset the memory store by:
            Deletes the vector store collection
            Resets the database
            Recreates the vector store collection
        """
        logger.warning("Resetting all memories")

        if hasattr(self.db, "connection") and self.db.connection:
            self.db.connection.execute("DROP TABLE IF EXISTS history")
            self.db.connection.close()

        self.db = SQLiteHistoryManager(self.history_db_path)

        self.vector_store.reset()

        # Reset entity store if initialized
        if self._entity_store is not None:
            try:
                self._entity_store.reset()
            except Exception as e:
                logger.warning(f"Failed to reset entity store: {e}")
            self._entity_store = None

    def close(self):
        """Release resources held by this MemoryStore instance (SQLite connection)."""
        if hasattr(self, "db") and self.db is not None:
            self.db.close()
            self.db = None
