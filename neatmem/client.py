"""mem0-compatible remote client for a NeatMem server.

Syntax-layer compatible with mem0-202606 ``Memory`` / ``MemoryClient``:
same class name, method signatures, return shapes, and validation messages.
Behavior-layer differences (rerank default, multi-signal search, dedup) are
intentional and documented in README.
"""

import os
from typing import Any, Dict, List, Optional

import httpx

from neatmem.exceptions import NeatMemValidationError

_FILTER_REQUIRED_MSG = (
    "filters must contain at least one of: user_id, agent_id, run_id. "
    "Example: filters={'user_id': 'u1'}"
)


def _validate_search_params(threshold: Optional[float] = None, top_k: Optional[int] = None) -> None:
    """Mirror mem0's _validate_search_params (memory/main.py:170), raising NeatMemValidationError."""
    if threshold is not None:
        if not isinstance(threshold, (int, float)):
            raise NeatMemValidationError("threshold must be a valid number")
        if threshold < 0 or threshold > 1:
            raise NeatMemValidationError(
                f"Invalid threshold: {threshold}. Must be between 0 and 1 (inclusive)."
            )
    if top_k is not None:
        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise NeatMemValidationError("top_k must be a valid integer")
        if top_k < 0:
            raise NeatMemValidationError(
                f"Invalid top_k: {top_k}. Must be a non-negative integer."
            )


def _validate_filters(filters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Require at least one session id in filters, same rule as mem0 OSS Memory."""
    if not filters or not any(k in filters for k in ("user_id", "agent_id", "run_id")):
        raise NeatMemValidationError(_FILTER_REQUIRED_MSG)
    return filters


_TOP_LEVEL_ENTITY_PARAMS = {"user_id", "agent_id", "run_id", "app_id"}


def _reject_top_level_entity_params(kwargs: Dict[str, Any], method_name: str) -> None:
    """Reject top-level entity params (user_id, agent_id, run_id, app_id).

    Same as mem0 OSS Memory: users must pass these inside ``filters`` instead
    of as keyword arguments. Raises NeatMemValidationError with a hint.
    """
    for param in _TOP_LEVEL_ENTITY_PARAMS:
        if param in kwargs:
            raise NeatMemValidationError(
                f"'{param}' is not a valid parameter for {method_name}(). "
                f"Use filters={{{param!r}: ...}} instead."
            )


class MemoryClient:
    """Remote client for a NeatMem server, signature-compatible with mem0 MemoryClient.

    Args:
        api_key: Accepted for signature compatibility and sent as an
            ``Authorization: Token`` header; the NeatMem server ignores it.
            Falls back to the NEATMEM_API_KEY env var.
        host: Base URL of the NeatMem server. Falls back to the NEATMEM_URL
            env var, then ``http://localhost:8790``.
        org_id: mem0 Platform parameter; passing a non-None value raises
            NotImplementedError.
        project_id: Same as org_id.
        timeout: HTTP timeout in seconds (NeatMem extension; mem0 has no such
            parameter, adding it does not break compatibility).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        host: Optional[str] = None,
        org_id: Optional[str] = None,
        project_id: Optional[str] = None,
        timeout: float = 300.0,
    ):
        if org_id is not None or project_id is not None:
            raise NotImplementedError(
                "org_id/project_id are mem0 Platform parameters and are not supported by NeatMem."
            )
        self.api_key = api_key or os.getenv("NEATMEM_API_KEY")
        self.host = (host or os.getenv("NEATMEM_URL") or "http://localhost:8790").rstrip("/")

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Token {self.api_key}"
        self.client = httpx.Client(base_url=self.host, headers=headers, timeout=timeout)
        self.messages = MessagesAPI(self)

    def _checked_json(self, response: httpx.Response, memory_id: Optional[str] = None) -> Any:
        if response.status_code == 404 and memory_id is not None:
            raise NeatMemValidationError(f"Memory with id {memory_id} not found")
        response.raise_for_status()
        return response.json()

    # --- mem0-compatible surface (aligned with mem0-202606 Memory) ---

    def add(
        self,
        messages,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[Any] = None,
        infer: bool = True,
        memory_type: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create new memories. Validation behavior mirrors mem0 OSS Memory.add."""
        if timestamp is not None:
            raise NotImplementedError(
                "timestamp is a mem0 Platform-only parameter and is not supported by NeatMem."
            )
        if memory_type is not None:
            raise NotImplementedError(
                "memory_type (procedural memory) is not supported by NeatMem."
            )
        if not any([user_id, agent_id, run_id]):
            # mem0 VALIDATION_001
            raise NeatMemValidationError(
                "At least one of 'user_id', 'agent_id', or 'run_id' must be provided."
            )

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        elif isinstance(messages, dict):
            messages = [messages]
        elif not isinstance(messages, list):
            # mem0 VALIDATION_003
            raise NeatMemValidationError("messages must be str, dict, or list[dict]")

        payload: Dict[str, Any] = {"messages": messages, "infer": infer}
        for key, value in {
            "user_id": user_id,
            "agent_id": agent_id,
            "run_id": run_id,
            "metadata": metadata,
            "prompt": prompt,
        }.items():
            if value is not None:
                payload[key] = value

        response = self.client.post("/v1/memories/", json=payload)
        return self._checked_json(response)

    def search(
        self,
        query: str,
        *,
        top_k: int = 20,
        filters: Optional[Dict[str, Any]] = None,
        threshold: float = 0.1,
        rerank: bool = False,
        explain: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Search memories. filters=None is rejected client-side, same as mem0 OSS."""
        _reject_top_level_entity_params(kwargs, "search")
        if explain:
            raise NotImplementedError(
                "explain=True (score_details) is not supported by NeatMem."
            )
        _validate_search_params(threshold=threshold, top_k=top_k)
        filters = _validate_filters(filters)

        payload = {
            "query": query,
            "top_k": top_k,
            "threshold": threshold,
            "filters": filters,
            "rerank": rerank,
        }
        response = self.client.post("/v2/memories/search/", json=payload)
        return self._checked_json(response)

    def get(self, memory_id: str) -> Dict[str, Any]:
        """Retrieve a specific memory by ID."""
        response = self.client.get(f"/v1/memories/{memory_id}/")
        return self._checked_json(response, memory_id=memory_id)

    def get_all(self, *, filters: Optional[Dict[str, Any]] = None, top_k: int = 20, **kwargs: Any) -> Dict[str, Any]:
        """List memories. filters=None is rejected client-side, same as mem0 OSS."""
        _reject_top_level_entity_params(kwargs, "get_all")
        _validate_search_params(top_k=top_k)
        filters = _validate_filters(filters)

        response = self.client.post(
            "/v2/memories/", json={"filters": filters, "page_size": top_k}
        )
        return self._checked_json(response)

    def update(self, memory_id: str, data: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Update a memory's text (and optionally metadata). Return shape normalized to mem0."""
        payload: Dict[str, Any] = {"text": data}
        if metadata is not None:
            payload["metadata"] = metadata
        response = self.client.put(f"/v1/memories/{memory_id}/", json=payload)
        self._checked_json(response, memory_id=memory_id)
        return {"message": "Memory updated successfully!"}

    def delete(self, memory_id: str) -> Dict[str, Any]:
        """Delete a memory by ID. Return shape normalized to mem0."""
        response = self.client.delete(f"/v1/memories/{memory_id}/")
        self._checked_json(response, memory_id=memory_id)
        return {"message": "Memory deleted successfully!"}

    def delete_all(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Delete all memories matching at least one session id (same rule as mem0)."""
        if not any([user_id, agent_id, run_id]):
            raise NeatMemValidationError(
                "At least one filter is required to delete all memories. "
                "If you want to delete all memories, use the `reset()` method."
            )
        params = {k: v for k, v in {
            "user_id": user_id,
            "agent_id": agent_id,
            "run_id": run_id,
        }.items() if v is not None}
        response = self.client.delete("/v1/memories/", params=params)
        return self._checked_json(response)

    def history(self, memory_id: str) -> List[Dict[str, Any]]:
        """Get the change history of a memory. Returns a bare list, same as mem0."""
        response = self.client.get(f"/v1/memories/{memory_id}/history/")
        data = self._checked_json(response, memory_id=memory_id)
        return data.get("results", [])

    def reset(self) -> Dict[str, Any]:
        """Delete ALL memories on the server. Message history is reset separately
        via ``client.messages.reset()``."""
        response = self.client.post("/v1/reset/")
        return self._checked_json(response)

    def chat(self, *args, **kwargs):
        """Not implemented (mem0 OSS does not implement it either)."""
        raise NotImplementedError("chat is not implemented (mem0 OSS does not implement it either).")

    # --- NeatMem extension surface (not part of the compatibility promise) ---

    def ping(self) -> Dict[str, Any]:
        """Health check."""
        response = self.client.get("/v1/ping/")
        return self._checked_json(response)

    def delete_entity(self, entity_type: str, entity_id: str) -> Dict[str, Any]:
        """Delete all memories belonging to an entity (user/agent/app/run)."""
        response = self.client.delete(f"/v2/entities/{entity_type}/{entity_id}/")
        return self._checked_json(response)

    # --- Queue-mode batching (NeatMem extension; mem0 has no equivalent) ---
    # Store-only ingest + cursor-driven batch scheduling. Extraction itself
    # still goes through the regular add path on the server; these methods
    # only drive the queue mechanics.

    def add_messages(
        self,
        messages,
        *,
        user_id: str,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        app_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Store messages without extraction (POST /v1/messages/add/).

        Returns {"results": [{"message_id", "seq"}], "count": N}.
        """
        if not user_id:
            raise NeatMemValidationError("user_id is required for add_messages().")
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        elif isinstance(messages, dict):
            messages = [messages]
        elif not isinstance(messages, list):
            raise NeatMemValidationError("messages must be str, dict, or list[dict]")

        payload: Dict[str, Any] = {"messages": messages, "user_id": user_id}
        for key, value in {
            "agent_id": agent_id,
            "run_id": run_id,
            "app_id": app_id,
        }.items():
            if value is not None:
                payload[key] = value
        response = self.client.post("/v1/messages/add/", json=payload)
        return self._checked_json(response)

    def get_next_batch(
        self,
        *,
        user_id: str,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        store: str = "vector",
    ) -> Dict[str, Any]:
        """Fetch the next batch due for extraction (POST /v1/messages/next-batch/).

        Returns {"message_ids": [...], "seqs": [...], "pending_count": N}
        (message IDs/seqs only, no content). Empty lists when no batch is due.
        """
        if not user_id:
            raise NeatMemValidationError("user_id is required for get_next_batch().")
        payload: Dict[str, Any] = {"user_id": user_id, "store": store}
        for key, value in {"agent_id": agent_id, "run_id": run_id}.items():
            if value is not None:
                payload[key] = value
        response = self.client.post("/v1/messages/next-batch/", json=payload)
        return self._checked_json(response)

    def mark_batch_processed(
        self,
        *,
        user_id: str,
        last_processed_seq: int,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        store: str = "vector",
    ) -> Dict[str, Any]:
        """Advance the extraction cursor (POST /v1/messages/mark-processed/).

        Call only after the batch was successfully extracted. The server
        rejects regression (seq not greater than the current cursor) with 409.
        """
        if not user_id:
            raise NeatMemValidationError("user_id is required for mark_batch_processed().")
        payload: Dict[str, Any] = {
            "user_id": user_id,
            "last_processed_seq": last_processed_seq,
            "store": store,
        }
        for key, value in {"agent_id": agent_id, "run_id": run_id}.items():
            if value is not None:
                payload[key] = value
        response = self.client.post("/v1/messages/mark-processed/", json=payload)
        return self._checked_json(response)


class MessagesAPI:
    """Raw message-history access (NeatMem extension; mem0 has no equivalent).

    Responses are passed through without shape normalization.
    """

    def __init__(self, owner: MemoryClient):
        self._owner = owner

    def _post(self, path: str, payload: Dict[str, Any]) -> Any:
        payload = {k: v for k, v in payload.items() if v is not None}
        response = self._owner.client.post(path, json=payload)
        response.raise_for_status()
        return response.json()

    def query(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        app_id: Optional[str] = None,
        roles: Optional[List[str]] = None,
        content_like: Optional[str] = None,
        after: Optional[str] = None,
        before: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        order: str = "desc",
    ) -> Dict[str, Any]:
        """Query stored raw messages with filters."""
        return self._post("/v1/messages/query/", {
            "user_id": user_id,
            "agent_id": agent_id,
            "run_id": run_id,
            "app_id": app_id,
            "roles": roles,
            "content_like": content_like,
            "after": after,
            "before": before,
            "limit": limit,
            "offset": offset,
            "order": order,
        })

    def sessions(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        app_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """List sessions that have stored messages."""
        return self._post("/v1/messages/sessions/", {
            "user_id": user_id,
            "agent_id": agent_id,
            "app_id": app_id,
            "limit": limit,
            "offset": offset,
        })

    def delete(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        app_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Delete stored raw messages matching at least one filter."""
        return self._post("/v1/messages/delete/", {
            "user_id": user_id,
            "agent_id": agent_id,
            "run_id": run_id,
            "app_id": app_id,
        })

    def reset(self) -> Dict[str, Any]:
        """Delete ALL stored raw messages."""
        response = self._owner.client.post("/v1/messages/reset/")
        response.raise_for_status()
        return response.json()
