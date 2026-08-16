"""HTTP backend for the NeatMem server (mem0-compatible REST, httpx direct).

NeatMem server endpoints (neatmem/main.py):
  add     POST /v1/memories/
  search  POST /v2/memories/search/   (rerank field = per-request LLM_RERANK override)
  list    POST /v2/memories/          (page/page_size as query params)
  update  PUT  /v1/memories/{id}/
  delete  DELETE /v1/memories/{id}/
  add_messages  POST /v1/messages/add/  (queue mode: store-only ingest)

The server currently performs no auth; the api_key is accepted for forward
compatibility and sent as `Authorization: Token <key>` (mem0 convention).
"""

from __future__ import annotations

from typing import Any


class MessagesAddUnsupportedError(Exception):
    """Server does not have /v1/messages/add/ (pre-queue-mode server)."""


class MessagesFlushUnsupportedError(Exception):
    """Server does not have /v1/messages/flush/ (pre-queue-mode server)."""


class NeatMemBackend:
    """Unified interface over the NeatMem REST server."""

    def __init__(self, base_url: str, api_key: str = ""):
        import httpx

        headers = {}
        if api_key:
            headers["Authorization"] = f"Token {api_key}"
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=httpx.Timeout(120.0, connect=5.0),
        )

    def search(self, query: str, *, filters: dict, top_k: int = 10, rerank: bool = False) -> list[dict]:
        resp = self._http.post("/v2/memories/search/", json={
            "query": query,
            "filters": filters,
            "top_k": top_k,
            "rerank": rerank,
        })
        resp.raise_for_status()
        return resp.json().get("results", [])

    def get_all(self, *, filters: dict, page: int = 1, page_size: int = 100) -> dict:
        resp = self._http.post(
            "/v2/memories/",
            params={"page": page, "page_size": page_size},
            json={"filters": filters},
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        return {"results": results, "count": len(results)}

    def add(
        self,
        messages: list,
        *,
        user_id: str,
        agent_id: str,
        infer: bool = False,
        metadata: dict | None = None,
    ) -> dict:
        payload: dict[str, Any] = {
            "messages": messages,
            "user_id": user_id,
            "agent_id": agent_id,
            "infer": infer,
        }
        if metadata:
            payload["metadata"] = metadata
        resp = self._http.post("/v1/memories/", json=payload)
        resp.raise_for_status()
        return resp.json()

    def add_messages(
        self,
        messages: list,
        *,
        user_id: str,
        agent_id: str | None = None,
        run_id: str | None = None,
    ) -> dict:
        """Store-only ingest (queue mode). Extraction is scheduled server-side.

        Raises MessagesAddUnsupportedError when the server predates the queue
        mode endpoints (404), so the caller can fall back to infer add.
        """
        payload: dict[str, Any] = {"messages": messages, "user_id": user_id}
        if agent_id:
            payload["agent_id"] = agent_id
        if run_id:
            payload["run_id"] = run_id
        resp = self._http.post("/v1/messages/add/", json=payload)
        if resp.status_code == 404:
            raise MessagesAddUnsupportedError(
                "server has no /v1/messages/add/ endpoint"
            )
        resp.raise_for_status()
        return resp.json()

    def flush(
        self,
        *,
        user_id: str,
        agent_id: str | None = None,
        run_id: str | None = None,
    ) -> dict:
        """Force immediate extraction of pending messages for the scope.

        Raises MessagesFlushUnsupportedError on pre-queue-mode servers (404).
        """
        payload: dict[str, Any] = {"user_id": user_id}
        if agent_id:
            payload["agent_id"] = agent_id
        if run_id:
            payload["run_id"] = run_id
        resp = self._http.post("/v1/messages/flush/", json=payload)
        if resp.status_code == 404:
            raise MessagesFlushUnsupportedError(
                "server has no /v1/messages/flush/ endpoint"
            )
        resp.raise_for_status()
        return resp.json()

    def update(self, memory_id: str, text: str) -> dict:
        resp = self._http.put(f"/v1/memories/{memory_id}/", json={"text": text})
        resp.raise_for_status()
        return resp.json()

    def delete(self, memory_id: str) -> dict:
        resp = self._http.delete(f"/v1/memories/{memory_id}/")
        resp.raise_for_status()
        return resp.json()

    def ping(self) -> bool:
        try:
            resp = self._http.get("/v1/ping/", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False

    def close(self) -> None:
        self._http.close()
