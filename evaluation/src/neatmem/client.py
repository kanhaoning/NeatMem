"""NeatMem HTTP client — 对齐 mem0 MemoryClient 接口，用于 LOCOMO 评测"""

import os
import requests


class NeatMemClient:
    def __init__(self, base_url=None):
        self.base_url = base_url or os.getenv("NEATMEM_URL", "http://localhost:8790")

    def add(self, messages, user_id=None, metadata=None, custom_instructions=None, **kwargs):
        body = {"messages": messages, "infer": True}
        if user_id:
            body["user_id"] = user_id
        if metadata:
            body["metadata"] = metadata
        if custom_instructions:
            body["custom_instructions"] = custom_instructions
        resp = requests.post(f"{self.base_url}/v1/memories/", json=body, timeout=120)
        resp.raise_for_status()
        return resp.json()

    def search(self, query, user_id=None, top_k=10, **kwargs):
        body = {"query": query, "top_k": top_k}
        if user_id:
            body["filters"] = {"user_id": user_id}
        resp = requests.post(f"{self.base_url}/v2/memories/search/", json=body, timeout=60)
        resp.raise_for_status()
        return resp.json().get("results", [])

    def delete_all(self, user_id=None, **kwargs):
        params = {}
        if user_id:
            params["user_id"] = user_id
        resp = requests.delete(f"{self.base_url}/v1/memories/", params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()
