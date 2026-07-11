"""No-op message store for external-injection mode (MESSAGE_STORE_BACKEND=none)."""

from typing import Any, Dict, List, Optional

from neatmem.storage.message.base import AbstractMessageStore


class NoOpMessageStore(AbstractMessageStore):
    """No-op message store.

    All write methods are silent no-ops.  All read methods return empty results.
    Use this when the host application has its own message db and only needs
    NeatMem for extract + vector storage (``MESSAGE_STORE_BACKEND=none``).  The
    host provides ``last_k_messages`` via the request body so the extraction
    prompt still gets conversational context.
    """

    def __init__(self, *, extract_last_k: int = 10):
        self.extract_last_k = extract_last_k

    def save_messages(
        self,
        messages: List[Dict[str, Any]],
        filters: Dict[str, Any],
    ) -> None:
        """No-op: messages are not persisted."""
        pass

    def get_last_messages(
        self,
        filters: Dict[str, Any],
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """No-op: always returns empty list.

        In external-injection mode, extract context comes from the
        ``last_k_messages`` request field, not from the store.
        """
        return []

    def query_messages(
        self,
        filters: Dict[str, Any],
        *,
        content_like: Optional[str] = None,
        after: Optional[str] = None,
        before: Optional[str] = None,
        roles: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0,
        order: str = "desc",
    ) -> List[Dict[str, Any]]:
        """No-op: always returns empty list."""
        return []

    def count_messages(
        self,
        filters: Dict[str, Any],
        *,
        content_like: Optional[str] = None,
        after: Optional[str] = None,
        before: Optional[str] = None,
        roles: Optional[List[str]] = None,
    ) -> int:
        """No-op: always returns 0."""
        return 0

    def list_sessions(
        self,
        filters: Dict[str, Any],
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """No-op: always returns empty list."""
        return []

    def delete_messages(self, filters: Dict[str, Any]) -> int:
        """No-op: always returns 0."""
        return 0

    def reset(self) -> None:
        """No-op: nothing to reset."""
        pass

    def close(self) -> None:
        """No-op: no resources to release."""
        pass
