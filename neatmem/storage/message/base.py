"""
Message store abstraction for NeatMem.

This module defines the interface that any message-history backend must
implement. The default backend is SQLite (see neatmem.storage.message.sqlite),
but the interface is kept small so that downstream applications can plug in
PostgreSQL, Redis, or any other store without changing the rest of NeatMem.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class AbstractMessageStore(ABC):
    """Abstract message store.

    A message store persists raw conversation messages for a given scope and
    provides methods to read them back for extraction context and for external
    querying.

    Scope is built from the following dimensions, all optional:
        app_id, user_id, agent_id, run_id
    A message belongs to exactly one scope, which is the non-empty combination
    of the dimensions provided when it was saved.
    """

    @abstractmethod
    def save_messages(
        self,
        messages: List[Dict[str, Any]],
        filters: Dict[str, Any],
    ) -> None:
        """Save raw messages for the scope derived from ``filters``.

        Args:
            messages: List of message dicts, each containing at least ``role``
                and ``content``. ``name`` is optional.
            filters: Dict that may contain ``app_id``, ``user_id``, ``agent_id``,
                ``run_id``. At least one should be provided; implementations may
                skip saving when the scope is empty.
        """
        ...

    @abstractmethod
    def get_last_messages(
        self,
        filters: Dict[str, Any],
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Return the most recent ``limit`` messages for the scope, oldest first.

        If ``limit`` is omitted, implementations should fall back to a
        service-level default (e.g. ``self.extract_last_k``).
        """
        ...

    @abstractmethod
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
        """Query messages with filtering and pagination.

        Args:
            filters: Scope filters (app_id/user_id/agent_id/run_id).
                At least one should be provided to avoid full-table scans.
            content_like: If provided, only include messages whose content
                matches this substring (SQL ``LIKE %x%``).
            after: ISO timestamp; only messages with ``created_at >= after``
                (closed interval).
            before: ISO timestamp; only messages with ``created_at <= before``
                (closed interval).
            roles: If provided, only include messages whose role is in this list.
            limit: Maximum number of messages to return.
            offset: Number of messages to skip.
            order: ``asc`` for chronological, ``desc`` for reverse chronological.
        """
        ...

    @abstractmethod
    def count_messages(
        self,
        filters: Dict[str, Any],
        *,
        content_like: Optional[str] = None,
        after: Optional[str] = None,
        before: Optional[str] = None,
        roles: Optional[List[str]] = None,
    ) -> int:
        """Count messages matching the filters."""
        ...

    @abstractmethod
    def list_sessions(
        self,
        filters: Dict[str, Any],
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List distinct ``run_id`` values for the scope, with last-active time.

        Args:
            filters: Scope filters (app_id/user_id/agent_id). ``run_id`` is not
                used as a filter here (it is the grouped dimension).
            limit: Maximum number of sessions to return.
            offset: Number of sessions to skip.

        Returns:
            List of dicts with ``run_id`` and ``last_active_at`` keys.
        """
        ...

    @abstractmethod
    def delete_messages(self, filters: Dict[str, Any]) -> int:
        """Delete messages matching the scope filters.

        Args:
            filters: Scope filters. At least one of app_id/user_id/agent_id/
                run_id must be provided to prevent accidental full-table deletes.

        Returns:
            Number of messages deleted.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Drop and recreate the messages table (full reset)."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Release any resources held by the store."""
        ...
