"""Memory usage feedback (plan: 20261005-citation-feedback-memory-importance).

Offline judgment of whether injected memories were used, feeding the
per-memory double counter that drives the eviction gate. Never on the
serving hot path — judgment is a batch process over the activity event
stream (see storage/activity.py).
"""
