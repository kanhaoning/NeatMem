"""Unit tests for MEMORY_FEEDBACK_* env parsing (plan 20261005 §5.7).

Same reload pattern as test_config_client_policy.py — config reads env at
import time, so assertions run inside the reload context.
"""

import importlib
import os
from contextlib import contextmanager

_KEYS = (
    "MEMORY_FEEDBACK_ENABLED",
    "MEMORY_FEEDBACK_JUDGE_MODEL",
    "MEMORY_FEEDBACK_JUDGE_BASE_URL",
    "MEMORY_FEEDBACK_JUDGE_API_KEY",
    "MEMORY_FEEDBACK_JUDGE_PROMPT",
    "MEMORY_FEEDBACK_EVICTION_ENABLED",
    "MEMORY_FEEDBACK_EVICTION_MIN_INJECTIONS",
    "ACTIVITY_DB_PATH",
)


@contextmanager
def _reload_config(env: dict):
    saved = {k: os.environ.get(k) for k in _KEYS}
    try:
        for k in _KEYS:
            os.environ.pop(k, None)
        os.environ.update(env)
        import neatmem.config as config
        yield importlib.reload(config)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        import neatmem.config as config
        importlib.reload(config)


def test_defaults_off():
    with _reload_config({}) as config:
        assert config.MEMORY_FEEDBACK_ENABLED is False
        assert config.MEMORY_FEEDBACK_EVICTION_ENABLED is False
        assert config.MEMORY_FEEDBACK_EVICTION_MIN_INJECTIONS == 10
        assert config.MEMORY_FEEDBACK_JUDGE_MODEL == ""
        assert config.MEMORY_FEEDBACK_JUDGE_PROMPT == ""
        assert config.ACTIVITY_DB_PATH.endswith("activity.db")


def test_enable_flags():
    with _reload_config({
        "MEMORY_FEEDBACK_ENABLED": "true",
        "MEMORY_FEEDBACK_EVICTION_ENABLED": "1",
        "MEMORY_FEEDBACK_EVICTION_MIN_INJECTIONS": "20",
    }) as config:
        assert config.MEMORY_FEEDBACK_ENABLED is True
        assert config.MEMORY_FEEDBACK_EVICTION_ENABLED is True
        assert config.MEMORY_FEEDBACK_EVICTION_MIN_INJECTIONS == 20


def test_judge_triple_override():
    with _reload_config({
        "MEMORY_FEEDBACK_JUDGE_MODEL": "judge-x",
        "MEMORY_FEEDBACK_JUDGE_BASE_URL": "http://localhost:9999/v1",
        "MEMORY_FEEDBACK_JUDGE_API_KEY": "sk-test",
    }) as config:
        assert config.MEMORY_FEEDBACK_JUDGE_MODEL == "judge-x"
        assert config.MEMORY_FEEDBACK_JUDGE_BASE_URL == "http://localhost:9999/v1"
        assert config.MEMORY_FEEDBACK_JUDGE_API_KEY == "sk-test"
