"""Unit tests for client-plugin policy env parsing (INJECT_TIMING / MIN_QUERY_CHARS).

The /v1/config/ endpoint itself is a thin read of these globals; a live smoke
check of the endpoint happens in the plugin e2e, not here (importing
neatmem.main boots the whole app).
"""

import importlib
import os
from contextlib import contextmanager

import pytest


@contextmanager
def _reload_config(env: dict):
    """Reload neatmem.config under a temporary env, restoring state on exit.

    importlib.reload mutates the module in place, so assertions must run
    inside the with block (before the cleanup reload restores the module).
    """
    keys = ("INJECT_TIMING", "MIN_QUERY_CHARS")
    saved = {k: os.environ.get(k) for k in keys}
    try:
        for k in keys:
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


def test_defaults():
    with _reload_config({}) as config:
        assert config.INJECT_TIMING == "first"
        assert config.MIN_QUERY_CHARS == 20


def test_env_override():
    with _reload_config({"INJECT_TIMING": "every", "MIN_QUERY_CHARS": "50"}) as config:
        assert config.INJECT_TIMING == "every"
        assert config.MIN_QUERY_CHARS == 50


def test_inject_timing_values():
    for value in ("off", "first", "every"):
        with _reload_config({"INJECT_TIMING": value}) as config:
            assert config.INJECT_TIMING == value


def test_min_query_chars_invalid():
    with pytest.raises(ValueError), _reload_config({"MIN_QUERY_CHARS": "abc"}):
        pass
