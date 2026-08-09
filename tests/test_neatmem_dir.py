"""NEATMEM_DIR data-root resolution tests.

config.py reads env at import time, so each case reloads the module with a
monkeypatched environment. No network access.
"""

import importlib
import os

import pytest

import neatmem.config as config

PATH_ENVS = ["NEATMEM_DIR", "MEM0_DIR", "QDRANT_PATH",
             "HISTORY_DB_PATH", "MEMORY_HISTORY_DB_PATH"]


@pytest.fixture
def clean_env(monkeypatch):
    for var in PATH_ENVS:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def reload_config():
    return importlib.reload(config)


def test_neatmem_dir_governs_all_defaults(clean_env, tmp_path):
    clean_env.setenv("NEATMEM_DIR", str(tmp_path))
    cfg = reload_config()
    assert cfg.QDRANT_PATH == str(tmp_path / "qdrant")
    assert cfg.HISTORY_DB_PATH == str(tmp_path / "messages.db")
    assert cfg.MEMORY_HISTORY_DB_PATH == str(tmp_path / "history.db")


def test_per_path_env_overrides_neatmem_dir(clean_env, tmp_path):
    clean_env.setenv("NEATMEM_DIR", str(tmp_path))
    clean_env.setenv("QDRANT_PATH", "/elsewhere/qdrant")
    clean_env.setenv("HISTORY_DB_PATH", "/elsewhere/messages.db")
    clean_env.setenv("MEMORY_HISTORY_DB_PATH", "/elsewhere/history.db")
    cfg = reload_config()
    assert cfg.QDRANT_PATH == "/elsewhere/qdrant"
    assert cfg.HISTORY_DB_PATH == "/elsewhere/messages.db"
    assert cfg.MEMORY_HISTORY_DB_PATH == "/elsewhere/history.db"


def test_default_root_is_home_dot_neatmem(clean_env):
    cfg = reload_config()
    home = os.path.expanduser("~")
    assert cfg.QDRANT_PATH == os.path.join(home, ".neatmem", "qdrant")
    assert cfg.MEMORY_HISTORY_DB_PATH == os.path.join(home, ".neatmem", "history.db")


def test_mem0_dir_legacy_fallback(clean_env, tmp_path):
    clean_env.setenv("MEM0_DIR", str(tmp_path))
    cfg = reload_config()
    assert cfg.QDRANT_PATH == str(tmp_path / "qdrant")
    # NEATMEM_DIR wins over MEM0_DIR when both are set
    clean_env.setenv("NEATMEM_DIR", "/priority")
    cfg = reload_config()
    assert cfg.QDRANT_PATH == "/priority/qdrant"


@pytest.fixture(autouse=True)
def restore_config():
    yield
    reload_config()
