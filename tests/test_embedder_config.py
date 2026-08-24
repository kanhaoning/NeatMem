"""EMBEDDER_* config resolution tests (2026-08-23 hard rename from EMBEDDING_*,
no compat layer) and the GRAPH_EMBEDDER_* fallback fix.

config.py reads env at import time, so each case reloads the module with a
monkeypatched environment; cwd is moved to tmp_path so the repo-root .env and
the default NEATMEM_DIR do not leak in. No network access (build_memory_store
is never called here).

Run: cd <repo root> && python -m pytest tests/test_embedder_config.py -q
"""

import importlib

import pytest

import neatmem.config as config

ALL_ENVS = [
    "EMBEDDER_PROVIDER", "EMBEDDER_MODEL", "EMBEDDER_BASE_URL",
    "EMBEDDER_API_KEY", "EMBEDDER_DIMS", "SILICONFLOW_API_KEY",
    "GRAPH_EMBEDDER_MODEL", "GRAPH_EMBEDDER_DIMS",
    "GRAPH_EMBEDDER_BASE_URL", "GRAPH_EMBEDDER_API_KEY",
    # legacy names, must be ignored after the rename
    "EMBEDDING_PROVIDER", "EMBEDDING_MODEL", "EMBEDDING_BASE_URL",
    "EMBEDDING_API_KEY", "EMBEDDING_DIMS",
    "GRAPH_EMBEDDING_MODEL", "GRAPH_EMBEDDING_DIMS",
    "GRAPH_EMBEDDING_BASE_URL", "GRAPH_EMBEDDING_API_KEY",
]

SILICONFLOW_URL = "https://api.siliconflow.cn/v1"


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NEATMEM_DIR", str(tmp_path / "data"))
    for var in ALL_ENVS:
        monkeypatch.delenv(var, raising=False)
    # config.py's load_dotenv() resolves from the module file location, not
    # cwd, so it would still find the repo-root .env; stub it out.
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    return monkeypatch


def reload_config():
    return importlib.reload(config)


def test_defaults_byte_identical_to_pre_rename(clean_env):
    """No env set: every value must equal the pre-rename hardcoded defaults,
    so existing SiliconFlow-default users see zero behavior change."""
    cfg = reload_config()
    assert cfg.EMBEDDER_PROVIDER == "siliconflow"
    assert cfg.EMBEDDER_MODEL == "BAAI/bge-m3"
    assert cfg.EMBEDDER_BASE_URL == SILICONFLOW_URL
    assert cfg.EMBEDDER_BATCH_SIZE == 100
    assert cfg.EMBEDDER_API_KEY == ""
    assert cfg.EMBEDDER_DIMS is None
    assert cfg.GRAPH_EMBEDDER_MODEL == "BAAI/bge-m3"
    assert cfg.GRAPH_EMBEDDER_DIMS == 1024
    assert cfg.GRAPH_EMBEDDER_BASE_URL == SILICONFLOW_URL
    assert cfg.GRAPH_EMBEDDER_API_KEY == ""


def test_legacy_embedding_names_ignored(clean_env):
    """Old EMBEDDING_* names must NOT take effect (hard rename, loud fallback
    to defaults rather than silently honoring the old name)."""
    clean_env.setenv("EMBEDDING_PROVIDER", "openai")
    clean_env.setenv("EMBEDDING_MODEL", "text-embedding-3-large")
    clean_env.setenv("EMBEDDING_API_KEY", "sk-legacy")
    clean_env.setenv("GRAPH_EMBEDDING_API_KEY", "sk-legacy-graph")
    cfg = reload_config()
    assert cfg.EMBEDDER_PROVIDER == "siliconflow"
    assert cfg.EMBEDDER_MODEL == "BAAI/bge-m3"
    assert cfg.EMBEDDER_API_KEY == ""
    assert cfg.GRAPH_EMBEDDER_API_KEY == ""


def test_siliconflow_key_fallback_preserved(clean_env):
    clean_env.setenv("SILICONFLOW_API_KEY", "sk-sf")
    cfg = reload_config()
    assert cfg.EMBEDDER_API_KEY == "sk-sf"
    assert cfg.GRAPH_EMBEDDER_API_KEY == "sk-sf"


def test_embedder_key_wins_over_siliconflow_fallback(clean_env):
    clean_env.setenv("EMBEDDER_API_KEY", "sk-main")
    clean_env.setenv("SILICONFLOW_API_KEY", "sk-sf")
    cfg = reload_config()
    assert cfg.EMBEDDER_API_KEY == "sk-main"


def test_provider_preset_switch(clean_env):
    clean_env.setenv("EMBEDDER_PROVIDER", "openai")
    cfg = reload_config()
    assert cfg.EMBEDDER_BASE_URL == "https://api.openai.com/v1"
    # graph base_url follows the main embedder by default (fallback fix)
    assert cfg.GRAPH_EMBEDDER_BASE_URL == "https://api.openai.com/v1"


def test_explicit_base_url_beats_preset(clean_env):
    clean_env.setenv("EMBEDDER_BASE_URL", "http://my-proxy/v1")
    cfg = reload_config()
    assert cfg.EMBEDDER_BASE_URL == "http://my-proxy/v1"
    assert cfg.GRAPH_EMBEDDER_BASE_URL == "http://my-proxy/v1"


def test_graph_follows_main_embedder(clean_env):
    """The fix: GRAPH_EMBEDDER_* default to the main embedder config instead
    of hardcoded SiliconFlow values."""
    clean_env.setenv("EMBEDDER_MODEL", "custom-embed")
    clean_env.setenv("EMBEDDER_BASE_URL", "http://my-proxy/v1")
    clean_env.setenv("EMBEDDER_API_KEY", "sk-main")
    cfg = reload_config()
    assert cfg.GRAPH_EMBEDDER_MODEL == "custom-embed"
    assert cfg.GRAPH_EMBEDDER_BASE_URL == "http://my-proxy/v1"
    assert cfg.GRAPH_EMBEDDER_API_KEY == "sk-main"
    # DIMS exception: stays a concrete default (graph schema needs a number;
    # the main side is None/auto-detect)
    assert cfg.GRAPH_EMBEDDER_DIMS == 1024


def test_graph_own_env_wins(clean_env):
    """GRAPH_EMBEDDER_API_KEY now actually works (pre-rename bug: config.py
    never read its own GRAPH_EMBEDDING_API_KEY env)."""
    clean_env.setenv("EMBEDDER_API_KEY", "sk-main")
    clean_env.setenv("GRAPH_EMBEDDER_API_KEY", "sk-graph")
    clean_env.setenv("GRAPH_EMBEDDER_MODEL", "graph-embed")
    clean_env.setenv("GRAPH_EMBEDDER_DIMS", "768")
    cfg = reload_config()
    assert cfg.GRAPH_EMBEDDER_API_KEY == "sk-graph"
    assert cfg.GRAPH_EMBEDDER_MODEL == "graph-embed"
    assert cfg.GRAPH_EMBEDDER_DIMS == 768
