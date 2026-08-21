"""Unit checks for neatmem.evaluation.orchestrator: env layering, forced
override, ambient filtering, judge mapping, resume validation, config
resolution, redaction, serve-flag translation. No servers started.

Run: cd <repo root> && python -m pytest tests/test_evaluate_orchestrator.py -q
"""

import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from neatmem.evaluation import orchestrator as ev

FORCED = {"QDRANT_PATH": "/forced/db", "NEATMEM_PORT": "9999"}


def args_ns(**kw):
    base = dict(env_file=None, reuse_db=None, force=False, dataset="/d.json",
                output_dir=tempfile.mkdtemp(), stages=["judge"], runs=1,
                limit=None, serve_args=[], top_k=None, batch_size=None)
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.fixture
def strat(tmp_path):
    p = tmp_path / "x.env"
    p.write_text("OPENAI_BASE_URL=http://strategy\nDEDUP_RESOLVER=edit\n")
    return p


def test_layering_config_beats_process(strat, monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://process")
    record, _ = ev.build_env(args_ns(), strat, {}, FORCED)
    assert record["OPENAI_BASE_URL"] == "http://strategy"
    assert record["DEDUP_RESOLVER"] == "edit"


def test_layering_process_fills_when_config_silent(strat, monkeypatch):
    strat.write_text("DEDUP_RESOLVER=edit\n")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://process")
    record, _ = ev.build_env(args_ns(), strat, {}, FORCED)
    assert record["OPENAI_BASE_URL"] == "http://process"


def test_layering_key_absent_when_unset_everywhere(strat, monkeypatch):
    strat.write_text("DEDUP_RESOLVER=edit\n")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    record, _ = ev.build_env(args_ns(env_file="/nonexistent.env"), strat, {}, FORCED)
    assert "OPENAI_BASE_URL" not in record


def test_flags_beat_config(strat):
    record, _ = ev.build_env(args_ns(), strat, {"DEDUP_RESOLVER": "rewrite"}, FORCED)
    assert record["DEDUP_RESOLVER"] == "rewrite"


def test_forced_beats_process(strat, monkeypatch):
    monkeypatch.setenv("QDRANT_PATH", "/tmp/user-wants-this")
    record, child = ev.build_env(args_ns(), strat, {}, FORCED)
    assert child["QDRANT_PATH"] == "/forced/db"


def test_unknown_future_var_passes_and_records(strat, monkeypatch):
    monkeypatch.setenv("DEDUP_SOME_FUTURE_KNOB", "1")
    record, child = ev.build_env(args_ns(), strat, {}, FORCED)
    assert child["DEDUP_SOME_FUTURE_KNOB"] == "1"
    assert record["DEDUP_SOME_FUTURE_KNOB"] == "1"


def test_record_filter():
    assert ev.is_recorded("DEDUP_SOME_FUTURE_KNOB")
    assert ev.is_recorded("HISTORY_DB_PATH")
    assert not ev.is_recorded("PATH")
    assert not ev.is_recorded("ANTHROPIC_AUTH_TOKEN")
    assert not ev.is_recorded("AutodlAutoPanelToken")
    assert not ev.is_recorded("")


def test_secret_passes_child_but_not_record(strat, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "tok-should-not-leak")
    record, child = ev.build_env(args_ns(), strat, {}, FORCED)
    assert child["ANTHROPIC_AUTH_TOKEN"] == "tok-should-not-leak"
    assert "ANTHROPIC_AUTH_TOKEN" not in record


def test_judge_namespace_mapping():
    child = {"OPENAI_BASE_URL": "http://server", "LLM_MODEL": "server-model",
             "JUDGE_BASE_URL": "http://judge", "JUDGE_MODEL": "judge-model"}
    jenv = ev.judge_env(child)
    assert jenv["OPENAI_BASE_URL"] == "http://judge"
    assert jenv["LLM_MODEL"] == "judge-model"
    jenv2 = ev.judge_env({"OPENAI_BASE_URL": "http://server", "LLM_MODEL": "m"})
    assert jenv2["OPENAI_BASE_URL"] == "http://server"


def test_resume_env_mismatch_errors(tmp_path):
    sd = tmp_path / "skip"
    sd.mkdir()
    args = args_ns(output_dir=str(tmp_path))
    ev.load_or_validate_manifest(sd, "skip", {"A": "1", "B": "2"},
                                 ["--dedup-resolver", "skip"], __file__, args)
    with pytest.raises(SystemExit):
        ev.load_or_validate_manifest(sd, "skip", {"A": "1", "B": "3"},
                                     ["--dedup-resolver", "skip"], __file__, args)


def test_resume_serve_args_mismatch_errors(tmp_path):
    sd = tmp_path / "skip"
    sd.mkdir()
    args = args_ns(output_dir=str(tmp_path))
    ev.load_or_validate_manifest(sd, "skip", {"A": "1"},
                                 ["--dedup-resolver", "skip"], __file__, args)
    with pytest.raises(SystemExit):
        ev.load_or_validate_manifest(sd, "skip", {"A": "1"},
                                     ["--dedup-resolver", "edit"], __file__, args)


def test_resume_same_config_passes(tmp_path):
    sd = tmp_path / "skip"
    sd.mkdir()
    args = args_ns(output_dir=str(tmp_path))
    ev.load_or_validate_manifest(sd, "skip", {"A": "1"}, [], __file__, args)
    m2, _ = ev.load_or_validate_manifest(sd, "skip", {"A": "1"}, [], __file__, args)
    assert m2.get("strategy") == "skip"


def test_resume_volatile_port_tolerated(tmp_path):
    sd = tmp_path / "skip"
    sd.mkdir()
    args = args_ns(output_dir=str(tmp_path))
    ev.load_or_validate_manifest(sd, "skip", {"A": "1", "NEATMEM_PORT": "9001"},
                                 [], __file__, args)
    m2, _ = ev.load_or_validate_manifest(sd, "skip", {"A": "1", "NEATMEM_PORT": "9002"},
                                         [], __file__, args)
    assert m2.get("strategy") == "skip"


def test_config_resolution_bundled():
    assert ev.resolve_config("skip")[1].name == "skip.env"
    assert ev.resolve_config("env") == ("env", None)


def test_config_resolution_path(tmp_path):
    p = tmp_path / "custom.env"
    p.write_text("X=1\n")
    assert ev.resolve_config(str(p))[0] == "custom"


def test_config_resolution_unknown_errors():
    with pytest.raises(SystemExit):
        ev.resolve_config("nonexistent")


def test_redaction():
    r = ev.redact({"OPENAI_API_KEY": "sk-x", "My_Secret": "s", "panelToken": "t",
                   "LLM_MODEL": "m"})
    assert r["OPENAI_API_KEY"] == "<redacted>"
    assert r["My_Secret"] == "<redacted>"
    assert r["panelToken"] == "<redacted>"
    assert r["LLM_MODEL"] == "m"


# --- serve-flag passthrough translation (the write-side fix) ---

def test_serve_flags_translated_to_env():
    env = ev.parse_serve_args(["--dedup-resolver", "edit", "--dedup-detector", "pointwise"])
    assert env == {"DEDUP_RESOLVER": "edit", "DEDUP_DETECTOR": "pointwise"}


def test_serve_flags_irregular_mappings():
    assert ev.parse_serve_args(["--rerank"]) == {"LLM_RERANK": "true"}
    assert ev.parse_serve_args(["--no-rerank"]) == {"LLM_RERANK": "false"}
    assert ev.parse_serve_args(["--dedup"]) == {"DEDUP_ENABLED": "true"}
    assert ev.parse_serve_args(["--no-dedup"]) == {"DEDUP_ENABLED": "false"}
    assert ev.parse_serve_args(["--llm-base-url", "http://x/v1"]) == {"OPENAI_BASE_URL": "http://x/v1"}


def test_serve_flags_exhaustive_mapping():
    """Every serve flag maps to the documented env var (single source:
    cli.serve_flags_to_env, shared with `neatmem serve`)."""
    env = ev.parse_serve_args([
        "--llm-model", "m1", "--llm-api-key", "k", "--llm-provider", "minimax",
        "--embedding-model", "e1", "--embedding-base-url", "http://e",
        "--embedding-api-key", "ek", "--embedding-provider", "siliconflow",
        "--extraction-prompt", "p1", "--dedup-prompt", "en", "--rewrite-prompt", "p3",
        "--edit-prompt", "p4", "--rerank-prompt", "p5",
        "--enable-bm25", "--enable-entity", "--enable-graph", "--dedup-thinking",
        "--extract-last-k-messages", "7", "--embedding-dims", "1024",
    ])
    assert env["LLM_MODEL"] == "m1"
    assert env["LLM_API_KEY"] == "k"
    assert env["LLM_PROVIDER"] == "minimax"
    assert env["EMBEDDING_MODEL"] == "e1"
    assert env["EMBEDDING_BASE_URL"] == "http://e"
    assert env["EMBEDDING_API_KEY"] == "ek"
    assert env["EMBEDDING_PROVIDER"] == "siliconflow"
    assert env["EXTRACTION_PROMPT"] == "p1"
    assert env["DEDUP_PROMPT"] == "en"
    assert env["REWRITE_PROMPT"] == "p3"
    assert env["EDIT_PROMPT"] == "p4"
    assert env["RERANK_PROMPT"] == "p5"
    assert env["ENABLE_BM25"] == "true"
    assert env["ENABLE_ENTITY"] == "true"
    assert env["ENABLE_GRAPH"] == "true"
    assert env["DEDUP_THINKING"] == "true"
    assert env["EXTRACT_LAST_K_MESSAGES"] == "7"
    assert env["EMBEDDING_DIMS"] == "1024"


def test_serve_flags_unknown_errors():
    with pytest.raises(SystemExit):
        ev.parse_serve_args(["--definitely-not-a-flag"])


def test_search_rerank_follows_env():
    # per-request --rerank on would silently defeat a --no-rerank arm
    assert ev.search_rerank_arg({}) == "on"  # package default true
    assert ev.search_rerank_arg({"LLM_RERANK": "true"}) == "on"
    assert ev.search_rerank_arg({"LLM_RERANK": "false"}) == "off"
