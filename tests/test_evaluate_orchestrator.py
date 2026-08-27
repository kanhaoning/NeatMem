"""Unit checks for neatmem.evaluation.orchestrator: env layering, forced
override, ambient filtering, judge mapping, resume validation, env-file/API-key
prechecks, redaction, serve-flag translation. No servers started.

2026-08-23: --config and the bundled strategy .env files removed; one run =
one strategy described by env (--env-file < process env) + flags + forced.

Run: cd <repo root> && python -m pytest tests/test_evaluate_orchestrator.py -q
"""

import json
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
def envfile(tmp_path):
    p = tmp_path / "x.env"
    p.write_text("OPENAI_BASE_URL=http://envfile\nDEDUP_RESOLVER=edit\n")
    return p


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path, monkeypatch):
    """No ambient ./.env (repo root has one), no leaked API keys."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def test_layering_process_beats_envfile(envfile, monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://process")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-t")
    record, _ = ev.build_env(args_ns(env_file=str(envfile)), {}, FORCED)
    assert record["OPENAI_BASE_URL"] == "http://process"
    assert record["DEDUP_RESOLVER"] == "edit"  # env-file fills when process silent


def test_layering_envfile_used_when_process_silent(envfile, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-t")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    record, _ = ev.build_env(args_ns(env_file=str(envfile)), {}, FORCED)
    assert record["OPENAI_BASE_URL"] == "http://envfile"


def test_layering_key_absent_when_unset_everywhere(envfile, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-t")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    envfile.write_text("DEDUP_RESOLVER=edit\n")
    record, _ = ev.build_env(args_ns(env_file=str(envfile)), {}, FORCED)
    assert "OPENAI_BASE_URL" not in record


def test_flags_beat_envfile_and_process(envfile, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-t")
    monkeypatch.setenv("DEDUP_RESOLVER", "skip")
    record, _ = ev.build_env(args_ns(env_file=str(envfile)),
                             {"DEDUP_RESOLVER": "rewrite"}, FORCED)
    assert record["DEDUP_RESOLVER"] == "rewrite"


def test_forced_beats_process(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-t")
    monkeypatch.setenv("QDRANT_PATH", "/tmp/user-wants-this")
    record, child = ev.build_env(args_ns(), {}, FORCED)
    assert child["QDRANT_PATH"] == "/forced/db"


def test_unknown_future_var_passes_and_records(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-t")
    monkeypatch.setenv("DEDUP_SOME_FUTURE_KNOB", "1")
    record, child = ev.build_env(args_ns(), {}, FORCED)
    assert child["DEDUP_SOME_FUTURE_KNOB"] == "1"
    assert record["DEDUP_SOME_FUTURE_KNOB"] == "1"


def test_explicit_env_file_missing_dies():
    with pytest.raises(SystemExit):
        ev.build_env(args_ns(env_file="/nonexistent.env"), {}, FORCED)


def test_default_dotenv_loaded(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("OPENAI_API_KEY=sk-from-dotenv\n")
    record, _ = ev.build_env(args_ns(), {}, FORCED)
    assert record["OPENAI_API_KEY"] == "sk-from-dotenv"


def test_missing_api_key_dies():
    with pytest.raises(SystemExit, match="OPENAI_API_KEY"):
        ev.build_env(args_ns(), {}, FORCED)


def test_record_filter():
    assert ev.is_recorded("DEDUP_SOME_FUTURE_KNOB")
    assert ev.is_recorded("HISTORY_DB_PATH")
    assert not ev.is_recorded("PATH")
    assert not ev.is_recorded("ANTHROPIC_AUTH_TOKEN")
    assert not ev.is_recorded("AutodlAutoPanelToken")
    assert not ev.is_recorded("")


def test_secret_passes_child_but_not_record(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-t")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "tok-should-not-leak")
    record, child = ev.build_env(args_ns(), {}, FORCED)
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
    assert ev.parse_serve_args(["--rerank", "off"]) == {"RERANK_MODE": "off"}
    assert ev.parse_serve_args(["--rerank", "cross_encoder"]) == {"RERANK_MODE": "cross_encoder"}
    assert ev.parse_serve_args(["--dedup"]) == {"DEDUP_ENABLED": "true"}
    assert ev.parse_serve_args(["--no-dedup"]) == {"DEDUP_ENABLED": "false"}
    assert ev.parse_serve_args(["--llm-base-url", "http://x/v1"]) == {"OPENAI_BASE_URL": "http://x/v1"}


def test_serve_flags_exhaustive_mapping():
    """Every serve flag maps to the documented env var (single source:
    cli.serve_flags_to_env, shared with `neatmem serve`)."""
    env = ev.parse_serve_args([
        "--llm-model", "m1", "--llm-api-key", "k", "--llm-provider", "minimax",
        "--embedder-model", "e1", "--embedder-base-url", "http://e",
        "--embedder-api-key", "ek", "--embedder-provider", "siliconflow",
        "--extraction-prompt", "p1", "--dedup-prompt", "p2", "--rewrite-prompt", "p3",
        "--edit-prompt", "p4", "--rerank-prompt", "p5",
        "--enable-bm25", "--enable-entity", "--enable-graph", "--dedup-thinking",
        "--extract-last-k-messages", "7", "--embedder-dims", "1024",
    ])
    assert env["LLM_MODEL"] == "m1"
    assert env["LLM_API_KEY"] == "k"
    assert env["LLM_PROVIDER"] == "minimax"
    assert env["EMBEDDER_MODEL"] == "e1"
    assert env["EMBEDDER_BASE_URL"] == "http://e"
    assert env["EMBEDDER_API_KEY"] == "ek"
    assert env["EMBEDDER_PROVIDER"] == "siliconflow"
    # Prompt flags carrying file paths are absolutized at parse time
    # (evaluate children run under the output dir).
    import pathlib
    cwd = str(pathlib.Path.cwd())
    assert env["EXTRACTION_PROMPT"] == f"{cwd}/p1"
    assert env["DEDUP_PROMPT"] == f"{cwd}/p2"
    assert env["REWRITE_PROMPT"] == f"{cwd}/p3"
    assert env["EDIT_PROMPT"] == f"{cwd}/p4"
    assert env["LLM_RERANK_PROMPT"] == f"{cwd}/p5"
    assert env["ENABLE_BM25"] == "true"
    assert env["ENABLE_ENTITY"] == "true"
    assert env["ENABLE_GRAPH"] == "true"
    assert env["DEDUP_THINKING"] == "true"
    assert env["EXTRACT_LAST_K_MESSAGES"] == "7"
    assert env["EMBEDDER_DIMS"] == "1024"


def test_serve_flags_unknown_errors():
    with pytest.raises(SystemExit):
        ev.parse_serve_args(["--definitely-not-a-flag"])


def test_search_rerank_follows_env():
    # per-request --rerank on would silently defeat a RERANK_MODE=off arm
    assert ev.search_rerank_arg({}) == "on"  # package default RERANK_MODE=llm
    assert ev.search_rerank_arg({"RERANK_MODE": "llm"}) == "on"
    assert ev.search_rerank_arg({"RERANK_MODE": "cross_encoder"}) == "on"
    assert ev.search_rerank_arg({"RERANK_MODE": "off"}) == "off"


# --- eval CLI flags (2026-08-23 mem0-aligned additions) ---

def parse_eval(argv):
    """Parse + normalize eval args; point file checks at this file."""
    args, serve_args = ev.build_eval_parser().parse_known_args(argv)
    args.serve_args = serve_args
    args.dataset = __file__
    args.qdrant_bin = __file__
    return ev.args_normalize(args)


def test_stages_default_all():
    assert parse_eval([]).stages == ["ingest", "search", "judge"]


def test_predict_only_alias():
    assert parse_eval(["--predict-only"]).stages == ["ingest", "search"]


def test_evaluate_only_alias():
    assert parse_eval(["--evaluate-only"]).stages == ["judge"]


def test_both_aliases_cover_all_stages():
    assert parse_eval(["--predict-only", "--evaluate-only"]).stages == \
        ["ingest", "search", "judge"]


def test_alias_conflict_errors():
    with pytest.raises(SystemExit, match="conflicts"):
        parse_eval(["--stages", "search,judge", "--predict-only"])


def test_alias_redundant_stages_allowed():
    args = parse_eval(["--stages", "ingest,search", "--predict-only"])
    assert args.stages == ["ingest", "search"]


def test_output_dir_default_derives_from_project_name():
    assert parse_eval([]).output_dir == "runs/default"
    assert parse_eval(["--project-name", "arm1"]).output_dir == "runs/arm1"


def test_output_dir_explicit_overrides_derivation():
    args = parse_eval(["--project-name", "arm1", "--output-dir", "/tmp/x"])
    assert args.output_dir == "/tmp/x"


def test_workers_builtin_default():
    args = parse_eval([])
    assert (args.ingest_workers, args.search_workers, args.judge_workers) == (4, 4, 4)


def test_max_workers_umbrella():
    args = parse_eval(["--max-workers", "16"])
    assert (args.ingest_workers, args.search_workers, args.judge_workers) == (16, 16, 16)


def test_per_stage_beats_umbrella():
    args = parse_eval(["--max-workers", "16", "--judge-workers", "2"])
    assert (args.ingest_workers, args.search_workers, args.judge_workers) == (16, 16, 2)


def test_eval_flags_to_env():
    args = parse_eval(["--top-k", "50", "--batch-size", "5",
                       "--answerer-model", "ans-m", "--judge-model", "j-m"])
    assert ev.eval_flags_to_env(args) == {
        "TOP_K": "50", "BATCH_SIZE": "5",
        "ANSWER_MODEL": "ans-m", "JUDGE_MODEL": "j-m",
    }
    assert ev.eval_flags_to_env(parse_eval([])) == {}


def test_model_flags_reach_judge_env():
    """--judge-model lands in JUDGE_MODEL; judge_env maps it to LLM_MODEL for
    the judge child (llm_judge.py reads OPENAI_*/LLM_MODEL)."""
    child = {"OPENAI_API_KEY": "sk", "LLM_MODEL": "main-m",
             **ev.eval_flags_to_env(parse_eval(["--judge-model", "j-m"]))}
    jenv = ev.judge_env(child)
    assert jenv["LLM_MODEL"] == "j-m"


def test_judge_resume_compares_qa_totals(tmp_path):
    """Judged files are keyed per judged task (category 5 excluded), results
    per conversation; the resume check must compare judged-QA totals
    (2026-08-23 bug: key counts never matched, judge re-ran every resume)."""
    res = tmp_path / "res.json"
    res.write_text(json.dumps({"0": [{"category": 1}, {"category": 5}],
                               "1": [{"category": 2}]}))
    judged = tmp_path / "judged.json"
    judged.write_text(json.dumps({"0": [{"category": 1}], "1": [{"category": 2}]}))
    assert ev.results_conversation_count(res) == 2
    assert ev.judged_qa_count(res) == 2
    assert ev.judged_qa_count(judged) == 2
