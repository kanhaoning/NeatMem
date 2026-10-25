"""L0 tests for the reasoning-effort feature (2026-10-25, plan:
docs/internal-notes/20261025-reasoning-effort-knob-plan.md).

Covers: env knob parsing + conflict rule, complete_chat pass-through
(byte-identical when unset), probe classification (200/400/transient),
option-B 400-abort gating on rerank + rewrite paths.
"""

import importlib
import os
from unittest import mock

import pytest
from openai import BadRequestError

from neatmem.utils import llm_client
from neatmem.utils.llm_client import complete_chat, effort_abort_enabled, get_effort_env
from neatmem.utils.startup_probe import ProbeStage, run_launch_probe


def _env(monkeypatch, **vars):
    for k in llm_client.EFFORT_ENV_VARS:
        monkeypatch.delenv(k, raising=False)
    for k, v in vars.items():
        monkeypatch.setenv(k, v)


# ---------------------------------------------------------------------------
# knob parsing + abort gate
# ---------------------------------------------------------------------------

def test_effort_env_unset_is_none(monkeypatch):
    _env(monkeypatch)
    assert get_effort_env("RERANK_REASONING_EFFORT") is None
    assert effort_abort_enabled() is False


def test_effort_env_blank_is_none(monkeypatch):
    _env(monkeypatch, RERANK_REASONING_EFFORT="   ")
    assert get_effort_env("RERANK_REASONING_EFFORT") is None
    assert effort_abort_enabled() is False


def test_effort_env_value_verbatim(monkeypatch):
    _env(monkeypatch, DEDUP_REASONING_EFFORT="xhigh")
    assert get_effort_env("DEDUP_REASONING_EFFORT") == "xhigh"
    assert effort_abort_enabled() is True


def test_config_conflict_explicit_false_plus_effort(monkeypatch):
    import neatmem.config as cfg  # import before env mutation
    _env(monkeypatch, DEDUP_REASONING_EFFORT="low", DEDUP_THINKING="false")
    with pytest.raises(ValueError, match="conflicts with explicit"):
        importlib.reload(cfg)
    monkeypatch.delenv("DEDUP_REASONING_EFFORT", raising=False)
    importlib.reload(cfg)  # restore clean module state


def test_config_no_conflict_when_thinking_unset(monkeypatch):
    import neatmem.config as cfg  # import before env mutation
    _env(monkeypatch, DEDUP_REASONING_EFFORT="low")
    monkeypatch.delenv("DEDUP_THINKING", raising=False)
    importlib.reload(cfg)  # must not raise
    assert cfg.DEDUP_REASONING_EFFORT == "low"
    monkeypatch.delenv("DEDUP_REASONING_EFFORT", raising=False)
    importlib.reload(cfg)


# ---------------------------------------------------------------------------
# complete_chat pass-through
# ---------------------------------------------------------------------------

class _CaptureCompletions:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return mock.MagicMock()


class _CaptureClient:
    def __init__(self):
        self.chat = mock.MagicMock()
        self.chat.completions = _CaptureCompletions()


def test_complete_chat_effort_pass_through_verbatim():
    client = _CaptureClient()
    complete_chat(client, "MiniMax-M3.1-Flash", [{"role": "user", "content": "x"}],
                  True, provider="minimax", effort="low", temperature=0.0)
    assert client.chat.completions.kwargs["extra_body"]["reasoning_effort"] == "low"


def test_complete_chat_no_effort_byte_identical():
    client = _CaptureClient()
    complete_chat(client, "MiniMax-M3", [{"role": "user", "content": "x"}],
                  False, provider=None, temperature=0.0)
    assert client.chat.completions.kwargs["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False},
        "thinking": {"type": "disabled"},
    }


def test_complete_chat_effort_arbitrary_string_accepted():
    # No whitelist: even a nonsense value passes through (validation is the
    # launch probe's job, not the client's).
    client = _CaptureClient()
    complete_chat(client, "m", [{"role": "user", "content": "x"}],
                  True, provider="minimax", effort="bogus-future-tier")
    assert client.chat.completions.kwargs["extra_body"]["reasoning_effort"] == \
        "bogus-future-tier"


# ---------------------------------------------------------------------------
# launch probe classification
# ---------------------------------------------------------------------------

def _resp_ok():
    return mock.MagicMock()


def _make_client(behavior):
    client = mock.MagicMock()
    if behavior == "ok":
        client.chat.completions.create.return_value = _resp_ok()
    else:
        client.chat.completions.create.side_effect = behavior
    return client


def _bad_request():
    return BadRequestError("invalid params: reasoning_effort 'bogus' "
                           "(allowed: low, medium, high, xhigh, max)",
                           response=mock.MagicMock(status_code=400), body=None)


def test_probe_skipped_when_no_stages():
    assert run_launch_probe([]) == 0


def test_probe_success_verifies():
    stage = ProbeStage("rerank", _make_client("ok"), "m",
                       {"enable": True, "effort": "low"})
    assert run_launch_probe([stage]) == 1


def test_probe_400_refuses_startup_with_provider_message():
    stage = ProbeStage("rerank", _make_client(_bad_request()), "m",
                       {"enable": True, "effort": "bogus"})
    with pytest.raises(SystemExit, match="allowed: low"):
        run_launch_probe([stage])


def test_probe_transient_warns_and_continues():
    from openai import RateLimitError
    err = RateLimitError("429", response=mock.MagicMock(status_code=429), body=None)
    stage = ProbeStage("rerank", _make_client(err), "m",
                       {"enable": True, "effort": "low"})
    assert run_launch_probe([stage]) == 0  # no SystemExit


def test_probe_passes_stage_params_verbatim():
    client = _make_client("ok")
    params = {"enable": True, "effort": "high", "temperature": 0.0,
              "max_tokens": 32768}
    run_launch_probe([ProbeStage("edit", client, "m", params)])
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["extra_body"]["reasoning_effort"] == "high"
    assert kwargs["max_tokens"] == 32768


# ---------------------------------------------------------------------------
# option-B runtime 400-abort gating
# ---------------------------------------------------------------------------

def _listwise_pieces(monkeypatch):
    import neatmem.rerank as rk
    monkeypatch.setattr(rk, "_get_prompt", lambda q, c: "prompt")
    monkeypatch.setattr(rk, "_build_candidates_text", lambda h: "cands")
    return rk


def test_rerank_400_aborts_when_effort_set(monkeypatch):
    rk = _listwise_pieces(monkeypatch)
    monkeypatch.setattr(rk, "_EFFORT_ABORT", True)
    client = _make_client(_bad_request())
    docs = [{"memory": f"m{i}", "score": 0.9 - i * 0.01} for i in range(3)]
    with pytest.raises(BadRequestError):
        rk._llm_rerank_listwise(client, "m", "q", docs)


def test_rerank_400_falls_back_when_no_effort(monkeypatch):
    rk = _listwise_pieces(monkeypatch)
    monkeypatch.setattr(rk, "_EFFORT_ABORT", False)
    client = _make_client(_bad_request())
    docs = [{"memory": f"m{i}", "score": 0.9 - i * 0.01} for i in range(3)]
    kept, dropped = rk._llm_rerank_listwise(client, "m", "q", docs)
    assert kept == docs and dropped == []  # historical silent fallback intact


def test_rewrite_400_aborts_only_when_effort_set(monkeypatch):
    import neatmem.query_rewrite as qr
    from neatmem.query_rewrite import rewrite_query
    boom = _bad_request()

    def chat_fn(client, model, messages, enable, **params):
        raise boom

    kw = dict(client=None, model="m", timeout_s=3, max_expansions=3,
              chat_fn=chat_fn)
    monkeypatch.setattr(qr, "effort_abort_enabled", lambda: False)
    res = rewrite_query("q", "", **kw)
    assert res.fallback_reason == "bad_request"  # default: fail-open
    monkeypatch.setattr(qr, "effort_abort_enabled", lambda: True)
    with pytest.raises(BadRequestError):
        rewrite_query("q", "", **kw)
