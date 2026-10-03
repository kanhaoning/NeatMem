"""Unit tests for neatmem.query_rewrite."""

import json
from types import SimpleNamespace

import httpx
import pytest
from openai import APITimeoutError, RateLimitError

from neatmem.query_rewrite import (
    CONTEXT_MAX_CHARS,
    build_context,
    rewrite_query,
)


def _resp(payload: dict):
    body = json.dumps(payload)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=body))]
    )


def _chat_returning(payload):
    def _fn(client, model, messages, enable, **params):
        return _resp(payload)
    return _fn


def _chat_raising(exc):
    def _fn(client, model, messages, enable, **params):
        raise exc
    return _fn


KW = dict(client=None, model="m", timeout_s=3, max_expansions=3)


# --- build_context ---

def test_build_context_basic_format():
    turns = [
        {"role": "user", "content": "明天天气"},
        {"role": "assistant", "content": "晴"},
        {"role": "user", "content": "后天呢"},
    ]
    assert build_context(turns) == "user: 明天天气\nassistant: 晴\nuser: 后天呢"


def test_build_context_strips_assistant_think():
    turns = [
        {"role": "assistant", "content": "<think>几千字内心戏</think>真实回复"},
        {"role": "user", "content": "继续"},
    ]
    assert build_context(turns) == "assistant: 真实回复\nuser: 继续"


def test_build_context_drops_empty_after_strip_and_non_dialogue_roles():
    turns = [
        {"role": "assistant", "content": "<think>只有思考被截断"},
        {"role": "system", "content": "系统提示"},
        {"role": "tool", "content": "工具输出"},
        {"role": "user", "content": "  "},
        {"role": "user", "content": "问题"},
    ]
    assert build_context(turns) == "user: 问题"


def test_build_context_keeps_user_literal_think_text():
    turns = [{"role": "user", "content": "怎么理解 <think> 标签？"}]
    out = build_context(turns)
    assert "<think>" in out


def test_build_context_newest_wins_over_budget():
    turns = [
        {"role": "user", "content": "旧" * 100},
        {"role": "user", "content": "新" * 100},
    ]
    out = build_context(turns, max_chars=150)
    assert "新" in out
    assert len(out) <= 150


def test_build_context_oversized_single_turn_tail_truncated():
    turns = [{"role": "assistant", "content": "x" * (CONTEXT_MAX_CHARS + 500)}]
    out = build_context(turns)
    assert len(out) == CONTEXT_MAX_CHARS
    assert out.endswith("x" * 100)


def test_build_context_empty():
    assert build_context([]) == ""


# --- rewrite_query ---

def test_rewrite_success_with_rephrase():
    result = rewrite_query(
        "那后天呢", "user: 明天天气\nassistant: 晴",
        chat_fn=_chat_returning({
            "rephrased_query": "后天的天气怎么样",
            "expansions": ["后天天气预报", "后天天气"],
        }),
        **KW,
    )
    assert result.final_query == "后天的天气怎么样"
    assert result.rephrased is True
    assert result.expansions == ("后天天气预报", "后天天气")
    assert result.fallback_reason is None
    assert result.latency_ms >= 0


def test_rewrite_self_judged_empty_keeps_original():
    result = rewrite_query(
        "什么是向量检索", "",
        chat_fn=_chat_returning({
            "rephrased_query": "",
            "expansions": ["vector search", "embedding retrieval"],
        }),
        **KW,
    )
    assert result.final_query == "什么是向量检索"
    assert result.rephrased is False
    assert len(result.expansions) == 2
    assert result.fallback_reason is None


def test_rewrite_verbatim_repeat_treated_as_empty():
    result = rewrite_query(
        "什么是 向量检索 ", "",
        chat_fn=_chat_returning({
            "rephrased_query": "什么是向量检索",
            "expansions": ["相似度搜索"],
        }),
        **KW,
    )
    assert result.rephrased is False
    assert result.final_query == "什么是 向量检索 "


def test_expansions_dedup_against_query_and_each_other():
    result = rewrite_query(
        "天气", "",
        chat_fn=_chat_returning({
            "rephrased_query": "",
            "expansions": ["天气", " 天气 ", "气温", "气温", "降水", "风力"],
        }),
        **KW,
    )
    assert result.expansions == ("气温", "降水", "风力")


def test_expansions_cap_and_zero_mode():
    payload = {"rephrased_query": "", "expansions": ["a", "b", "c", "d"]}
    capped = rewrite_query("q", "", chat_fn=_chat_returning(payload), **{**KW, "max_expansions": 2})
    assert capped.expansions == ("a", "b")
    rewrite_only = rewrite_query("q", "", chat_fn=_chat_returning(payload), **{**KW, "max_expansions": 0})
    assert rewrite_only.expansions == ()
    assert rewrite_only.final_query == "q"


def test_expansions_zero_is_valid_outcome():
    result = rewrite_query(
        "q", "", chat_fn=_chat_returning({"rephrased_query": "改写", "expansions": []}), **KW
    )
    assert result.final_query == "改写"
    assert result.expansions == ()
    assert result.fallback_reason is None


def test_parse_failure_falls_back():
    def bad_json(client, model, messages, enable, **params):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="not json at all"))]
        )
    result = rewrite_query("q", "", chat_fn=bad_json, **KW)
    assert result.final_query == "q"
    assert result.expansions == ()
    assert result.fallback_reason == "parse"


def test_code_fenced_json_accepted():
    def fenced(client, model, messages, enable, **params):
        body = '```json\n{"rephrased_query": "改写后", "expansions": ["扩展"]}\n```'
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=body))]
        )
    result = rewrite_query("q", "", chat_fn=fenced, **KW)
    assert result.final_query == "改写后"
    assert result.expansions == ("扩展",)


def test_timeout_falls_back():
    result = rewrite_query(
        "q", "",
        chat_fn=_chat_raising(APITimeoutError(request=None)),
        **KW,
    )
    assert result.fallback_reason == "timeout"
    assert result.final_query == "q"


def test_rate_limit_falls_back():
    response = httpx.Response(429, request=httpx.Request("POST", "http://test"))
    result = rewrite_query(
        "q", "",
        chat_fn=_chat_raising(RateLimitError("429", response=response, body=None)),
        **KW,
    )
    assert result.fallback_reason == "rate_limited"


def test_generic_error_falls_back_with_type_name():
    result = rewrite_query(
        "q", "", chat_fn=_chat_raising(ConnectionError("boom")), **KW
    )
    assert result.fallback_reason == "error:ConnectionError"
    assert result.final_query == "q"


# --- retries (QUERY_REWRITE_RETRIES; default 0 = production single attempt) ---

@pytest.fixture
def no_sleep(monkeypatch):
    import neatmem.query_rewrite as qr
    monkeypatch.setattr(qr.time, "sleep", lambda s: None)


def _chat_flaky(exc_factory, fail_times, payload):
    """Fails ``fail_times`` times with exc_factory(), then returns payload."""
    calls = {"n": 0}

    def _fn(client, model, messages, enable, **params):
        calls["n"] += 1
        if calls["n"] <= fail_times:
            raise exc_factory()
        return _resp(payload)

    return _fn, calls


OK_PAYLOAD = {"rephrased_query": "", "expansions": ["a", "b"]}


def test_transient_timeout_retried_then_success(no_sleep):
    fn, calls = _chat_flaky(lambda: APITimeoutError(request=None), 2, OK_PAYLOAD)
    result = rewrite_query("q", "", chat_fn=fn, retries=5, **KW)
    assert result.fallback_reason is None
    assert result.retries_used == 2
    assert result.final_query == "q"  # self-judged empty -> original
    assert list(result.expansions) == ["a", "b"]
    assert calls["n"] == 3


def test_transient_retries_exhausted_falls_back(no_sleep):
    fn, calls = _chat_flaky(lambda: APITimeoutError(request=None), 99, OK_PAYLOAD)
    result = rewrite_query("q", "", chat_fn=fn, retries=2, **KW)
    assert result.fallback_reason == "timeout"
    assert result.final_query == "q"
    assert result.retries_used == 2
    assert calls["n"] == 3  # 1 initial + 2 retries


def test_529_overloaded_string_error_is_transient(no_sleep):
    fn, calls = _chat_flaky(
        lambda: Exception("Error code: 529 - {'error': {'type': 'overloaded_error'}}"),
        1, OK_PAYLOAD,
    )
    result = rewrite_query("q", "", chat_fn=fn, retries=5, **KW)
    assert result.fallback_reason is None
    assert result.retries_used == 1
    assert calls["n"] == 2


def test_parse_failure_not_retried(no_sleep):
    def bad_json(client, model, messages, enable, **params):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="not json at all"))]
        )
    result = rewrite_query("q", "", chat_fn=bad_json, retries=5, **KW)
    assert result.fallback_reason == "parse"
    assert result.retries_used == 0


def test_non_transient_error_not_retried(no_sleep):
    fn, calls = _chat_flaky(lambda: ConnectionError("boom"), 99, OK_PAYLOAD)
    result = rewrite_query("q", "", chat_fn=fn, retries=5, **KW)
    assert result.fallback_reason == "error:ConnectionError"
    assert result.retries_used == 0
    assert calls["n"] == 1


def test_default_retries_zero_is_single_attempt(no_sleep):
    fn, calls = _chat_flaky(lambda: APITimeoutError(request=None), 99, OK_PAYLOAD)
    result = rewrite_query("q", "", chat_fn=fn, **KW)  # retries defaults to 0
    assert result.fallback_reason == "timeout"
    assert result.retries_used == 0
    assert calls["n"] == 1
