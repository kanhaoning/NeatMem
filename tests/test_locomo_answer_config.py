"""LoCoMo answer thinking/budget controls and truncation integrity."""

from collections import defaultdict
from types import SimpleNamespace

import pytest
from jinja2 import Template

from neatmem.evaluation.locomo import search as search_mod
from neatmem.evaluation.prompts import ANSWER_PROMPT


MODEL = "/root/autodl-tmp/Qwen3.6-35B-A3B-FP8"
HITS = [{
    "id": "mem-1",
    "memory": "Caroline adopted a dog named Milo.",
    "score": 0.91,
    "metadata": {"timestamp": "2023-01-01"},
}]


class _StubNeatMemClient:
    def search_with_graph(self, query, user_id=None, top_k=10, rerank=None):
        return HITS, []


class _CaptureCompletions:
    def __init__(self, finish_reason="stop"):
        self.finish_reason = finish_reason
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        message = SimpleNamespace(content="<think>reasoning</think> final answer")
        choice = SimpleNamespace(message=message, finish_reason=self.finish_reason)
        return SimpleNamespace(choices=[choice])


def _make_search(tmp_path, monkeypatch, finish_reason="stop"):
    monkeypatch.setenv("EVAL_ANSWER_MODEL", MODEL)
    search = search_mod.NeatMemSearch.__new__(search_mod.NeatMemSearch)
    search.top_k = 10
    search.rerank = None
    search.client = _StubNeatMemClient()
    completions = _CaptureCompletions(finish_reason)
    search.openai_client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )
    search.results = defaultdict(list)
    search.output_path = str(tmp_path / "results.json")
    search.answer_template = Template(ANSWER_PROMPT)
    return search, completions


def _answer(search):
    return search.answer_question(
        "Caroline_0", "Melanie_0", "what pet?", "a dog", 2,
    )


def test_default_answer_request_uses_thinking_and_32k(tmp_path, monkeypatch):
    monkeypatch.delenv("EVAL_ANSWER_THINKING", raising=False)
    monkeypatch.delenv("EVAL_ANSWER_MAX_TOKENS", raising=False)
    search, completions = _make_search(tmp_path, monkeypatch)
    assert _answer(search)[0] == "final answer"
    assert completions.kwargs["max_tokens"] == 32768
    assert completions.kwargs["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": True}
    }


def test_unprefixed_experimental_names_are_not_read(tmp_path, monkeypatch):
    monkeypatch.delenv("EVAL_ANSWER_THINKING", raising=False)
    monkeypatch.delenv("EVAL_ANSWER_MAX_TOKENS", raising=False)
    monkeypatch.setenv("ANSWER_THINKING", "false")
    monkeypatch.setenv("ANSWER_MAX_TOKENS", "2000")
    search, completions = _make_search(tmp_path, monkeypatch)
    assert _answer(search)[0] == "final answer"
    assert completions.kwargs["max_tokens"] == 32768
    assert completions.kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True


def test_no_thinking_answer_request_with_explicit_32k(tmp_path, monkeypatch):
    monkeypatch.setenv("EVAL_ANSWER_THINKING", "false")
    monkeypatch.setenv("EVAL_ANSWER_MAX_TOKENS", "32768")
    search, completions = _make_search(tmp_path, monkeypatch)
    assert _answer(search)[0] == "final answer"
    assert completions.kwargs["max_tokens"] == 32768
    assert completions.kwargs["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }


def test_explicit_16k_budget_is_honored(tmp_path, monkeypatch):
    monkeypatch.setenv("EVAL_ANSWER_THINKING", "true")
    monkeypatch.setenv("EVAL_ANSWER_MAX_TOKENS", "16384")
    search, completions = _make_search(tmp_path, monkeypatch)
    assert _answer(search)[0] == "final answer"
    assert completions.kwargs["max_tokens"] == 16384
    assert completions.kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True


def test_eval_answer_model_preferred_over_legacy(tmp_path, monkeypatch):
    monkeypatch.setenv("EVAL_ANSWER_MODEL", MODEL)
    monkeypatch.setenv("ANSWER_MODEL", "legacy-model")
    search, completions = _make_search(tmp_path, monkeypatch)
    assert _answer(search)[0] == "final answer"
    assert completions.kwargs["model"] == MODEL


def test_legacy_answer_model_fallback(tmp_path, monkeypatch):
    monkeypatch.delenv("EVAL_ANSWER_MODEL", raising=False)
    monkeypatch.setenv("ANSWER_MODEL", MODEL)
    search, completions = _make_search(tmp_path, monkeypatch)
    assert _answer(search)[0] == "final answer"
    assert completions.kwargs["model"] == MODEL


def test_shared_llm_model_fallback(tmp_path, monkeypatch):
    monkeypatch.delenv("EVAL_ANSWER_MODEL", raising=False)
    monkeypatch.delenv("ANSWER_MODEL", raising=False)
    monkeypatch.setenv("LLM_MODEL", MODEL)
    search, completions = _make_search(tmp_path, monkeypatch)
    assert _answer(search)[0] == "final answer"
    assert completions.kwargs["model"] == MODEL


def test_length_finish_reason_fails_before_empty_answer(tmp_path, monkeypatch):
    monkeypatch.setenv("EVAL_ANSWER_THINKING", "true")
    monkeypatch.setenv("EVAL_ANSWER_MAX_TOKENS", "2000")
    search, _ = _make_search(tmp_path, monkeypatch, finish_reason="length")
    with pytest.raises(RuntimeError, match="truncated at EVAL_ANSWER_MAX_TOKENS=2000"):
        _answer(search)


def _client_config(tmp_path, monkeypatch):
    captured = {}

    def fake_openai(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(search_mod, "OpenAI", fake_openai)
    monkeypatch.setattr(search_mod, "NeatMemClient", _StubNeatMemClient)
    search_mod.NeatMemSearch(output_path=str(tmp_path / "results.json"))
    return captured


def test_eval_answer_client_config_preferred(tmp_path, monkeypatch):
    monkeypatch.setenv("EVAL_ANSWER_API_KEY", "eval-key")
    monkeypatch.setenv("EVAL_ANSWER_BASE_URL", "http://eval")
    monkeypatch.setenv("ANSWER_API_KEY", "legacy-key")
    monkeypatch.setenv("ANSWER_BASE_URL", "http://legacy")
    monkeypatch.setenv("OPENAI_API_KEY", "shared-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://shared")
    captured = _client_config(tmp_path, monkeypatch)
    assert captured == {"api_key": "eval-key", "base_url": "http://eval"}


def test_legacy_answer_client_config_fallback(tmp_path, monkeypatch):
    monkeypatch.delenv("EVAL_ANSWER_API_KEY", raising=False)
    monkeypatch.delenv("EVAL_ANSWER_BASE_URL", raising=False)
    monkeypatch.setenv("ANSWER_API_KEY", "legacy-key")
    monkeypatch.setenv("ANSWER_BASE_URL", "http://legacy")
    monkeypatch.setenv("OPENAI_API_KEY", "shared-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://shared")
    captured = _client_config(tmp_path, monkeypatch)
    assert captured == {"api_key": "legacy-key", "base_url": "http://legacy"}
