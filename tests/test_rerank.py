"""Unit tests for the rerank module (RERANK_MODE dispatcher, LLM listwise /
pointwise, cross-encoder multi-provider + local).

HTTP is mocked by patching neatmem.rerank.httpx.post; LLM calls by patching
neatmem.rerank.complete_chat. Env-validation tests use importlib.reload with
monkeypatched env and restore the module afterwards.

Run:  cd <repo root> && python -m pytest tests/test_rerank.py -x -q
"""

import importlib
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from neatmem import rerank as rerank_mod
from neatmem.rerank import (
    llm_rerank,
    _call_rerank_api,
    _cross_encoder_rerank,
    _llm_rerank_pointwise,
    LLMRerankResult,
)

_RERANK_ENVS = (
    "RERANK_MODE", "LLM_RERANK_MODE", "LLM_RERANK_CANDS", "LLM_RERANK_CAND_TEXT_LEN",
    "CROSS_ENCODER_PROVIDER", "CROSS_ENCODER_MODEL", "CROSS_ENCODER_BASE_URL",
    "CROSS_ENCODER_API_KEY", "CROSS_ENCODER_MODE", "CROSS_ENCODER_CANDS",
    "CROSS_ENCODER_CAND_TEXT_LEN", "CROSS_ENCODER_REL_THRESHOLD",
)


def _reload(monkeypatch, **env):
    """Reload rerank with a clean rerank env plus the given overrides."""
    for k in _RERANK_ENVS:
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    importlib.reload(rerank_mod)


@pytest.fixture
def restore_rerank(monkeypatch):
    """Reload rerank with defaults after the test (for reload-based tests)."""
    yield
    for k in _RERANK_ENVS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.undo()
    importlib.reload(rerank_mod)


@pytest.fixture(autouse=True)
def _fake_ce_key(monkeypatch):
    """HTTP adapter tests need an API key; individual tests may override."""
    monkeypatch.setattr(rerank_mod, "CROSS_ENCODER_API_KEY", "test-key")


# ---------------------------------------------------------------------------
# Env validation / dispatch config
# ---------------------------------------------------------------------------


def test_default_mode_is_llm_listwise(monkeypatch, restore_rerank):
    _reload(monkeypatch)
    assert rerank_mod.RERANK_MODE == "llm"
    assert rerank_mod.LLM_RERANK_MODE == "listwise"
    assert rerank_mod.LLM_RERANK_CANDS == 20
    assert rerank_mod.CROSS_ENCODER_CANDS == 100


def test_invalid_rerank_mode_raises(monkeypatch, restore_rerank):
    with pytest.raises(ValueError, match="Invalid RERANK_MODE"):
        _reload(monkeypatch, RERANK_MODE="llm_listwise")  # legacy value is dead


def test_invalid_llm_rerank_mode_raises(monkeypatch, restore_rerank):
    with pytest.raises(ValueError, match="Invalid LLM_RERANK_MODE"):
        _reload(monkeypatch, LLM_RERANK_MODE="pairwise")


def test_invalid_cross_encoder_provider_raises(monkeypatch, restore_rerank):
    with pytest.raises(ValueError, match="Invalid CROSS_ENCODER_PROVIDER"):
        _reload(monkeypatch, CROSS_ENCODER_PROVIDER="openai")


def test_cross_encoder_listwise_fails_fast(monkeypatch, restore_rerank):
    """listwise is a reserved value: selecting it must raise at startup."""
    with pytest.raises(ValueError, match="listwise is reserved"):
        _reload(monkeypatch, RERANK_MODE="cross_encoder", CROSS_ENCODER_MODE="listwise")


# ---------------------------------------------------------------------------
# Dispatcher routing
# ---------------------------------------------------------------------------


def _make_docs(n: int):
    return [
        {"memory": f"doc {i}", "score": 0.9 - i * 0.01, "id": f"m{i}"}
        for i in range(n)
    ]


def test_dispatch_off_truncates(monkeypatch):
    monkeypatch.setattr(rerank_mod, "RERANK_MODE", "off")
    result = llm_rerank(None, "m", "q", _make_docs(5), top_k=2)
    assert [d["id"] for d in result.kept] == ["m0", "m1"]
    assert len(result.dropped) == 3


def test_dispatch_llm_listwise(monkeypatch):
    monkeypatch.setattr(rerank_mod, "RERANK_MODE", "llm")
    monkeypatch.setattr(rerank_mod, "LLM_RERANK_MODE", "listwise")
    with patch.object(rerank_mod, "_llm_rerank_listwise",
                      return_value=(["k"], ["d"])) as m:
        result = llm_rerank(None, "m", "q", _make_docs(3), top_k=2)
    m.assert_called_once()
    assert result.kept == ["k"] and result.dropped == ["d"]


def test_dispatch_llm_pointwise(monkeypatch):
    monkeypatch.setattr(rerank_mod, "RERANK_MODE", "llm")
    monkeypatch.setattr(rerank_mod, "LLM_RERANK_MODE", "pointwise")
    with patch.object(rerank_mod, "_llm_rerank_pointwise",
                      return_value=(["k"], [])) as m:
        result = llm_rerank(None, "m", "q", _make_docs(3), top_k=2)
    m.assert_called_once()
    assert result.kept == ["k"]


def test_dispatch_cross_encoder(monkeypatch):
    monkeypatch.setattr(rerank_mod, "RERANK_MODE", "cross_encoder")
    with patch.object(rerank_mod, "_cross_encoder_rerank",
                      return_value=LLMRerankResult(kept=["k"], dropped=[])) as m:
        result = llm_rerank(None, "m", "q", _make_docs(3), top_k=2)
    m.assert_called_once()
    assert result.kept == ["k"]


# ---------------------------------------------------------------------------
# Pointwise LLM rerank
# ---------------------------------------------------------------------------


def _fake_llm_response(text: str):
    resp = MagicMock()
    resp.choices[0].message.content = text
    return resp


def test_pointwise_sorts_by_score(monkeypatch):
    monkeypatch.setattr(rerank_mod, "LLM_RERANK_CANDS", 3)
    scores = iter([3, 9, 5])
    with patch.object(rerank_mod, "complete_chat",
                      side_effect=lambda *a, **k: _fake_llm_response(f'{{"score": {next(scores)}}}')):
        kept, dropped = _llm_rerank_pointwise(None, "m", "q", _make_docs(5), top_k=5)
    # head = m0,m1,m2 scored 3/9/5 -> sorted m1,m2,m0; tail m3,m4 appended.
    assert [d["id"] for d in kept] == ["m1", "m2", "m0", "m3", "m4"]
    assert dropped == []


def test_pointwise_parse_failure_sinks_to_bottom(monkeypatch):
    monkeypatch.setattr(rerank_mod, "LLM_RERANK_CANDS", 3)
    responses = iter(['{"score": 8}', "garbage", '{"score": 5}'])
    with patch.object(rerank_mod, "complete_chat",
                      side_effect=lambda *a, **k: _fake_llm_response(next(responses))):
        kept, dropped = _llm_rerank_pointwise(None, "m", "q", _make_docs(3), top_k=3)
    # m1 unparseable -> -inf, stays in kept below scored docs.
    assert [d["id"] for d in kept] == ["m0", "m2", "m1"]
    assert dropped == []


def test_pointwise_empty_input():
    kept, dropped = _llm_rerank_pointwise(None, "m", "q", [], top_k=5)
    assert kept == [] and dropped == []


# ---------------------------------------------------------------------------
# Cross-encoder HTTP adapter
# ---------------------------------------------------------------------------


class _FakeResponse:
    """Minimal stand-in for httpx.Response used by _call_rerank_api."""

    def __init__(self, status_code: int, json_data=None, text: str = ""):
        self.status_code = status_code
        self._json = json_data
        self.text = text or ""

    def json(self):
        if self._json is None:
            raise ValueError("no JSON")
        return self._json

    def raise_for_status(self):
        if 400 <= self.status_code < 600:
            raise RuntimeError(f"HTTP {self.status_code}: {self.text}")


def test_call_realigns_scores_by_index():
    """The API returns results sorted desc by score. We must realign by `index`
    so scores[i] corresponds to input doc_texts[i], not score order."""
    fake_resp = _FakeResponse(
        200,
        {
            "results": [
                {"index": 2, "relevance_score": 0.9},
                {"index": 0, "relevance_score": 0.5},
                {"index": 1, "relevance_score": 0.1},
            ],
            "meta": {"tokens": {"input_tokens": 100}},
        },
    )
    with patch("neatmem.rerank.httpx.post", return_value=fake_resp):
        scores = _call_rerank_api("q", ["d0", "d1", "d2"])
    assert scores == [0.5, 0.1, 0.9]


def test_call_rejects_empty_docs():
    with pytest.raises(ValueError, match="non-empty"):
        _call_rerank_api("q", [])


def test_call_rejects_missing_api_key(monkeypatch):
    monkeypatch.setattr(rerank_mod, "CROSS_ENCODER_API_KEY", "")
    with pytest.raises(ValueError, match="CROSS_ENCODER_API_KEY not set"):
        _call_rerank_api("q", ["d1"])


def test_call_quota_exhausted_2056_no_retry():
    """0718 lesson: code 2056 = quota exhausted (5h rolling window).
    Must raise immediately, NOT retry 3 times."""
    fake_resp = _FakeResponse(
        200, {"code": 2056, "message": "quota exhausted", "data": None},
    )
    mock_post = MagicMock(return_value=fake_resp)
    with patch("neatmem.rerank.httpx.post", mock_post):
        with pytest.raises(RuntimeError, match="quota exhausted"):
            _call_rerank_api("q", ["d1", "d2"])
    # Critical: only 1 call, not 3 retries.
    assert mock_post.call_count == 1


def test_call_429_retries_3_times_then_fails():
    """429 must retry 3 times with backoff, then raise RuntimeError."""
    fake_resp = _FakeResponse(429, text="rate limit")
    mock_post = MagicMock(return_value=fake_resp)
    with patch("neatmem.rerank.httpx.post", mock_post):
        with patch("neatmem.rerank.time.sleep") as mock_sleep:
            with pytest.raises(RuntimeError, match="failed after 3 retries"):
                _call_rerank_api("q", ["d1"])
    assert mock_post.call_count == 3
    # 3 retries -> 3 sleeps (1s, 2s, 4s).
    assert mock_sleep.call_count == 3


def test_call_5xx_retries_then_succeeds():
    """5xx retries; if 3rd attempt succeeds, return scores."""
    fail_resp = _FakeResponse(503, text="server error")
    ok_resp = _FakeResponse(
        200,
        {"results": [{"index": 0, "relevance_score": 0.7}]},
    )
    mock_post = MagicMock(side_effect=[fail_resp, fail_resp, ok_resp])
    with patch("neatmem.rerank.httpx.post", mock_post):
        with patch("neatmem.rerank.time.sleep"):
            scores = _call_rerank_api("q", ["d1"])
    assert scores == [0.7]
    assert mock_post.call_count == 3


def test_call_detects_silent_drop():
    """If the API returns fewer results than docs sent, fail loudly per
    CLAUDE.md rule 7 (no silent propagation of error state)."""
    fake_resp = _FakeResponse(
        200,
        # Only 1 result for 3 docs = silent drop.
        {"results": [{"index": 0, "relevance_score": 0.5}]},
    )
    with patch("neatmem.rerank.httpx.post", return_value=fake_resp):
        with pytest.raises(RuntimeError, match="silent drop"):
            _call_rerank_api("q", ["d0", "d1", "d2"])


def test_call_missing_index_raises():
    """Each result must have `index`; missing means contract changed."""
    fake_resp = _FakeResponse(
        200,
        {"results": [{"relevance_score": 0.5}]},  # no index field
    )
    with patch("neatmem.rerank.httpx.post", return_value=fake_resp):
        with pytest.raises(RuntimeError, match="missing index"):
            _call_rerank_api("q", ["d0"])


def test_call_non_siliconflow_skips_error_envelope(monkeypatch):
    """The {code: ...} business-error envelope is SiliconFlow-specific; other
    providers must not have their payloads misread as errors."""
    monkeypatch.setattr(rerank_mod, "CROSS_ENCODER_PROVIDER", "jina")
    fake_resp = _FakeResponse(
        200,
        {"results": [{"index": 0, "relevance_score": 0.7}]},
    )
    with patch("neatmem.rerank.httpx.post", return_value=fake_resp):
        assert _call_rerank_api("q", ["d0"]) == [0.7]


# ---------------------------------------------------------------------------
# _cross_encoder_rerank (head/tail, threshold filter, mem0 style)
# ---------------------------------------------------------------------------


def test_cross_encoder_head_tail_split(monkeypatch):
    """With CROSS_ENCODER_CANDS=3 and 5 docs, only top 3 go to the scorer;
    tail 2 appended after kept_head."""
    monkeypatch.setattr(rerank_mod, "CROSS_ENCODER_CANDS", 3)
    monkeypatch.setattr(rerank_mod, "CROSS_ENCODER_REL_THRESHOLD", 0.0)

    def fake_call(query, doc_texts):
        return [0.9 - i * 0.1 for i in range(len(doc_texts))]

    with patch("neatmem.rerank._call_rerank_api", side_effect=fake_call):
        result = _cross_encoder_rerank("q", _make_docs(5), top_k=5)

    # rerank_mod.* (not the top-level import) because reload tests replace the
    # class object on the shared module namespace.
    assert isinstance(result, rerank_mod.LLMRerankResult)
    # head (3) + tail (2) = 5 in kept.
    assert len(result.kept) == 5
    assert result.dropped == []


def test_cross_encoder_threshold_filter(monkeypatch):
    """threshold > 0 drops docs below ratio * top_score."""
    monkeypatch.setattr(rerank_mod, "CROSS_ENCODER_CANDS", 5)
    monkeypatch.setattr(rerank_mod, "CROSS_ENCODER_REL_THRESHOLD", 0.5)

    # Mock scores: top=1.0, second=0.6 (>=0.5 threshold), third=0.3 (dropped).
    def fake_call(query, doc_texts):
        return [1.0, 0.6, 0.3, 0.1, 0.05]

    with patch("neatmem.rerank._call_rerank_api", side_effect=fake_call):
        result = _cross_encoder_rerank("q", _make_docs(5), top_k=5)

    # threshold = 1.0 * 0.5 = 0.5; kept_head has 2 docs (1.0, 0.6), dropped 3.
    kept_scores = [d["rerank_score"] for d in result.kept if "rerank_score" in d]
    assert kept_scores == [1.0, 0.6]
    assert len(result.dropped) == 3


def test_cross_encoder_mem0_style_no_filter(monkeypatch):
    """threshold=0 = mem0 style = pure sort, no filter."""
    monkeypatch.setattr(rerank_mod, "CROSS_ENCODER_CANDS", 3)
    monkeypatch.setattr(rerank_mod, "CROSS_ENCODER_REL_THRESHOLD", 0.0)

    # Mock scores in reverse so rerank reorders.
    def fake_call(query, doc_texts):
        return [0.1, 0.5, 0.9]

    with patch("neatmem.rerank._call_rerank_api", side_effect=fake_call):
        result = _cross_encoder_rerank("q", _make_docs(3), top_k=3)

    # All 3 kept (no filter), sorted by rerank_score desc.
    kept_scored = [d for d in result.kept if "rerank_score" in d]
    assert len(kept_scored) == 3
    assert [d["rerank_score"] for d in kept_scored] == [0.9, 0.5, 0.1]
    assert result.dropped == []


def test_cross_encoder_empty_input():
    """Empty documents returns empty LLMRerankResult without calling API."""
    with patch("neatmem.rerank._call_rerank_api") as mock_call:
        result = _cross_encoder_rerank("q", [], top_k=5)
    assert result.kept == [] and result.dropped == []
    mock_call.assert_not_called()


def test_cross_encoder_local_dispatch(monkeypatch):
    """provider=local routes to the in-process scorer, not the HTTP API."""
    monkeypatch.setattr(rerank_mod, "CROSS_ENCODER_PROVIDER", "local")
    monkeypatch.setattr(rerank_mod, "CROSS_ENCODER_REL_THRESHOLD", 0.0)
    with patch("neatmem.rerank._call_local_rerank", return_value=[0.5, 0.9]) as m_local, \
         patch("neatmem.rerank._call_rerank_api") as m_api:
        result = _cross_encoder_rerank("q", _make_docs(2), top_k=2)
    m_local.assert_called_once()
    m_api.assert_not_called()
    assert [d["id"] for d in result.kept] == ["m1", "m0"]


def test_local_rerank_uses_sentence_transformers(monkeypatch):
    """local scorer loads CrossEncoder once and returns float scores."""
    fake_st = types.ModuleType("sentence_transformers")

    class _FakeCrossEncoder:
        def __init__(self, model, device=None):
            self.model, self.device = model, device

        def predict(self, pairs, batch_size=32):
            return [0.1 * i for i, _ in enumerate(pairs)]

    fake_st.CrossEncoder = _FakeCrossEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st)
    monkeypatch.setattr(rerank_mod, "_LOCAL_MODEL", None)
    monkeypatch.setattr(rerank_mod, "CROSS_ENCODER_MODEL", "m-local")
    monkeypatch.setattr(rerank_mod, "CROSS_ENCODER_DEVICE", "cpu")

    scores = rerank_mod._call_local_rerank("q", ["a", "b"])
    assert scores == [0.0, 0.1]
    # Second call reuses the cached model instance.
    assert rerank_mod._LOCAL_MODEL is not None
    monkeypatch.setattr(rerank_mod, "_LOCAL_MODEL", None)
