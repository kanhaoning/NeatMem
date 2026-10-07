"""Tests for feedback.judge — verdict mapping, fail-loud paths, chunk split."""

import json

import pytest

from neatmem.feedback.judge import (
    JudgeError,
    _extract_json,
    judge_memories,
    load_judge_template,
)


class _FakeChoice:
    def __init__(self, content, finish_reason="stop"):
        self.message = type("M", (), {"content": content})
        self.finish_reason = finish_reason


class _FakeCompletions:
    def __init__(self, responder):
        self.responder = responder
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        content, finish = self.responder(kwargs)
        return type("R", (), {"choices": [_FakeChoice(content, finish)]})


class _FakeClient:
    def __init__(self, responder):
        self.chat = type("C", (), {"completions": _FakeCompletions(responder)})


def _responder_ok(payload):
    def respond(_kwargs):
        return json.dumps(payload), "stop"

    return respond


MEMORIES = [{"id": "uuid-aaa", "text": "t1"}, {"id": "uuid-bbb", "text": "t2"}]


def test_verdict_mapping_and_evidence_passthrough():
    payload = {"verdicts": [
        {"memory_id": "M1", "verdict": "used", "evidence": "ans quote"},
        {"memory_id": "M2", "verdict": "unused", "evidence": ""},
    ]}
    out = judge_memories(_FakeClient(_responder_ok(payload)), "m", "q", MEMORIES, "a")
    assert out["verdicts"] == {"uuid-aaa": "used", "uuid-bbb": "unused"}
    assert out["evidence"]["uuid-aaa"] == "ans quote"


def test_unknown_label_raises_and_retries_then_splits():
    # Label miscopy (hallucinated M9) is a JudgeError → 3 attempts → split
    # into singletons; singleton still miscopies → JudgeError propagates.
    bad = {"verdicts": [{"memory_id": "M9", "verdict": "used", "evidence": ""}]}
    client = _FakeClient(_responder_ok(bad))
    with pytest.raises(JudgeError, match="unknown memory label"):
        judge_memories(client, "m", "q", MEMORIES, "a")
    assert client.chat.completions.calls > 3  # retried, then split


def test_truncation_raises_loud():
    def respond(_kwargs):
        return '{"verdicts": [', "length"

    with pytest.raises(JudgeError, match="truncated"):
        judge_memories(_FakeClient(respond), "m", "q", MEMORIES[:1], "a")


def test_missing_verdict_raises():
    # Empty verdict list stays incomplete after retries and chunk splitting
    # (a singleton chunk has no missing labels only if the judge returns it).
    payload = {"verdicts": []}
    with pytest.raises(JudgeError, match="missing verdicts"):
        judge_memories(_FakeClient(_responder_ok(payload)), "m", "q", MEMORIES, "a")


def test_empty_memories_and_empty_answer_rejected():
    with pytest.raises(JudgeError, match="empty memories"):
        judge_memories(_FakeClient(_responder_ok({})), "m", "q", [], "a")
    with pytest.raises(JudgeError, match="empty answer"):
        judge_memories(_FakeClient(_responder_ok({})), "m", "q", MEMORIES, "  ")


def test_extract_json_recovers_trailing_garbage():
    text = '<think>reasoning</think>{"a": {"b": 1}} trailing {"broken'
    assert json.loads(_extract_json(text)) == {"a": {"b": 1}}


def test_extract_json_skips_broken_prefix():
    text = '{"unclosed": ... {"ok": true}'
    assert json.loads(_extract_json(text)) == {"ok": True}


def test_template_resolves_packaged_default():
    template = load_judge_template()
    for placeholder in ("{question}", "{memories_block}", "{answer}", "{memory_ids}"):
        assert placeholder in template
