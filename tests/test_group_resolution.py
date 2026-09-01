"""Unit tests for group resolution (multi-target update + rewrite).

Covers the 2026-09-01 landing: when the listwise_multitarget detector judges
>=2 updates in one call and DEDUP_RESOLVER=rewrite, the resolver fuses the
new fact + all targets in one _memos_resolve_group call — merged text goes to
the highest-score target, the rest are deleted. On "No" / error the caller
falls back to the per-target rewrite loop.

Run:  cd <repo root> && python -m pytest tests/test_group_resolution.py -q
"""

import types

import pytest

import neatmem.memory_add as ma


class _FakeMemory:
    def __init__(self):
        self.updates = []  # [(memory_id, data)]
        self.deletes = []  # [memory_id]

    def update(self, memory_id, data, metadata=None):
        self.updates.append((memory_id, data))

    def delete(self, memory_id):
        self.deletes.append(memory_id)


def _resp(content):
    msg = types.SimpleNamespace(content=content)
    choice = types.SimpleNamespace(message=msg, finish_reason="stop")
    return types.SimpleNamespace(choices=[choice])


DETECTOR_TWO_UPDATES = (
    '{"judgments": [{"action": "update", "targetId": "1", "reason": "a"},'
    ' {"action": "update", "targetId": "2", "reason": "b"}]}'
)

CANDIDATES = [
    {"id": "id-1", "score": 0.80,
     "payload": {"data": "old memory one", "created_at": "2026-08-01"}},
    {"id": "id-2", "score": 0.70,
     "payload": {"data": "old memory two", "created_at": "2026-08-02"}},
]


@pytest.fixture
def mt_rewrite_env(monkeypatch):
    monkeypatch.setattr(ma, "DEDUP_RESOLVER", "rewrite")
    monkeypatch.setattr(ma, "_DEDUP_MT", True)
    monkeypatch.setattr(ma, "DEDUP_DRY_RUN", False)
    monkeypatch.setattr(ma, "DEDUP_RECALL_THRESHOLD", 0.40)
    monkeypatch.setattr(
        ma, "search_memories",
        lambda **kwargs: {"results": [dict(c) for c in CANDIDATES]},
    )


def _run(memory, scripted_llm):
    return ma.dedup_memories_action(
        memory=memory,
        openai_client=None,
        llm_model="fake-model",
        extracted_memories=[{"text": "new fact covering both olds"}],
        search_filters={"user_id": "u"},
        req_id="t",
    )


def test_group_fusion_writes_primary_deletes_rest(mt_rewrite_env, monkeypatch):
    calls = iter([DETECTOR_TWO_UPDATES, "<answer>merged group text</answer>"])
    monkeypatch.setattr(ma, "complete_chat",
                        lambda *a, **k: _resp(next(calls)))
    mem = _FakeMemory()
    result = _run(mem, None)

    assert mem.updates == [("id-1", "merged group text")]  # highest score target
    assert mem.deletes == ["id-2"]
    relations = [d["relation"] for d in result.duplicates]
    assert relations == ["update_rewrite_group_primary", "update_rewrite_group_deleted"]
    assert result.to_add == []


def test_group_no_falls_back_to_per_target(mt_rewrite_env, monkeypatch):
    calls = iter([
        DETECTOR_TWO_UPDATES,
        "<answer>No</answer>",               # group resolver declines
        "<answer>merged one</answer>",       # per-target rewrite #1
        "<answer>merged two</answer>",       # per-target rewrite #2
    ])
    monkeypatch.setattr(ma, "complete_chat",
                        lambda *a, **k: _resp(next(calls)))
    mem = _FakeMemory()
    result = _run(mem, None)

    assert sorted(uid for uid, _ in mem.updates) == ["id-1", "id-2"]
    assert mem.deletes == []
    assert all(d["relation"] == "update_rewrite" for d in result.duplicates)


def test_group_error_falls_back_to_per_target(mt_rewrite_env, monkeypatch):
    responses = iter([
        _resp(DETECTOR_TWO_UPDATES),
        RuntimeError("boom"),                # group resolver raises
        _resp("<answer>merged one</answer>"),
        _resp("<answer>merged two</answer>"),
    ])

    def fake_complete(*a, **k):
        item = next(responses)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(ma, "complete_chat", fake_complete)
    mem = _FakeMemory()
    result = _run(mem, None)

    assert sorted(uid for uid, _ in mem.updates) == ["id-1", "id-2"]
    assert mem.deletes == []
    assert all(d["relation"] == "update_rewrite" for d in result.duplicates)


def test_single_update_does_not_trigger_group(mt_rewrite_env, monkeypatch):
    detector = '{"judgments": [{"action": "update", "targetId": "1", "reason": "a"},' \
               ' {"action": "keep", "targetId": "2", "reason": "b"}]}'
    calls = iter([detector, "<answer>merged one</answer>"])
    monkeypatch.setattr(ma, "complete_chat",
                        lambda *a, **k: _resp(next(calls)))
    mem = _FakeMemory()
    result = _run(mem, None)

    assert mem.updates == [("id-1", "merged one")]
    assert mem.deletes == []
    assert [d["relation"] for d in result.duplicates] == ["update_rewrite"]
