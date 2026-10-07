"""Tests for feedback.runner — the offline judge batch."""

import pytest

from neatmem import config
from neatmem.feedback.runner import run_judge_batch
from neatmem.storage.activity import (
    ActivityStore,
    EVICTION_DERIVED,
    KIND_INJECTION,
    KIND_JUDGMENT,
)


@pytest.fixture(autouse=True)
def _judge_model_env(monkeypatch):
    # make_judge_client resolves MEMORY_FEEDBACK_JUDGE_MODEL or LLM_MODEL.
    monkeypatch.setenv("LLM_MODEL", "test-judge-model")


@pytest.fixture()
def store(tmp_path):
    s = ActivityStore(str(tmp_path / "activity.db"))
    yield s
    s.close()


def _messages():
    return [
        {"message_id": "u1", "role": "user", "content": "q1"},
        {"message_id": "a1", "role": "assistant", "content": "ans1"},
        {"message_id": "u2", "role": "user", "content": "q2"},
        {"message_id": "a2", "role": "assistant", "content": "ans2"},
    ]


def _load_messages(_filters):
    return _messages()


def _inject(store, memory_ids, source="user_prompt", anchor="u1"):
    return store.record_event(
        KIND_INJECTION,
        user_id="u",
        payload={
            "memory_ids": list(memory_ids),
            "memories": [{"id": m, "text": f"text of {m}"} for m in memory_ids],
            "source": source,
            "anchor": anchor,
        },
    )


def _fake_judge(verdict_map):
    def judge(client, model, question, memories, answer, template=None):
        return {
            "verdicts": {m["id"]: verdict_map.get(m["id"], "unused") for m in memories},
            "evidence": {m["id"]: "" for m in memories},
        }

    return judge


def test_judged_injection_updates_counters(store):
    _inject(store, ["m1", "m2"])
    summary = run_judge_batch(
        store, _load_messages, client=object(),
        judge_fn=_fake_judge({"m1": "used"}),
    )
    assert summary["judged"] == 1
    assert store.get_feedback("m1")["used_count"] == 1
    assert store.get_feedback("m2")["used_count"] == 0
    assert store.get_feedback("m2")["inject_count"] == 1
    # per-memory judgment events with back-reference
    judgments = list(store.iter_events(KIND_JUDGMENT, subject_id="m1"))
    assert judgments[0]["payload"]["verdict"] == "used"
    assert "injection_event_id" in judgments[0]["payload"]
    assert store.pending_injections() == []


def test_midtask_source_skipped_without_counting(store):
    _inject(store, ["m1"], source="midtask")
    summary = run_judge_batch(
        store, _load_messages, client=object(), judge_fn=_fake_judge({}),
    )
    assert summary["skipped_source"] == 1
    assert store.get_feedback("m1") is None
    assert store.pending_injections() == []


def test_no_window_skips_counters(store):
    _inject(store, ["m1"], anchor="u2")
    # u2's window HAS an assistant turn (a2) → judged. Use an anchor with none:
    store2_eid = _inject(store, ["m9"], anchor="u-missing")
    summary = run_judge_batch(
        store, lambda f: _messages(), client=object(), judge_fn=_fake_judge({}),
    )
    assert summary["judged"] == 1  # m1 judged
    assert summary["no_window"] == 1  # m9 anchor missing
    assert store.get_feedback("m9") is None


def test_eviction_gate_marks_derived(store, monkeypatch):
    monkeypatch.setattr(config, "MEMORY_FEEDBACK_EVICTION_MIN_INJECTIONS", 2)
    _inject(store, ["m1"])
    run_judge_batch(store, _load_messages, client=object(),
                    judge_fn=_fake_judge({}), min_injections=2, eviction_enabled=True)
    assert store.get_feedback("m1")["eviction_state"] == "none"  # 1 < 2
    _inject(store, ["m1"])
    run_judge_batch(store, _load_messages, client=object(),
                    judge_fn=_fake_judge({}), min_injections=2, eviction_enabled=True)
    assert store.get_feedback("m1")["eviction_state"] == EVICTION_DERIVED


def test_eviction_gate_respects_used_and_manual(store):
    store.upsert_feedback_counts("m1", inject_delta=9, used_delta=1)
    _inject(store, ["m1"])
    run_judge_batch(store, _load_messages, client=object(),
                    judge_fn=_fake_judge({}), min_injections=10, eviction_enabled=True)
    assert store.get_feedback("m1")["eviction_state"] == "none"  # used_count > 0
    store.set_eviction("m2", "manual")
    store.upsert_feedback_counts("m2", inject_delta=9)
    _inject(store, ["m2"])
    run_judge_batch(store, _load_messages, client=object(),
                    judge_fn=_fake_judge({}), min_injections=10, eviction_enabled=True)
    assert store.get_feedback("m2")["eviction_state"] == "manual"  # never auto-touched


def test_eviction_disabled_never_marks(store):
    _inject(store, ["m1"])
    run_judge_batch(store, _load_messages, client=object(),
                    judge_fn=_fake_judge({}), min_injections=1, eviction_enabled=False)
    assert store.get_feedback("m1")["eviction_state"] == "none"


def test_judge_error_leaves_injection_pending(store):
    from neatmem.feedback.judge import JudgeError

    def boom(client, model, question, memories, answer, template=None):
        raise JudgeError("simulated failure")

    _inject(store, ["m1"])
    summary = run_judge_batch(store, _load_messages, client=object(), judge_fn=boom)
    assert summary["failed"] == 1
    assert len(store.pending_injections()) == 1  # retry next batch
    assert store.get_feedback("m1") is None  # no partial counters
