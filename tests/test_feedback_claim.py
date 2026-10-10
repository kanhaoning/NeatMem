"""Tests for judgment-claim mutual exclusion (plan 20261010 §3.2).

The serve auto-judge thread and manual `feedback judge` runs both scan the
pending queue; claims guarantee an injection is judged once. Double judging
would double-count inject/used, so exclusivity is a correctness property.
"""

import pytest

from concurrent.futures import ThreadPoolExecutor

from neatmem.feedback.runner import run_judge_batch
from neatmem.storage.activity import (
    ActivityStore,
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


_MESSAGES = [
    {"message_id": "u1", "role": "user", "content": "q1"},
    {"message_id": "a1", "role": "assistant", "content": "ans1"},
]


def _load_messages(_filters):
    return _MESSAGES


def _fake_judge(client, model, question, memories, answer, template=None):
    return {
        "verdicts": {m["id"]: "used" for m in memories},
        "evidence": {m["id"]: "" for m in memories},
    }


def test_claim_roundtrip(store):
    inj = _inject(store, ["m1"])
    assert store.claim_injection(inj) is True
    # Second claim while the first is live: refused.
    assert store.claim_injection(inj) is False
    assert store.live_judge_claims() == [inj]


def test_claim_refused_after_judgment(store):
    inj = _inject(store, ["m1"])
    store.record_event(KIND_JUDGMENT, subject_id="m1",
                       payload={"injection_event_id": inj, "verdict": "used"})
    assert store.claim_injection(inj) is False
    assert store.live_judge_claims() == []


def test_expired_claim_is_reclaimable(store):
    inj = _inject(store, ["m1"])
    assert store.claim_injection(inj, ttl_seconds=0) is True
    # ttl=0 expires immediately — a crashed runner's claim must not wedge the
    # queue forever.
    assert store.claim_injection(inj, ttl_seconds=0) is True


def test_concurrent_claims_single_winner(store):
    inj = _inject(store, ["m1"])
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: store.claim_injection(inj), range(16)))
    assert results.count(True) == 1


def test_runner_skips_claimed_injection(store):
    inj = _inject(store, ["m1"])
    # Simulate the auto-judge thread holding the claim.
    assert store.claim_injection(inj) is True
    summary = run_judge_batch(store, _load_messages, judge_fn=_fake_judge)
    assert summary["skipped_claimed"] == 1
    assert summary["judged"] == 0
    # Untouched by the other runner: no judgment, no counter row.
    assert store.get_feedback("m1") is None


def test_judged_injection_not_reclaimed(store):
    _inject(store, ["m1"])
    summary = run_judge_batch(store, _load_messages, judge_fn=_fake_judge)
    assert summary["judged"] == 1
    row = store.get_feedback("m1")
    assert row["inject_count"] == 1 and row["used_count"] == 1
    # A second batch (e.g. manual + thread overlap) must not double-count.
    summary2 = run_judge_batch(store, _load_messages, judge_fn=_fake_judge)
    assert summary2["pending"] == 0
    row = store.get_feedback("m1")
    assert row["inject_count"] == 1 and row["used_count"] == 1
