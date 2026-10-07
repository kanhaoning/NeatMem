"""Tests for storage.activity — ActivityStore events + projection."""

import pytest

from neatmem.storage.activity import (
    ActivityStore,
    EVICTION_DERIVED,
    EVICTION_MANUAL,
    KIND_INJECTION,
    KIND_JUDGMENT,
    KIND_SEARCH,
)


@pytest.fixture()
def store(tmp_path):
    s = ActivityStore(str(tmp_path / "activity.db"))
    yield s
    s.close()


def _inject(store, event_memory_ids=("m1",), source="user_prompt", anchor="u1"):
    return store.record_event(
        KIND_INJECTION,
        user_id="u",
        payload={"memory_ids": list(event_memory_ids), "source": source, "anchor": anchor},
    )


def test_record_and_iter_events(store):
    eid = store.record_event(KIND_SEARCH, user_id="u", payload={"query": "q"})
    assert eid == 1
    events = list(store.iter_events(KIND_SEARCH, user_id="u"))
    assert len(events) == 1
    assert events[0]["payload"] == {"query": "q"}


def test_pending_injections_excludes_judged(store):
    eid = _inject(store)
    assert [e["id"] for e in store.pending_injections()] == [eid]
    store.record_event(KIND_JUDGMENT, user_id="u", subject_id="m1",
                       payload={"injection_event_id": eid, "verdict": "used"})
    assert store.pending_injections() == []


def test_pending_injections_marker_event_clears_queue(store):
    # no_window / skipped markers have subject_id=None and no verdict.
    eid = _inject(store)
    store.record_event(KIND_JUDGMENT, payload={"injection_event_id": eid,
                                               "skipped": "no_window"})
    assert store.pending_injections() == []


def test_upsert_feedback_counts_increments(store):
    store.upsert_feedback_counts("m1", inject_delta=1, used_delta=0)
    store.upsert_feedback_counts("m1", inject_delta=1, used_delta=1)
    row = store.get_feedback("m1")
    assert row["inject_count"] == 2
    assert row["used_count"] == 1
    assert row["eviction_state"] == "none"


def test_set_eviction_validates_state(store):
    with pytest.raises(ValueError):
        store.set_eviction("m1", "bogus")


def test_list_evicted_ids_covers_derived_and_manual(store):
    store.set_eviction("m1", EVICTION_DERIVED)
    store.set_eviction("m2", EVICTION_MANUAL)
    store.upsert_feedback_counts("m3", inject_delta=1)  # stays 'none'
    assert sorted(store.list_evicted_ids()) == ["m1", "m2"]
    store.set_eviction("m1", "none")  # restore
    assert store.list_evicted_ids() == ["m2"]


def test_list_feedback_filters(store):
    store.upsert_feedback_counts("m1", inject_delta=3)
    store.upsert_feedback_counts("m2", inject_delta=12)
    store.set_eviction("m2", EVICTION_DERIVED)
    assert [r["memory_id"] for r in store.list_feedback(min_injections=10)] == ["m2"]
    assert [r["memory_id"] for r in store.list_feedback(eviction_state=EVICTION_DERIVED)] == ["m2"]
