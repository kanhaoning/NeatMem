"""Tests for feedback.auto — the serve-side auto-judge thread (plan 20261010 §3)."""

import threading
import time

from neatmem.feedback.auto import start_auto_judge


def test_thread_runs_batches_until_stopped():
    calls = []
    ran_twice = threading.Event()

    def fake_batch(store, load_messages):
        calls.append(1)
        if len(calls) >= 2:
            ran_twice.set()
        return {"pending": 0, "judged": 1, "no_window": 0,
                "skipped_source": 0, "skipped_claimed": 0, "failed": 0,
                "evicted": 0}

    stop = start_auto_judge(None, lambda f: [], 0.05, run_batch=fake_batch)
    try:
        assert ran_twice.wait(timeout=5), "thread did not run repeated batches"
    finally:
        stop.set()


def test_batch_failure_does_not_kill_loop():
    calls = []
    recovered = threading.Event()

    def flaky_batch(store, load_messages):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("upstream 529")
        recovered.set()
        return {"pending": 0, "judged": 0, "no_window": 0,
                "skipped_source": 0, "skipped_claimed": 0, "failed": 0,
                "evicted": 0}

    stop = start_auto_judge(None, lambda f: [], 0.05, run_batch=flaky_batch)
    try:
        assert recovered.wait(timeout=5), "loop died on first batch failure"
    finally:
        stop.set()


def test_stop_event_ends_thread():
    def fake_batch(store, load_messages):
        return {"pending": 0, "judged": 0, "no_window": 0,
                "skipped_source": 0, "skipped_claimed": 0, "failed": 0,
                "evicted": 0}

    stop = start_auto_judge(None, lambda f: [], 3600, run_batch=fake_batch)
    thread = [t for t in threading.enumerate() if t.name == "feedback-auto-judge"]
    assert thread, "auto judge thread not started"
    stop.set()
    thread[0].join(timeout=5)
    assert not thread[0].is_alive()
    # No batch should have run within the long interval.
    time.sleep(0.1)
