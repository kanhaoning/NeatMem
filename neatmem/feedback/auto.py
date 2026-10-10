"""Serve-side auto judge: run the offline judge batch on a background thread.

Plan: docs/internal-notes/20261010-feedback-capture-rename-auto-judge-plan.md
§3. This is the manual/cron `neatmem feedback judge` loop built into serve —
a deployment-form change only, NOT online judging: the thread never touches
the request path, only judges already-closed windows, and its failures are
isolated from serving.

Failure policy (rule 7 in thread context): one batch raising fails that batch
loudly (logged with traceback) and the loop waits for the next interval — the
error is never swallowed. If the loop itself somehow dies, a supervising
wrapper restarts it with capped backoff and logs CRITICAL.
"""

import logging
import threading
from typing import Any, Callable, Dict, List

from neatmem.feedback.runner import run_judge_batch

logger = logging.getLogger(__name__)

# Restart backoff ladder for the supervising wrapper; capped at the last rung.
_RESTART_BACKOFFS = (5, 30, 120)


def _loop(
    store,
    load_messages: Callable[[Dict[str, Any]], List[Dict[str, Any]]],
    interval_seconds: int,
    stop: threading.Event,
    run_batch: Callable[..., Dict[str, int]],
) -> None:
    """Judge one batch per interval until stopped. First batch after one
    interval — serve warm-up (extraction/collection boot) goes first."""
    while not stop.wait(interval_seconds):
        try:
            summary = run_batch(store, load_messages)
            if any(summary[k] for k in ("judged", "failed", "no_window",
                                        "skipped_claimed")):
                logger.info("[feedback] auto judge batch: %s", summary)
        except Exception:
            logger.exception(
                "[feedback] auto judge batch failed; retrying next interval"
            )


def start_auto_judge(
    store,
    load_messages: Callable[[Dict[str, Any]], List[Dict[str, Any]]],
    interval_seconds: int,
    *,
    run_batch: Callable[..., Dict[str, int]] = run_judge_batch,
) -> threading.Event:
    """Start the daemon auto-judge thread; returns its stop Event.

    run_batch is injectable for tests. The caller owns `stop` — set it on
    shutdown (daemon thread also dies with the process, so this is only for
    clean tests/restarts).
    """
    stop = threading.Event()

    def guarded() -> None:
        backoff_idx = 0
        while not stop.is_set():
            try:
                _loop(store, load_messages, interval_seconds, stop, run_batch)
            except Exception:
                # _loop already swallows per-batch failures; reaching here
                # means the loop machinery itself broke — restart it.
                logger.critical(
                    "[feedback] auto judge thread died; restarting",
                    exc_info=True,
                )
            if stop.is_set():
                break
            delay = _RESTART_BACKOFFS[min(backoff_idx, len(_RESTART_BACKOFFS) - 1)]
            backoff_idx += 1
            stop.wait(delay)

    thread = threading.Thread(
        target=guarded, name="feedback-auto-judge", daemon=True
    )
    thread.start()
    logger.info(
        "[feedback] auto judge thread started (interval=%ss)", interval_seconds
    )
    return stop
