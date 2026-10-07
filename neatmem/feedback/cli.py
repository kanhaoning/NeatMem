"""``neatmem feedback`` — offline judge batch + eviction-gate management.

Plan: docs/internal-notes/20261005-citation-feedback-memory-importance-plan.md
§5.6 (command surface). Four subcommands:

- ``judge``   run one offline judge batch over pending injections
- ``status``  projection summary: counters, evicted, evictable candidates
- ``evict``   manually evict a memory (eviction_state='manual', survives
              projection rebuilds)
- ``restore`` clear a memory's eviction (eviction_state='none')

All operate directly on ACTIVITY_DB_PATH (messages store for judge windows);
no running server required. Rule 16: judge LLM calls use max_tokens=32768
(enforced inside feedback.judge).
"""

import argparse
import sys
from typing import Any, Dict, List

from neatmem.storage.activity import EVICTION_MANUAL, EVICTION_NONE


def _open_store(args):
    from neatmem import config
    from neatmem.storage.activity import ActivityStore

    return ActivityStore(args.activity_db_path or config.ACTIVITY_DB_PATH)


def _load_scope_messages(message_store, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
    """All messages of a scope, chronological (paginated asc)."""
    out: List[Dict[str, Any]] = []
    offset = 0
    while True:
        page = message_store.query_messages(filters, limit=1000, offset=offset, order="asc")
        if not page:
            return out
        out.extend(page)
        if len(page) < 1000:
            return out
        offset += len(page)


def _cmd_judge(args) -> int:
    from neatmem import config  # after .env load in run_feedback
    from neatmem.feedback.runner import run_judge_batch
    from neatmem.storage.message.factory import create_message_store

    store = _open_store(args)
    message_store = create_message_store(
        config.MESSAGES_DB_PATH, extract_last_k=config.EXTRACT_LAST_K_MESSAGES,
        backend=config.MESSAGE_STORE_BACKEND,
    )
    try:
        summary = run_judge_batch(
            store,
            lambda filters: _load_scope_messages(message_store, filters),
            limit=args.limit,
        )
    finally:
        store.close()
        message_store.close()
    print(
        "judge batch: pending={pending} judged={judged} no_window={no_window} "
        "skipped_source={skipped_source} failed={failed}".format(**summary)
    )
    return 1 if summary["failed"] else 0


def _cmd_status(args) -> int:
    from neatmem import config

    store = _open_store(args)
    try:
        rows = store.list_feedback()
        evicted = store.list_evicted_ids()
        min_inj = config.MEMORY_FEEDBACK_EVICTION_MIN_INJECTIONS
        evictable = [
            r for r in rows
            if r["eviction_state"] == EVICTION_NONE
            and r["inject_count"] >= min_inj
            and r["used_count"] == 0
        ]
        total_injections = sum(r["inject_count"] for r in rows)
        total_used = sum(r["used_count"] for r in rows)
        print(f"tracked memories : {len(rows)}")
        if total_injections:
            print(f"injections judged: {total_injections} (used {total_used}, "
                  f"used rate {total_used / total_injections:.3f})")
        else:
            print("injections judged: 0")
        print(f"evicted          : {len(evicted)} "
              f"(derived {sum(1 for r in rows if r['eviction_state'] == 'derived')}, "
              f"manual {sum(1 for r in rows if r['eviction_state'] == 'manual')})")
        print(f"evictable (N={min_inj}, unused, not yet evicted): {len(evictable)}")
        if args.verbose:
            for r in rows[: args.top]:
                print(f"  {r['memory_id'][:36]:36s} inject={r['inject_count']:<4d} "
                      f"used={r['used_count']:<4d} state={r['eviction_state']}")
    finally:
        store.close()
    return 0


def _cmd_evict(args) -> int:
    store = _open_store(args)
    try:
        store.set_eviction(args.memory_id, EVICTION_MANUAL)
    finally:
        store.close()
    print(f"evicted (manual): {args.memory_id}")
    return 0


def _cmd_restore(args) -> int:
    store = _open_store(args)
    try:
        store.set_eviction(args.memory_id, EVICTION_NONE)
    finally:
        store.close()
    print(f"restored: {args.memory_id}")
    return 0


def run_feedback(argv: List[str]) -> None:
    from dotenv import load_dotenv

    # config reads env at import time — .env must load before any neatmem
    # config-dependent import below (same ordering as `neatmem serve`).
    load_dotenv(".env")  # existing env wins over .env

    parser = argparse.ArgumentParser(prog="neatmem feedback")
    parser.add_argument("--activity-db-path",
                        help="Activity DB path (env ACTIVITY_DB_PATH, default $NEATMEM_DIR/activity.db)")
    sub = parser.add_subparsers(dest="feedback_command", required=True)

    p_judge = sub.add_parser("judge", help="Run one offline judge batch over pending injections")
    p_judge.add_argument("--limit", type=int, default=None,
                         help="Max injections to judge this batch (default: all pending)")

    p_status = sub.add_parser("status", help="Projection summary (counters, evicted, evictable)")
    p_status.add_argument("--verbose", "-v", action="store_true", help="List per-memory rows")
    p_status.add_argument("--top", type=int, default=20, help="Rows to show with --verbose")

    p_evict = sub.add_parser("evict", help="Manually evict a memory (survives rebuilds)")
    p_evict.add_argument("memory_id")

    p_restore = sub.add_parser("restore", help="Clear a memory's eviction state")
    p_restore.add_argument("memory_id")

    args = parser.parse_args(argv)
    handlers = {
        "judge": _cmd_judge,
        "status": _cmd_status,
        "evict": _cmd_evict,
        "restore": _cmd_restore,
    }
    sys.exit(handlers[args.feedback_command](args))
