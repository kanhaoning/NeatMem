"""Shared hook orchestration for the NeatMem agent plugin."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

from memory_core import (
    EvidenceStore,
    _session_id,
    api_key,
    bounded,
    cache_plugin_api_key,
    checkpoint_session,
    clear_stale_api_key_cache,
    client_policy,
    data_dir,
    detached_process_kwargs,
    fetch_client_policy,
    format_context,
    forward_turn,
    inject_timing,
    injection_payload,
    per_turn_forward_enabled,
    plugin_enabled,
    recall_banner,
    midtask_reminder_load_state,
    midtask_reminder_reset,
    record_session_start,
    record_tool,
    record_user_prompt,
    redact,
    search_memories,
    search_timeout_seconds,
    write_last_recall,
)

STALE_RUNNING_SECONDS = 300
PENDING_EXPIRY_SECONDS = 7 * 24 * 60 * 60
PENDING_LAUNCH_LIMIT = 5
DEFAULT_IDLE_FLUSH_SECONDS = 300

_core_dir: Path = Path(__file__).resolve().parent


def read_hook_input() -> dict:
    try:
        value = json.load(sys.stdin)
        return value if isinstance(value, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def default_record_stop(store: EvidenceStore, hook_input: dict):
    """Record the assistant's response without transcript parsing."""
    session_id = _session_id(hook_input)
    repo = store.repo_for_session(session_id, hook_input.get("cwd"))
    message = redact(hook_input.get("last_assistant_message", "")).strip()
    if message:
        store.record_assistant_response(repo, session_id, message)
    return repo, session_id


def prompt_memory_output(store: EvidenceStore, hook_input: dict) -> dict:
    """Search and inject memories as governed by the inject_timing policy.

    off: never search. first: only before the session's first prompt.
    every: before every prompt. Short prompts below the server's
    min_query_chars are skipped in every mode unless the server's
    query_rewrite policy is on (the server then decides per prompt).

    The user_prompt event is always recorded (extraction depends on it); when
    memories were actually injected, the event carries the injected id set so
    the upload path can anchor preceded_by_injection (plan 20261005 §5.2).
    """
    session_id = _session_id(hook_input)
    repo = store.repo_for_session(session_id, hook_input.get("cwd"))
    prompt = redact(hook_input.get("prompt", "")).strip()
    is_first_prompt = not store.has_event(repo.identity, session_id, "user_prompt")
    timing = inject_timing(store)
    injection = None
    output: dict = {}
    if timing != "off" and (timing != "first" or is_first_prompt):
        policy = client_policy(store)
        minimum_query_chars = int(policy["min_query_chars"])
        # With server-side query rewrite on, short prompts are still sent: the
        # server decides (short + no context → empty; short + context → rewrite).
        if not policy.get("query_rewrite") and len(prompt.strip()) < max(
            minimum_query_chars, 1
        ):
            write_last_recall(
                operation="prompt-search",
                session_id=session_id,
                query=prompt,
                skipped=f"prompt shorter than min_query_chars ({minimum_query_chars})",
            )
        else:
            result = search_memories(
                store, repo, session_id, bounded(prompt, 6000),
                top_k=5, operation="prompt-search", timeout=search_timeout_seconds(),
            )
            if not result.memories:
                write_last_recall(
                    operation="prompt-search",
                    session_id=session_id,
                    query=prompt,
                    memories=[],
                    skipped=None if result.succeeded else "search failed (see operations log)",
                )
            else:
                write_last_recall(
                    operation="prompt-search",
                    session_id=session_id,
                    query=prompt,
                    memories=result.memories,
                )
                injection = injection_payload(result.memories, "user_prompt")
                context = format_context(
                    result.memories,
                    "NeatMem found these relevant memories from earlier work in this repository:",
                )
                output = {
                    "hookSpecificOutput": {
                        "hookEventName": "UserPromptSubmit",
                        "additionalContext": context,
                    },
                }
                banner = recall_banner(result.memories)
                if banner:
                    output["systemMessage"] = banner
    record_user_prompt(store, hook_input, injection=injection)
    return output


def _launch_handoff(handoff_path: Path) -> bool:
    running_path = handoff_path.with_suffix(".running")
    try:
        handoff_path.replace(running_path)
    except OSError:
        return False
    worker = _core_dir / "flush_worker.py"
    log_path = data_dir() / "flush-worker.log"
    log_handle = open(log_path, "a", encoding="utf-8")
    child_env = os.environ.copy()
    child_env["NEATMEM_CODE_DATA_DIR"] = str(data_dir())
    try:
        subprocess.Popen(
            [sys.executable, str(worker), str(running_path)],
            stdin=subprocess.DEVNULL,
            stdout=log_handle, stderr=log_handle,
            close_fds=True,
            env=child_env,
            **detached_process_kwargs(),
        )
    finally:
        log_handle.close()
    return True


def recover_pending_handoffs() -> int:
    pending_dir = data_dir() / "pending"
    pending_dir.mkdir(parents=True, exist_ok=True)
    now = time.time()
    for running in pending_dir.glob("*.running"):
        try:
            if now - running.stat().st_mtime > STALE_RUNNING_SECONDS:
                running.replace(running.with_suffix(".json"))
        except OSError:
            continue
    recoverable = []
    for handoff in pending_dir.glob("*.json"):
        try:
            age = now - handoff.stat().st_mtime
        except OSError:
            continue
        if age > PENDING_EXPIRY_SECONDS:
            handoff.unlink(missing_ok=True)
            continue
        recoverable.append((age, handoff))
    recoverable.sort(key=lambda item: item[0], reverse=True)
    launched = 0
    for _, handoff in recoverable[:PENDING_LAUNCH_LIMIT]:
        launched += int(_launch_handoff(handoff))
    return launched


def refresh_pending_handoffs() -> None:
    pending_dir = data_dir() / "pending"
    if not pending_dir.is_dir():
        return
    for pattern in ("*.json", "*.running"):
        for handoff in pending_dir.glob(pattern):
            try:
                os.utime(handoff)
            except OSError:
                continue


def hand_off_flush(
    hook_input: dict, reason: str, *, wait_for_inflight: bool = False,
) -> None:
    pending_dir = data_dir() / "pending"
    pending_dir.mkdir(parents=True, exist_ok=True)
    material = (
        f"{hook_input.get('cwd', '')}\0{hook_input.get('session_id', '')}\0{reason}"
    )
    digest = hashlib.sha256(material.encode()).hexdigest()[:24]
    handoff_path = pending_dir / f"{digest}-{uuid.uuid4().hex[:8]}.json"
    temporary_path = handoff_path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps({
            "hook_input": hook_input,
            "reason": reason,
            "wait_for_inflight": wait_for_inflight,
        }),
        encoding="utf-8",
    )
    temporary_path.replace(handoff_path)
    _launch_handoff(handoff_path)


def schedule_periodic_checkpoint(
    store: EvidenceStore, hook_input: dict, repo, session_id: str,
) -> bool:
    if not store.checkpoint_due(repo.identity, session_id):
        return False
    if store.prepare_flush(repo, session_id, "periodic") is None:
        return False
    hand_off_flush(hook_input, "periodic")
    return True


def _idle_flush_seconds() -> int:
    try:
        return max(
            int(os.environ.get("NEATMEM_CODE_IDLE_FLUSH_SECONDS", str(DEFAULT_IDLE_FLUSH_SECONDS))),
            0,
        )
    except ValueError:
        return DEFAULT_IDLE_FLUSH_SECONDS


def schedule_idle_flush(
    store: EvidenceStore, hook_input: dict, repo, session_id: str,
) -> bool:
    delay = _idle_flush_seconds()
    if delay <= 0:
        return False
    if store.has_inflight_flush(repo.identity, session_id):
        return False
    if not store.has_unflushed_events(repo.identity, session_id):
        return False
    pending_dir = data_dir() / "pending"
    pending_dir.mkdir(parents=True, exist_ok=True)
    material = f"idle\0{hook_input.get('cwd', '')}\0{hook_input.get('session_id', '')}"
    digest = hashlib.sha256(material.encode()).hexdigest()[:24]
    for old in pending_dir.glob(f"idle-{digest}*"):
        old.unlink(missing_ok=True)
    handoff_path = pending_dir / f"idle-{digest}-{uuid.uuid4().hex[:8]}.json"
    temporary_path = handoff_path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps({
            "hook_input": hook_input,
            "reason": "idle",
            "delay_seconds": delay,
        }),
        encoding="utf-8",
    )
    temporary_path.replace(handoff_path)
    _launch_handoff(handoff_path)
    return True


def log_failure(exc: Exception) -> None:
    try:
        log_path = data_dir() / "plugin-errors.log"
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{time.time():.3f} {type(exc).__name__}: {exc}\n")
    except OSError:
        pass


def run(
    *,
    record_stop_fn=None,
    post_tool_after=None,
    extra_actions: dict | None = None,
    data_dir_env: str = "NEATMEM_CODE_DATA_DIR",
    automatic_flush_reasons: set | None = None,
) -> int:
    if record_stop_fn is None:
        record_stop_fn = default_record_stop
    if automatic_flush_reasons is None:
        automatic_flush_reasons = {"session-end"}

    base_actions = ["session-start", "user-prompt", "post-tool", "stop", "flush"]
    all_actions = base_actions + list((extra_actions or {}).keys())

    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=all_actions)
    parser.add_argument("--reason", default="manual")
    parser.add_argument("--plugin-data-dir", default="")
    args = parser.parse_args()

    if args.plugin_data_dir:
        os.environ[data_dir_env] = args.plugin_data_dir

    cache_plugin_api_key()
    if args.action == "session-start":
        clear_stale_api_key_cache()

    if not plugin_enabled():
        return 0

    hook_input = read_hook_input()
    store = EvidenceStore()
    try:
        if store.is_paused():
            if args.action == "session-start":
                refresh_pending_handoffs()
            return 0

        if args.action == "session-start":
            recover_pending_handoffs()
            record_session_start(store, hook_input)
            # Compaction/clear drops injected memories from the context
            # window, so the watermark state is meaningless afterwards.
            if hook_input.get("source") in {"compact", "clear"}:
                midtask_reminder_session = _session_id(hook_input)
                if midtask_reminder_load_state(store, midtask_reminder_session):
                    midtask_reminder_reset(store, midtask_reminder_session)
            policy = fetch_client_policy(store)
            if policy["source"] != "server":
                detail = policy["error"] or "no client_policy in response"
                print(json.dumps({
                    "hookSpecificOutput": {
                        "hookEventName": "SessionStart",
                        "additionalContext": (
                            "NeatMem: could not load the server client policy "
                            f"({detail}); using built-in defaults "
                            f"(inject_timing={policy['inject_timing']}, "
                            f"min_query_chars={policy['min_query_chars']})."
                        ),
                    },
                }))
        elif args.action == "user-prompt":
            output = prompt_memory_output(store, hook_input)
            if output:
                print(json.dumps(output))
        elif args.action == "post-tool":
            record_tool(store, hook_input)
            if post_tool_after is not None:
                output = post_tool_after(store, hook_input)
                if output:
                    print(json.dumps(output))
        elif args.action == "stop":
            repo, session_id = record_stop_fn(store, hook_input)
            if per_turn_forward_enabled(store):
                forward_turn(store, repo, session_id)
            if not schedule_periodic_checkpoint(store, hook_input, repo, session_id):
                schedule_idle_flush(store, hook_input, repo, session_id)
        elif args.action == "flush":
            if args.reason == "session-end":
                record_stop_fn(store, hook_input)
            if os.environ.get("NEATMEM_CODE_SYNC_FLUSH") == "1":
                print(json.dumps(checkpoint_session(store, hook_input, args.reason)))
            else:
                session_id = str(hook_input.get("session_id") or "unknown-session")
                repo = store.repo_for_session(session_id, hook_input.get("cwd"))
                already_running = store.has_inflight_flush(repo.identity, session_id)
                if already_running and args.reason == "session-end":
                    hand_off_flush(hook_input, args.reason, wait_for_inflight=True)
                elif not already_running and (
                    store.prepare_flush(repo, session_id, args.reason) is not None
                    or (
                        # Per-turn mode: local events are already uploaded, but
                        # the server queue may hold an under-batch tail that the
                        # worker must force-flush at the session boundary.
                        args.reason in automatic_flush_reasons
                        and per_turn_forward_enabled(store)
                    )
                ):
                    hand_off_flush(hook_input, args.reason)
        elif extra_actions and args.action in extra_actions:
            result = extra_actions[args.action](store, hook_input)
            if result:
                print(json.dumps(result))
    finally:
        store.close()
    return 0


def entry_point(
    *,
    record_stop_fn=None,
    post_tool_after=None,
    extra_actions: dict | None = None,
    data_dir_env: str = "NEATMEM_CODE_DATA_DIR",
    automatic_flush_reasons: set | None = None,
) -> None:
    try:
        raise SystemExit(run(
            record_stop_fn=record_stop_fn,
            post_tool_after=post_tool_after,
            extra_actions=extra_actions,
            data_dir_env=data_dir_env,
            automatic_flush_reasons=automatic_flush_reasons,
        ))
    except Exception as exc:
        log_failure(exc)
        raise SystemExit(0)


if __name__ == "__main__":
    entry_point()
