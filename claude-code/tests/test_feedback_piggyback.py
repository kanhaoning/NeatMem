"""Tests for the feedback piggyback contract (plan 20261005 §5.2).

The plugin reports actually-injected memory ids by attaching
``preceded_by_injection`` to the first message after the injection point;
the server strips it into a kind='injection' event. These tests pin the
client side: payload construction, event recording, and the anchoring logic
in build_episode / build_extraction_messages.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

HOST_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HOST_ROOT / "core"))
sys.path.insert(0, str(HOST_ROOT / "adapters" / "claude"))

import memory_core  # noqa: E402

import pytest  # noqa: E402


@pytest.fixture
def isolated_env(tmp_path, monkeypatch):
    """Same isolation as test_memory_core.isolated_env (kept local so this
    file stays self-contained)."""
    monkeypatch.setenv("NEATMEM_CODE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("NEATMEM_USER_ID", "test-user")
    for name in (
        "NEATMEM_API_KEY",
        "CLAUDE_PLUGIN_OPTION_API_KEY",
        "CLAUDE_PLUGIN_OPTION_NEATMEM_API_KEY",
        "CLAUDE_PLUGIN_DATA",
        "NEATMEM_PROJECT_ID",
        "NEATMEM_RESOLVED_USER_ID",
        "NEATMEM_API_URL",
        "CLAUDE_PLUGIN_OPTION_USER_ID",
    ):
        monkeypatch.delenv(name, raising=False)
    memory_core._resolve_repo_cached.cache_clear()
    return tmp_path


def repo() -> memory_core.RepoContext:
    return memory_core.RepoContext(
        cwd="/tmp/repo",
        root="/tmp/repo",
        identity="https://github.com/example/repo",
        app_id="code-example",
        branch="main",
        head_sha="abc123",
        project_id="code-example",
    )


def _episode_messages(events):
    _, structured = memory_core.build_episode(repo(), "s1", "packet", events)
    return memory_core.build_extraction_messages(structured)


def _prompt_event(text, injection=None):
    payload = {"text": text}
    if injection is not None:
        payload["injection"] = injection
    return {"kind": "user_prompt", "payload": payload}


def _stop_event(text, transcript_messages=None):
    payload = {"text": text}
    if transcript_messages:
        payload["transcript_messages"] = transcript_messages
    return {"kind": "assistant_stop", "payload": payload}


def test_injection_payload_empty_is_none():
    assert memory_core.injection_payload([], "user_prompt") is None
    assert memory_core.injection_payload([{"memory": "no id"}], "user_prompt") is None


def test_injection_payload_ids_and_source():
    payload = memory_core.injection_payload(
        [{"id": "m1"}, {"id": 42}], "midtask"
    )
    assert payload == {"memory_ids": ["m1", "42"], "source": "midtask"}


def test_record_user_prompt_carries_injection(isolated_env):
    store = memory_core.EvidenceStore()
    with patch.object(memory_core, "resolve_repo", return_value=repo()):
        memory_core.record_user_prompt(
            store,
            {"session_id": "s1", "cwd": "/tmp/repo", "prompt": "q"},
            injection={"memory_ids": ["m1"], "source": "user_prompt"},
        )
        memory_core.record_user_prompt(
            store, {"session_id": "s1", "cwd": "/tmp/repo", "prompt": "q2"}
        )
    rows = store.conn.execute(
        "SELECT payload_json FROM events WHERE kind = 'user_prompt' ORDER BY id"
    ).fetchall()
    store.close()
    first = json.loads(rows[0]["payload_json"])
    second = json.loads(rows[1]["payload_json"])
    assert first["injection"] == {"memory_ids": ["m1"], "source": "user_prompt"}
    assert "injection" not in second


def test_episode_anchors_user_prompt_marker_pending_path():
    events = [
        _prompt_event("Where is the parser?", {"memory_ids": ["m1", "m2"], "source": "user_prompt"}),
        _stop_event("In memory_core.py."),
    ]
    messages = _episode_messages(events)
    assert messages[0]["role"] == "user"
    assert messages[0]["preceded_by_injection"] == {
        "memory_ids": ["m1", "m2"],
        "source": "user_prompt",
    }
    assert "preceded_by_injection" not in messages[1]


def test_episode_anchors_marker_on_transcript_copy():
    # The transcript already contains the user prompt: the pending copy is
    # deduped away, so the marker must land on the transcript's user message.
    events = [
        _prompt_event("q1", {"memory_ids": ["m1"], "source": "user_prompt"}),
        _stop_event(
            "a1",
            transcript_messages=[
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1"},
            ],
        ),
    ]
    messages = _episode_messages(events)
    users = [m for m in messages if m["role"] == "user"]
    assert len(users) == 1  # dedup preserved
    assert users[0]["preceded_by_injection"]["memory_ids"] == ["m1"]


def test_episode_anchors_midtask_marker_on_next_assistant():
    events = [
        _prompt_event("q1"),
        _stop_event("first reply"),
        {"kind": "injection", "payload": {"memory_ids": ["m9"], "source": "midtask"}},
        _stop_event("second reply"),
    ]
    messages = _episode_messages(events)
    assert "preceded_by_injection" not in messages[1]  # first reply: before injection
    assert messages[2]["preceded_by_injection"] == {
        "memory_ids": ["m9"],
        "source": "midtask",
    }


def test_episode_midtask_marker_dropped_without_following_assistant():
    # Session ends right after the injection: no anchor message exists, the
    # marker must not misfire onto an unrelated message.
    events = [
        _prompt_event("q1"),
        _stop_event("only reply"),
        {"kind": "injection", "payload": {"memory_ids": ["m9"], "source": "midtask"}},
    ]
    messages = _episode_messages(events)
    assert all("preceded_by_injection" not in m for m in messages)


def test_episode_injection_event_produces_no_message():
    events = [
        _prompt_event("q1"),
        {"kind": "injection", "payload": {"memory_ids": ["m9"], "source": "midtask"}},
        _stop_event("a1"),
    ]
    messages = _episode_messages(events)
    assert [m["role"] for m in messages] == ["user", "assistant"]


def test_extraction_messages_passthrough_marker():
    structured = {
        "extraction_messages": [
            {
                "role": "user",
                "content": "q",
                "preceded_by_injection": {"memory_ids": ["m1"], "source": "user_prompt"},
            },
            {"role": "assistant", "content": "a"},
        ],
        "files_modified": [],
    }
    messages = memory_core.build_extraction_messages(structured)
    assert messages[0]["preceded_by_injection"] == {
        "memory_ids": ["m1"],
        "source": "user_prompt",
    }
    assert "preceded_by_injection" not in messages[1]


def test_prompt_memory_output_records_injected_ids(isolated_env, monkeypatch):
    monkeypatch.setenv("NEATMEM_API_KEY", "test-key")
    import hook_runner

    store = memory_core.EvidenceStore()
    results = [
        {"id": "memory-1", "memory": "fact one", "score": 0.9},
        {"id": "memory-2", "memory": "fact two", "score": 0.8},
    ]
    with (
        patch.object(memory_core, "resolve_repo", return_value=repo()),
        patch.object(
            memory_core, "_request_json", return_value=({"results": results}, 100, 200)
        ),
    ):
        output = hook_runner.prompt_memory_output(
            store, {"session_id": "s1", "cwd": "/tmp/repo", "prompt": "the question"}
        )
    assert output  # injected
    row = store.conn.execute(
        "SELECT payload_json FROM events WHERE kind = 'user_prompt'"
    ).fetchone()
    store.close()
    payload = json.loads(row["payload_json"])
    assert payload["injection"] == {
        "memory_ids": ["memory-1", "memory-2"],
        "source": "user_prompt",
    }


def test_prompt_memory_output_records_event_without_injection_on_empty(isolated_env, monkeypatch):
    monkeypatch.setenv("NEATMEM_API_KEY", "test-key")
    import hook_runner

    store = memory_core.EvidenceStore()
    with (
        patch.object(memory_core, "resolve_repo", return_value=repo()),
        patch.object(
            memory_core, "_request_json", return_value=({"results": []}, 100, 50)
        ),
    ):
        output = hook_runner.prompt_memory_output(
            store, {"session_id": "s1", "cwd": "/tmp/repo", "prompt": "the question"}
        )
    assert output == {}
    row = store.conn.execute(
        "SELECT payload_json FROM events WHERE kind = 'user_prompt'"
    ).fetchone()
    store.close()
    # The event is still recorded (extraction depends on it), with no marker.
    assert "injection" not in json.loads(row["payload_json"])


def test_prompt_memory_output_records_event_when_timing_off(isolated_env, monkeypatch):
    monkeypatch.setenv("NEATMEM_API_KEY", "test-key")
    monkeypatch.setenv("NEATMEM_CODE_INJECT_TIMING", "off")
    import hook_runner

    store = memory_core.EvidenceStore()
    with patch.object(memory_core, "resolve_repo", return_value=repo()):
        output = hook_runner.prompt_memory_output(
            store, {"session_id": "s1", "cwd": "/tmp/repo", "prompt": "the question"}
        )
    assert output == {}
    count = store.conn.execute(
        "SELECT COUNT(*) AS n FROM events WHERE kind = 'user_prompt'"
    ).fetchone()["n"]
    store.close()
    assert count == 1


def test_flush_posts_preceded_by_injection(isolated_env, monkeypatch):
    """End-to-end through flush_session: the marker reaches the POST body."""
    monkeypatch.setenv("NEATMEM_API_KEY", "test-key")
    store = memory_core.EvidenceStore()
    with patch.object(memory_core, "resolve_repo", return_value=repo()):
        memory_core.record_user_prompt(
            store,
            {"session_id": "s1", "cwd": "/tmp/repo", "prompt": "q1"},
            injection={"memory_ids": ["m1"], "source": "user_prompt"},
        )
        store.record_assistant_response(repo(), "s1", "a1")

    def fake_request(url, key, payload, timeout):
        if url.endswith("/v1/messages/flush/"):
            return {"batches": 1, "extracted_count": 2}, 100, 20
        return {}, 100, 20

    calls = []
    with (
        patch.object(memory_core, "resolve_repo", return_value=repo()),
        patch.object(memory_core, "_request_json_with_network_retry",
                     side_effect=lambda url, key, payload, timeout: (
                         calls.append((url, payload)),
                         fake_request(url, key, payload, timeout),
                     )[1]),
    ):
        result = memory_core.flush_session(
            store, {"session_id": "s1", "cwd": "/tmp/repo"}, "session-end"
        )
    store.close()
    assert result["status"] == "succeeded"
    add_bodies = [p for u, p in calls if u.endswith("/v1/messages/add/")]
    assert len(add_bodies) == 1
    messages = add_bodies[0]["messages"]
    assert messages[0]["preceded_by_injection"] == {
        "memory_ids": ["m1"],
        "source": "user_prompt",
    }
    assert "preceded_by_injection" not in messages[1]


def _write_transcript(path, session_id, entries):
    parent = None
    rows = []
    for index, entry in enumerate(entries, 1):
        row = {
            "uuid": entry.get("uuid", f"entry-{index}"),
            "parentUuid": entry.get("parentUuid", parent),
            "sessionId": session_id,
            "isSidechain": False,
            "type": entry["type"],
            "message": entry.get("message", {}),
        }
        rows.append(row)
        parent = row["uuid"]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_midtask_injection_records_event(isolated_env, monkeypatch):
    """midtask channel: an 'injection' event is recorded (v1: events-only)."""
    import transcript as transcript_mod

    monkeypatch.setenv("NEATMEM_CODE_MIDTASK_REMINDER_ENABLED", "1")
    monkeypatch.setenv("NEATMEM_CODE_MIDTASK_REMINDER_TOKENS", "10")
    monkeypatch.setenv("NEATMEM_API_KEY", "test-key")
    store = memory_core.EvidenceStore()
    transcript = isolated_env / "t.jsonl"
    _write_transcript(transcript, "s1", [{
        "type": "assistant",
        "message": {"role": "assistant",
                    "content": [{"type": "text", "text": "next I will deploy the release."}]},
    }])
    results = [{"id": "mem-1", "memory": "The release codeword is BLUEFIN.", "score": 0.9}]
    with (
        patch.object(memory_core, "resolve_repo", return_value=repo()),
        patch.object(
            memory_core, "_request_json_with_network_retry",
            return_value=({"results": results}, 100, 100),
        ),
    ):
        output = transcript_mod.midtask_reminder_after_tool(
            store,
            {"session_id": "s1", "cwd": "/tmp/repo",
             "transcript_path": str(transcript),
             "tool_name": "Read", "tool_input": {},
             "tool_response": {"success": True}},
        )
    assert output is not None  # injection happened
    row = store.conn.execute(
        "SELECT payload_json FROM events WHERE kind = 'injection'"
    ).fetchone()
    store.close()
    payload = json.loads(row["payload_json"])
    assert payload == {"memory_ids": ["mem-1"], "source": "midtask"}
