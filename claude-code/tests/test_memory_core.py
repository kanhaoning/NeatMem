from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

HOST_ROOT = Path(__file__).resolve().parents[1]
PLUGIN_ROOT = HOST_ROOT
CORE = HOST_ROOT / "core"
ADAPTER = HOST_ROOT / "adapters" / "claude" / "hook.py"
sys.path.insert(0, str(CORE))
sys.path.insert(0, str(ADAPTER.parent))

import mcp_server  # noqa: E402
import memory_cli  # noqa: E402
import memory_core  # noqa: E402
import transcript as transcript_mod  # noqa: E402


@pytest.fixture
def isolated_env(tmp_path, monkeypatch):
    monkeypatch.setenv("NEATMEM_CODE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("NEATMEM_USER_ID", "test-user")
    monkeypatch.delenv("NEATMEM_API_KEY", raising=False)
    monkeypatch.delenv("CLAUDE_PLUGIN_OPTION_API_KEY", raising=False)
    monkeypatch.delenv("CLAUDE_PLUGIN_OPTION_NEATMEM_API_KEY", raising=False)
    monkeypatch.delenv("CLAUDE_PLUGIN_DATA", raising=False)
    # The 0.2.x plugin exports these into every hooked shell; without this the
    # suite fails for anyone running it inside a session with that plugin active.
    monkeypatch.delenv("NEATMEM_PROJECT_ID", raising=False)
    monkeypatch.delenv("NEATMEM_RESOLVED_USER_ID", raising=False)
    monkeypatch.delenv("NEATMEM_API_URL", raising=False)
    monkeypatch.delenv("CLAUDE_PLUGIN_OPTION_USER_ID", raising=False)
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


def _flush_server(extracted_count: int = 2):
    """Fake the NeatMem two-step flush endpoints as a _request_json stand-in."""

    def request(url, key, payload, timeout):
        if url.endswith("/v1/messages/flush/"):
            return {"batches": 1, "extracted_count": extracted_count}, 100, 20
        return {}, 100, 20

    return request


def _add_bodies(request) -> list[dict]:
    """Bodies posted to /v1/messages/add/ out of a patched _request_json."""
    return [
        call.args[2]
        for call in request.call_args_list
        if call.args[0].endswith("/v1/messages/add/")
    ]


def test_portable_plugin_uses_standard_data_directory(tmp_path, monkeypatch):
    monkeypatch.delenv("NEATMEM_CODE_DATA_DIR", raising=False)
    monkeypatch.delenv("CLAUDE_PLUGIN_DATA", raising=False)
    monkeypatch.setenv("PLUGIN_DATA", str(tmp_path))

    assert memory_core.data_dir() == tmp_path


def _write_transcript(path: Path, session_id: str, entries: list[dict]) -> None:
    parent = None
    rows = []
    for index, entry in enumerate(entries, 1):
        row = {
            "uuid": entry.get("uuid", f"entry-{index}"),
            "parentUuid": entry.get("parentUuid", parent),
            "sessionId": session_id,
            "isSidechain": entry.get("isSidechain", False),
            "type": entry["type"],
            "message": entry.get("message", {}),
        }
        if "origin" in entry:
            row["origin"] = entry["origin"]
        rows.append(row)
        parent = row["uuid"]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_transcript_extraction_keeps_meaningful_messages_and_excludes_raw_tools(
    tmp_path,
):
    transcript = tmp_path / "session.jsonl"
    _write_transcript(
        transcript,
        "s1",
        [
            {
                "type": "user",
                "origin": {"kind": "human"},
                "message": {"role": "user", "content": "Fix repository scoping."},
            },
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "thinking", "thinking": "hidden"}],
                },
            },
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "I found the scope resolver."}
                    ],
                },
            },
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "agent-1",
                            "name": "Agent",
                            "input": {
                                "description": "Trace repository scope",
                                "subagent_type": "Explore",
                                "prompt": "Find where the repository scope is selected.",
                            },
                        }
                    ],
                },
            },
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "agent-1",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Scope comes from resolve_repo in memory_core.py.",
                                }
                            ],
                        }
                    ],
                },
            },
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "question-1",
                            "name": "AskUserQuestion",
                            "input": {"questions": [{"question": "Use repository scope?"}]},
                        }
                    ],
                },
            },
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "question-1",
                            "content": '"Use repository scope?"="Yes"',
                        }
                    ],
                },
            },
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "plan-1",
                            "name": "ExitPlanMode",
                            "input": {
                                "plan": "Use the Git repository as the memory scope."
                            },
                        }
                    ],
                },
            },
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "plan-1",
                            "content": "Plan approved.",
                        }
                    ],
                },
            },
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "read-1",
                            "name": "Read",
                            "input": {"file_path": "/tmp/repo/secret.py"},
                        }
                    ],
                },
            },
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "read-1",
                            "content": "raw file contents must not be sent",
                        }
                    ],
                },
            },
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Repository scope is now stable."}
                    ],
                },
            },
        ],
    )

    messages, leaf, _ = transcript_mod.transcript_extraction_messages(
        str(transcript), "s1"
    )
    serialized = json.dumps(messages)

    assert leaf == "entry-12"
    assert [message["role"] for message in messages] == [
        "user",
        "assistant",
        "assistant",
        "assistant",
        "user",
        "assistant",
        "assistant",
    ]
    assert "Fix repository scoping" in serialized
    assert "I found the scope resolver" in serialized
    assert "Find where the repository scope is selected" in serialized
    assert "Scope comes from resolve_repo" in serialized
    assert "Use repository scope" in serialized
    assert "Approved implementation plan" in serialized
    assert "Repository scope is now stable" in serialized
    assert "Main Claude response" not in serialized
    assert "hidden" not in serialized
    assert "raw file contents" not in serialized
    assert "secret.py" not in serialized


def test_transcript_extraction_pairs_background_agent_notification(tmp_path):
    transcript = tmp_path / "session.jsonl"
    notification = """<task-notification>
<task-id>worker-1</task-id>
<tool-use-id>agent-1</tool-use-id>
<status>completed</status>
<summary>Explore agent finished</summary>
<result>The parser selects repository scope from the Git root.</result>
</task-notification>"""
    _write_transcript(
        transcript,
        "s1",
        [
            {
                "type": "user",
                "origin": {"kind": "human"},
                "message": {"role": "user", "content": "Inspect repository scope."},
            },
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "agent-1",
                            "name": "Agent",
                            "input": {
                                "description": "Inspect scope",
                                "subagent_type": "Explore",
                                "prompt": "Trace repository scope selection.",
                            },
                        }
                    ],
                },
            },
            {
                "uuid": "launch-result",
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "agent-1",
                            "content": "Async agent launched successfully. agentId: worker-1",
                        }
                    ],
                },
            },
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Waiting for the agent."}],
                },
            },
            {
                "type": "user",
                "origin": {"kind": "task-notification"},
                "message": {"role": "user", "content": notification},
            },
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "The scope is understood."}],
                },
            },
        ],
    )

    messages, _, _ = transcript_mod.transcript_extraction_messages(
        str(transcript), "s1", previous_leaf_uuid="launch-result"
    )
    serialized = json.dumps(messages)

    assert "Trace repository scope selection" in serialized
    assert "selects repository scope from the Git root" in serialized
    assert "Async agent launched" not in serialized
    assert "task-notification" not in serialized


def test_transcript_extraction_omits_claude_ui_messages_but_keeps_attachments(
    tmp_path,
):
    transcript = tmp_path / "session.jsonl"
    _write_transcript(
        transcript,
        "s1",
        [
            {
                "type": "user",
                "origin": {"kind": "human"},
                "message": {
                    "role": "user",
                    "content": "<local-command-caveat>Do not answer this output.</local-command-caveat>",
                },
            },
            {
                "type": "user",
                "origin": {"kind": "human"},
                "message": {
                    "role": "user",
                    "content": "<command-name>/status</command-name>",
                },
            },
            {
                "type": "user",
                "origin": {"kind": "human"},
                "message": {
                    "role": "user",
                    "content": "<!-- attach -->The attached issue says repository scope is wrong.",
                },
            },
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "I will inspect the scope."}],
                },
            },
        ],
    )

    messages, _, _ = transcript_mod.transcript_extraction_messages(
        str(transcript), "s1", prompt_hint="The attached issue says repository scope is wrong."
    )
    serialized = json.dumps(messages)

    assert "The attached issue says repository scope is wrong" in serialized
    assert "I will inspect the scope" in serialized
    assert "local-command" not in serialized
    assert "command-name" not in serialized
    assert "<!-- attach -->" not in serialized


def test_transcript_extraction_uses_active_branch_and_omits_rejected_plan(tmp_path):
    transcript = tmp_path / "session.jsonl"
    rows = [
        {
            "uuid": "prompt",
            "parentUuid": None,
            "sessionId": "s1",
            "isSidechain": False,
            "type": "user",
            "origin": {"kind": "human"},
            "message": {"role": "user", "content": "Design repository memory."},
        },
        {
            "uuid": "rejected-plan",
            "parentUuid": "prompt",
            "sessionId": "s1",
            "isSidechain": False,
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "plan-1",
                        "name": "ExitPlanMode",
                        "input": {"plan": "Store every raw tool result."},
                    }
                ],
            },
        },
        {
            "uuid": "rejection",
            "parentUuid": "rejected-plan",
            "sessionId": "s1",
            "isSidechain": False,
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "plan-1",
                        "is_error": True,
                        "content": "The user rejected this plan.",
                    }
                ],
            },
        },
        {
            "uuid": "abandoned",
            "parentUuid": "rejection",
            "sessionId": "s1",
            "isSidechain": False,
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Abandoned branch conclusion."}],
            },
        },
        {
            "uuid": "correction",
            "parentUuid": "rejection",
            "sessionId": "s1",
            "isSidechain": False,
            "type": "user",
            "origin": {"kind": "human"},
            "message": {
                "role": "user",
                "content": "Keep conclusions, not raw tool results.",
            },
        },
        {
            "uuid": "final",
            "parentUuid": "correction",
            "sessionId": "s1",
            "isSidechain": False,
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Understood."}],
            },
        },
    ]
    transcript.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    messages, leaf, _ = transcript_mod.transcript_extraction_messages(
        str(transcript), "s1"
    )
    serialized = json.dumps(messages)

    assert leaf == "final"
    assert "Keep conclusions" in serialized
    assert "Understood" in serialized
    assert "Store every raw tool result" not in serialized
    assert "Abandoned branch conclusion" not in serialized


def test_record_stop_processes_only_new_transcript_entries(
    isolated_env, tmp_path, monkeypatch
):
    transcript = tmp_path / "session.jsonl"
    first_entries = [
        {
            "type": "user",
            "origin": {"kind": "human"},
            "message": {"role": "user", "content": "Inspect the parser."},
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "The parser reads JSONL."}],
            },
        },
    ]
    _write_transcript(transcript, "s1", first_entries)
    store = memory_core.EvidenceStore()

    with patch.object(memory_core, "resolve_repo", return_value=repo()):
        memory_core.record_user_prompt(
            store,
            {"session_id": "s1", "cwd": "/tmp/repo", "prompt": "Inspect the parser."},
        )
        transcript_mod.record_stop(
            store,
            {
                "session_id": "s1",
                "cwd": "/tmp/repo",
                "transcript_path": str(transcript),
                "last_assistant_message": "The parser reads JSONL.",
            },
        )

    second_entries = first_entries + [
        {
            "type": "user",
            "origin": {"kind": "human"},
            "message": {"role": "user", "content": "Where is that implemented?"},
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "It is implemented in memory_core.py."}
                ],
            },
        },
    ]
    _write_transcript(transcript, "s1", second_entries)

    with patch.object(memory_core, "resolve_repo", return_value=repo()):
        memory_core.record_user_prompt(
            store,
            {
                "session_id": "s1",
                "cwd": "/tmp/repo",
                "prompt": "Where is that implemented?",
            },
        )
        transcript_mod.record_stop(
            store,
            {
                "session_id": "s1",
                "cwd": "/tmp/repo",
                "transcript_path": str(transcript),
                "last_assistant_message": "It is implemented in memory_core.py.",
            },
        )

    events = [
        {
            "kind": row["kind"],
            "payload": json.loads(row["payload_json"]),
        }
        for row in store.conn.execute("SELECT kind, payload_json FROM events ORDER BY id")
    ]
    _, structured = memory_core.build_episode(repo(), "s1", "packet", events)
    messages = memory_core.build_extraction_messages(structured)
    store.close()

    assert [message["content"] for message in messages] == [
        "Inspect the parser.",
        "Main Claude response:\nThe parser reads JSONL.",
        "Where is that implemented?",
        "Main Claude response:\nIt is implemented in memory_core.py.",
    ]


def test_extraction_messages_preserve_long_content():
    long_response = "repository detail " * 4000
    structured = {
        "extraction_messages": [
            {"role": "user", "content": "Explain the repository."},
            {"role": "assistant", "content": long_response},
        ],
        "files_modified": [],
    }

    messages = memory_core.build_extraction_messages(structured)

    assert messages == [
        {"role": "user", "content": "Explain the repository."},
        {"role": "assistant", "content": long_response.strip()},
    ]


@pytest.mark.parametrize("capture", ["hook", "transcript"])
def test_long_conversation_reaches_extraction_without_truncation(isolated_env, monkeypatch, capture):
    import hook_runner

    monkeypatch.setenv("NEATMEM_API_KEY", "test-key")
    monkeypatch.setattr(memory_core, "resolve_repo", lambda cwd: repo())
    prompt = "Repository question. " * 2000 + "Final user requirement. api_key=hidden-user-secret"
    answer = "Repository answer. " * 4000 + "Final implementation detail. password=hidden-agent-secret"
    payload = {"session_id": "s1", "cwd": "/tmp/repo", "prompt": prompt, "last_assistant_message": answer}
    store = memory_core.EvidenceStore()
    memory_core.record_user_prompt(store, payload)
    if capture == "transcript":
        transcript = isolated_env / "long-session.jsonl"
        _write_transcript(transcript, "s1", [
            {"type": "user", "message": {"role": "user", "content": prompt}},
            {"type": "assistant", "message": {"role": "assistant", "content": answer}},
        ])
        transcript_mod.record_stop(store, {**payload, "transcript_path": str(transcript)})
    else:
        hook_runner.default_record_stop(store, payload)

    with patch.object(
        memory_core, "_request_json", side_effect=_flush_server(1)
    ) as request:
        assert memory_core.flush_session(store, payload, "session-end")["status"] == "succeeded"

    assert request.call_args_list[-1].args[0].endswith("/v1/messages/flush/")
    batches = [
        call.args[2]["messages"]
        for call in request.call_args_list
        if call.args[0].endswith("/v1/messages/add/")
    ]
    assert len(batches) > 1
    assert all(memory_core._message_tokens(batch) <= memory_core.MAX_EXTRACTION_INPUT_TOKENS for batch in batches)
    for role, text in (("user", prompt), ("assistant", answer)):
        actual = "".join(message["content"] for batch in batches for message in batch if message["role"] == role)
        prefix = "Main Claude response:\n" if capture == "transcript" and role == "assistant" else ""
        assert actual == prefix + memory_core.redact(text)
    store.close()


def test_upgrade_removes_legacy_snapshot_tables(isolated_env):
    database = Path(os.environ["NEATMEM_CODE_DATA_DIR"]) / "evidence.sqlite3"
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE sidekick_packets (packet_id TEXT);
        CREATE TABLE sidekick_state (repo_id TEXT);
        CREATE TABLE sidekick_snapshots (repo_id TEXT);
        CREATE TABLE sidekick_calls (id INTEGER);
        """
    )
    connection.close()

    store = memory_core.EvidenceStore()
    table_names = {
        row[0]
        for row in store.conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    store.close()

    assert (
        not {
            "sidekick_packets",
            "sidekick_state",
            "sidekick_snapshots",
            "sidekick_calls",
        }
        & table_names
    )


def test_upgrade_adds_retrieval_observability_columns(isolated_env):
    database = Path(os.environ["NEATMEM_CODE_DATA_DIR"]) / "evidence.sqlite3"
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database)
    connection.execute(
        """CREATE TABLE retrievals (
               session_id TEXT NOT NULL,
               repo_id TEXT NOT NULL,
               memory_id TEXT NOT NULL,
               injected_at TEXT NOT NULL,
               PRIMARY KEY(session_id, repo_id, memory_id)
           )"""
    )
    connection.close()

    store = memory_core.EvidenceStore()
    columns = {
        row["name"] for row in store.conn.execute("PRAGMA table_info(retrievals)")
    }
    store.close()

    assert {"rank", "score", "memory_text", "context_chars"} <= columns


def test_hook_config_key_is_cached_for_checkpoint_and_search(isolated_env, monkeypatch):
    monkeypatch.setenv("CLAUDE_PLUGIN_OPTION_API_KEY", "m0-plugin-key")

    assert memory_core.cache_plugin_api_key() is True
    key_path = Path(os.environ["NEATMEM_CODE_DATA_DIR"]) / "api-key"
    assert key_path.stat().st_mode & 0o777 == 0o600

    monkeypatch.delenv("CLAUDE_PLUGIN_OPTION_API_KEY")
    assert memory_core.api_key() == "m0-plugin-key"


def test_doctor_verifies_server_connection(isolated_env, monkeypatch):
    with (
        patch.object(memory_core, "resolve_repo", return_value=repo()),
        patch.object(memory_core, "_get_json", return_value=({}, 10)) as request,
    ):
        result = memory_core.doctor("/tmp/repo")

    assert result["ok"] is True
    assert set(result["checks"]) == {
        "python",
        "data_directory",
        "api_url",
        "repository",
        "user_id",
        "server_connection",
    }
    assert result["checks"]["server_connection"]["ok"] is True
    assert "connected" in result["checks"]["server_connection"]["detail"]
    assert request.call_args.args[0].endswith("/v1/ping/")


def test_doctor_reports_an_unreachable_server(isolated_env, monkeypatch):
    with (
        patch.object(memory_core, "resolve_repo", return_value=repo()),
        patch.object(
            memory_core,
            "_get_json",
            side_effect=RuntimeError("HTTP Error 401: Unauthorized"),
        ),
    ):
        result = memory_core.doctor("/tmp/repo")

    assert result["ok"] is False
    assert result["checks"]["server_connection"] == {
        "ok": False,
        "detail": "HTTP Error 401: Unauthorized",
    }


def test_tool_capture_is_bounded_and_does_not_store_edited_code():
    payload = memory_core.tool_payload(
        {
            "tool_name": "Edit",
            "tool_input": {
                "file_path": "/tmp/repo/app.py",
                "old_string": "old secret code",
                "new_string": "new secret code",
            },
            "tool_response": {"success": True},
            "duration_ms": 12,
        }
    )

    assert payload["path"] == "/tmp/repo/app.py"
    assert payload["mutation_chars"] == len("new secret code")
    assert "old secret code" not in json.dumps(payload)
    assert "new secret code" not in json.dumps(payload)


def test_tool_capture_identifies_main_and_sidekick_roles():
    main = memory_core.tool_payload(
        {"tool_name": "Read", "tool_input": {"file_path": "/tmp/repo/app.py"}}
    )
    sidekick = memory_core.tool_payload(
        {
            "tool_name": "Read",
            "tool_input": {"file_path": "/tmp/repo/app.py"},
            "agent_id": "agent-123",
            "agent_type": "neatmem:sidekick",
        }
    )

    assert main["agent_role"] == "main"
    assert sidekick["agent_role"] == "subagent"
    assert sidekick["agent_id"] == "agent-123"
    assert sidekick["agent_type"] == "neatmem:sidekick"


def test_episode_uses_repository_relative_paths():
    r = repo()
    episode, structured = memory_core.build_episode(
        r,
        "s1",
        "packet",
        [
            {
                "kind": "tool_result",
                "payload": {
                    "tool": "Read",
                    "path": "/tmp/repo/src/parser.py",
                    "failed": False,
                },
            },
            {
                "kind": "tool_result",
                "payload": {
                    "tool": "Edit",
                    "path": "/tmp/repo/tests/test_parser.py",
                    "failed": False,
                },
            },
            {
                "kind": "tool_result",
                "payload": {
                    "tool": "Write",
                    "path": "/Users/example/.claude/projects/repo/memory/MEMORY.md",
                    "failed": False,
                },
            },
        ],
    )

    assert structured["files_read"] == ["src/parser.py"]
    assert structured["files_modified"] == ["tests/test_parser.py"]
    assert "/tmp/repo" not in episode
    assert ".claude/projects" not in episode


def test_semantic_evidence_keeps_changed_paths_but_omits_test_output():
    evidence = memory_core.build_semantic_evidence(
        {
            "files_modified": ["src/parser.py"],
            "commands": [
                {
                    "kind": "test",
                    "status": "succeeded",
                    "command": "pytest -q",
                    "result": "136 passed",
                }
            ],
        }
    )

    assert "src/parser.py" in evidence
    assert "pytest" not in evidence
    assert "136 passed" not in evidence


def test_combined_context_has_one_budget_and_removes_exact_duplicate_lines(
    isolated_env, monkeypatch
):
    monkeypatch.setenv("NEATMEM_CODE_MAX_CONTEXT_CHARS", "1000")
    combined = memory_core.combine_context(
        "Memory context:\n- shared fact\n" + "a" * 900,
        "Sidekick context:\n- shared fact\n" + "b" * 900,
    )

    assert len(combined) <= 1035
    assert combined.count("- shared fact") == 1
    assert "truncated" in combined


def test_secrets_are_redacted_from_commands_and_results():
    payload = memory_core.tool_payload(
        {
            "tool_name": "Bash",
            "tool_input": {"command": "export API_KEY=sk-abcdefghijklmnopqrstuvwxyz"},
            "tool_response": {
                "stdout": "Authorization: Bearer psk-abcdefghijklmnopqrstuvwxyz"
            },
        }
    )

    serialized = json.dumps(payload)
    assert "abcdefghijklmnopqrstuvwxyz" not in serialized
    assert "[REDACTED]" in serialized


def test_common_tokens_and_private_keys_are_redacted():
    value = memory_core.redact(
        "url=https://example.com?access_token=secret-value&next=1 "
        "github_pat_abcdefghijklmnopqrstuvwxyz "
        "-----BEGIN PRIVATE KEY-----\nprivate-material\n-----END PRIVATE KEY-----"
    )

    assert "secret-value" not in value
    assert "github_pat_" not in value
    assert "private-material" not in value


def test_json_shaped_secrets_are_redacted():
    value = memory_core.redact('{"apiKey": "sk-secret-12345", "name": "test"}')
    assert "sk-secret-12345" not in value
    assert "[REDACTED]" in value
    assert '"name": "test"' in value

    value = memory_core.redact('{"password": "hunter2", "count": 1}')
    assert "hunter2" not in value

    value = memory_core.redact('{"secret_access_key": "ABCDEFGHIJ1234567890", "region": "us-east-1"}')
    assert "ABCDEFGHIJ1234567890" not in value
    assert '"region": "us-east-1"' in value


def test_remote_identity_removes_embedded_credentials():
    normalized = memory_core._normalize_remote(
        "https://secret-token@github.com/example/repo.git"
    )
    assert normalized == "https://github.com/example/repo"
    assert "secret-token" not in normalized


def test_different_git_hosts_produce_different_project_ids():
    github = memory_core._project_id("/tmp/repo", "https://github.com/acme/api", "acme-api")
    gitlab = memory_core._project_id("/tmp/repo", "https://gitlab.internal/acme/api", "acme-api")

    assert github != gitlab
    assert github.startswith("acme-api-")
    assert gitlab.startswith("acme-api-")


def test_non_git_session_keeps_starting_project_scope_after_nested_commands(
    isolated_env,
):
    project = isolated_env / "ENACT"
    nested = project / "dolphinbench-supplement"
    nested.mkdir(parents=True)
    store = memory_core.EvidenceStore()

    memory_core.record_session_start(
        store,
        {"session_id": "session-1", "cwd": str(project), "source": "startup"},
    )
    memory_core.record_tool(
        store,
        {
            "session_id": "session-1",
            "cwd": str(nested),
            "tool_name": "Bash",
            "tool_input": {"command": "cat README.md"},
            "tool_response": {"stdout": "DolphinBench supplement"},
        },
    )

    rows = store.conn.execute(
        "SELECT repo_id, app_id FROM events WHERE session_id = ? ORDER BY id",
        ("session-1",),
    ).fetchall()
    store.close()

    expected = memory_core.resolve_repo(str(project))
    assert {row["repo_id"] for row in rows} == {expected.identity}
    assert {row["app_id"] for row in rows} == {expected.app_id}


def test_prepare_flush_marks_only_the_events_in_its_packet(isolated_env):
    store = memory_core.EvidenceStore()
    project = repo()
    nested = memory_core.RepoContext(
        cwd="/tmp/repo/supplement",
        root="/tmp/repo/supplement",
        identity="local:/tmp/repo/supplement",
        app_id="code-supplement",
        branch="detached",
        head_sha="",
    )
    project_first = store.record_event(
        project, "session-1", "user_prompt", {"text": "Explore the project."}
    )
    nested_event = store.record_event(
        nested,
        "session-1",
        "tool_result",
        {"command": "cat README.md", "result_preview": "Project details"},
    )
    project_last = store.record_event(
        project,
        "session-1",
        "assistant_stop",
        {"text": "Here is what the project does."},
    )

    packet_id, events = store.prepare_flush(project, "session-1", "session-end")
    flushed = {
        row["id"]: row["flush_id"]
        for row in store.conn.execute(
            "SELECT id, flush_id FROM events WHERE session_id = ? ORDER BY id",
            ("session-1",),
        ).fetchall()
    }
    store.close()

    assert [event["id"] for event in events] == [project_first, project_last]
    assert flushed[project_first] == packet_id
    assert flushed[project_last] == packet_id
    assert flushed[nested_event] is None


def _record_exchange(
    store: memory_core.EvidenceStore,
    number: int,
    *, session_id: str = "s1",
    text_size: int = 0,
) -> None:
    suffix = "x" * text_size
    store.record_event(
        repo(), session_id, "user_prompt", {"text": f"Question {number} {suffix}"}
    )
    store.record_event(
        repo(),
        session_id,
        "assistant_stop",
        {"text": f"Answer {number} {suffix}"},
    )


def test_periodic_checkpoint_waits_for_five_completed_exchanges(isolated_env):
    store = memory_core.EvidenceStore()
    for number in range(1, 5):
        _record_exchange(store, number)

    assert store.checkpoint_due(repo().identity, "s1") is False
    assert store.prepare_flush(repo(), "s1", "periodic") is None

    _record_exchange(store, 5)
    assert store.checkpoint_due(repo().identity, "s1") is True
    packet_id, events = store.prepare_flush(repo(), "s1", "periodic")

    assert packet_id
    assert memory_core.checkpoint_stats(events)[:2] == (5, 10)
    assert store.checkpoint_due(repo().identity, "s1") is False
    store.close()


def test_periodic_checkpoint_keeps_later_exchanges_for_the_next_packet(
    isolated_env,
):
    store = memory_core.EvidenceStore()
    for number in range(1, 8):
        _record_exchange(store, number)

    _, first = store.prepare_flush(repo(), "s1", "periodic")
    remaining = store.conn.execute(
        "SELECT payload_json FROM events WHERE flush_id IS NULL ORDER BY id"
    ).fetchall()

    assert memory_core.checkpoint_stats(first)[:2] == (5, 10)
    assert len(remaining) == 4
    assert "Question 6" in remaining[0]["payload_json"]
    store.close()


def test_large_completed_block_triggers_before_five_exchanges(isolated_env):
    store = memory_core.EvidenceStore()
    for number in range(1, 5):
        _record_exchange(store, number, text_size=6000)

    assert store.checkpoint_due(repo().identity, "s1") is True
    _, events = store.prepare_flush(repo(), "s1", "periodic")
    completed, messages, source_chars = memory_core.checkpoint_stats(events)

    assert completed == 4
    assert messages == 8
    assert source_chars >= memory_core.CHECKPOINT_SOURCE_CHARS
    store.close()


def test_forced_checkpoint_sends_an_incomplete_remainder(isolated_env):
    store = memory_core.EvidenceStore()
    _record_exchange(store, 1)
    store.record_event(repo(), "s1", "user_prompt", {"text": "Final question"})

    assert store.prepare_flush(repo(), "s1", "periodic") is None
    _, events = store.prepare_flush(repo(), "s1", "session-end")

    assert [event["kind"] for event in events] == [
        "user_prompt",
        "assistant_stop",
        "user_prompt",
    ]
    store.close()


def test_stop_schedules_one_background_checkpoint_when_block_is_ready(
    isolated_env,
):
    import hook_runner

    store = memory_core.EvidenceStore()
    for number in range(1, 5):
        _record_exchange(store, number)

    with (
        patch.object(hook_runner, "api_key", return_value="m0-test-key"),
        patch.object(hook_runner, "hand_off_flush") as handoff,
    ):
        assert (
            hook_runner.schedule_periodic_checkpoint(
                store, {"session_id": "s1", "cwd": "/tmp/repo"}, repo(), "s1"
            )
            is False
        )
        _record_exchange(store, 5)
        assert (
            hook_runner.schedule_periodic_checkpoint(
                store, {"session_id": "s1", "cwd": "/tmp/repo"}, repo(), "s1"
            )
            is True
        )
        _record_exchange(store, 6)
        assert (
            hook_runner.schedule_periodic_checkpoint(
                store, {"session_id": "s1", "cwd": "/tmp/repo"}, repo(), "s1"
            )
            is False
        )

    handoff.assert_called_once_with(
        {"session_id": "s1", "cwd": "/tmp/repo"}, "periodic"
    )
    assert store.conn.execute("SELECT COUNT(*) FROM flushes").fetchone()[0] == 1
    store.close()


def test_stop_schedules_idle_flush_when_periodic_not_due(isolated_env):
    import hook_runner

    store = memory_core.EvidenceStore()
    _record_exchange(store, 1)

    hook_input = {"session_id": "s1", "cwd": "/tmp/repo"}
    with (
        patch.object(hook_runner, "api_key", return_value="m0-test-key"),
        patch.object(hook_runner, "_launch_handoff", return_value=True) as launch,
    ):
        assert not hook_runner.schedule_periodic_checkpoint(store, hook_input, repo(), "s1")
        assert hook_runner.schedule_idle_flush(store, hook_input, repo(), "s1")

    payload = json.loads(launch.call_args.args[0].read_text(encoding="utf-8"))
    assert payload["reason"] == "idle"
    assert payload["delay_seconds"] == hook_runner.DEFAULT_IDLE_FLUSH_SECONDS
    store.close()


def test_idle_flush_skipped_when_inflight_flush_exists(isolated_env):
    import hook_runner

    store = memory_core.EvidenceStore()
    for number in range(1, 6):
        _record_exchange(store, number)

    hook_input = {"session_id": "s1", "cwd": "/tmp/repo"}
    with patch.object(hook_runner, "api_key", return_value="m0-test-key"):
        store.prepare_flush(repo(), "s1", "periodic")
        assert not hook_runner.schedule_idle_flush(store, hook_input, repo(), "s1")
    store.close()


def test_idle_flush_skipped_when_no_unflushed_events(isolated_env):
    import hook_runner

    store = memory_core.EvidenceStore()
    hook_input = {"session_id": "s1", "cwd": "/tmp/repo"}
    with patch.object(hook_runner, "api_key", return_value="m0-test-key"):
        assert not hook_runner.schedule_idle_flush(store, hook_input, repo(), "s1")
    store.close()


def test_idle_flush_skipped_when_disabled(isolated_env, monkeypatch):
    import hook_runner

    store = memory_core.EvidenceStore()
    _record_exchange(store, 1)

    monkeypatch.setenv("NEATMEM_CODE_IDLE_FLUSH_SECONDS", "0")
    hook_input = {"session_id": "s1", "cwd": "/tmp/repo"}
    with patch.object(hook_runner, "api_key", return_value="m0-test-key"):
        assert not hook_runner.schedule_idle_flush(store, hook_input, repo(), "s1")
    store.close()


def test_stop_hook_falls_through_to_idle_flush(isolated_env, monkeypatch):
    import hook_runner

    store = memory_core.EvidenceStore()
    _record_exchange(store, 1)

    hook_input = {"session_id": "s1", "cwd": "/tmp/repo"}
    periodic_called = []
    idle_called = []

    original_periodic = hook_runner.schedule_periodic_checkpoint

    def mock_periodic(*a, **kw):
        result = original_periodic(*a, **kw)
        periodic_called.append(result)
        return result

    def mock_idle(*a, **kw):
        idle_called.append(True)
        return True

    with (
        patch.object(hook_runner, "api_key", return_value="m0-test-key"),
        patch.object(hook_runner, "schedule_periodic_checkpoint", side_effect=mock_periodic),
        patch.object(hook_runner, "schedule_idle_flush", side_effect=mock_idle),
    ):
        monkeypatch.setattr("sys.stdin", __import__("io").StringIO(json.dumps(hook_input)))
        monkeypatch.setattr("sys.argv", ["hook.py", "stop"])
        hook_runner.run(
            record_stop_fn=lambda s, h: (repo(), "s1"),
            data_dir_env="NEATMEM_CODE_DATA_DIR",
            automatic_flush_reasons={"session-end", "pre-compact"},
        )

    assert periodic_called == [False]
    assert idle_called == [True]
    store.close()


def test_flush_worker_sleeps_for_delay_seconds(isolated_env, monkeypatch):
    import flush_worker

    data_dir_path = Path(os.environ["NEATMEM_CODE_DATA_DIR"])
    pending_dir = data_dir_path / "pending"
    pending_dir.mkdir(parents=True, exist_ok=True)
    handoff_path = pending_dir / "idle-test.running"
    handoff_path.write_text(
        json.dumps({
            "hook_input": {"session_id": "s1", "cwd": "/tmp/repo"},
            "reason": "idle",
            "delay_seconds": 42,
        }),
        encoding="utf-8",
    )

    slept = []
    rewritten_content = {}

    def mock_sleep(seconds):
        slept.append(seconds)
        rewritten_content.update(
            json.loads(handoff_path.read_text(encoding="utf-8"))
        )

    monkeypatch.setattr(time, "sleep", mock_sleep)
    monkeypatch.setattr(
        "sys.argv", ["flush_worker.py", str(handoff_path)]
    )
    with patch.object(flush_worker, "checkpoint_session", return_value={"status": "nothing-to-flush"}):
        flush_worker.main()

    assert slept == [42.0]
    assert "delay_seconds" not in rewritten_content


def test_session_end_worker_does_not_repeat_captured_final_response(isolated_env, monkeypatch):
    import flush_worker

    monkeypatch.setenv("NEATMEM_API_KEY", "test-key")
    monkeypatch.setattr(memory_core, "resolve_repo", lambda cwd: repo())
    store = memory_core.EvidenceStore()
    for index in range(5):
        store.record_event(repo(), "s1", "user_prompt", {"text": f"Earlier prompt {index}"})
        store.record_event(repo(), "s1", "assistant_stop", {"text": f"Earlier answer {index}"})
    first_packet, _ = store.prepare_flush(repo(), "s1", "periodic")
    store.update_flush(first_packet, status="prepared")

    prompt = "How does the serializer handle dates?"
    answer = "The serializer preserves timezone-naive dates."
    transcript = isolated_env / "session.jsonl"
    _write_transcript(transcript, "s1", [
        {"type": "user", "message": {"role": "user", "content": prompt}},
        {"type": "assistant", "message": {"role": "assistant", "content": answer}},
    ])
    hook_input = {
        "session_id": "s1", "cwd": "/tmp/repo",
        "transcript_path": str(transcript), "last_assistant_message": answer,
    }
    memory_core.record_user_prompt(store, {**hook_input, "prompt": prompt})
    transcript_mod.record_stop(store, hook_input)
    # SessionEnd captures again before handing off, using the same transcript cursor.
    transcript_mod.record_stop(store, hook_input)
    handoff = isolated_env / "session-end.running"
    handoff.write_text(json.dumps({
        "hook_input": hook_input, "reason": "session-end", "wait_for_inflight": True,
    }))
    monkeypatch.setattr(sys, "argv", ["flush_worker.py", str(handoff)])
    monkeypatch.setenv("NEATMEM_CODE_HANDOFF_PATH", str(handoff))
    monkeypatch.setattr(flush_worker.time, "sleep", lambda seconds: store.update_flush(
        first_packet, status="succeeded",
    ))
    with patch.object(
        memory_core, "_request_json", side_effect=_flush_server(1)
    ) as request:
        assert flush_worker.main() == 0

    add_bodies = _add_bodies(request)
    assert len(add_bodies) == 1
    messages = add_bodies[0]["messages"]
    assert sum(prompt in message["content"] for message in messages) == 1
    assert sum(answer in message["content"] for message in messages) == 1
    store.close()


def test_default_stop_records_each_exchange_once_across_repeated_hooks(isolated_env, monkeypatch):
    import hook_runner

    monkeypatch.setattr(memory_core, "resolve_repo", lambda cwd: repo())
    store = memory_core.EvidenceStore()
    hook_input = {"session_id": "s1", "cwd": "/tmp/repo", "last_assistant_message": "The parser uses UTC."}
    for _ in range(2):
        memory_core.record_user_prompt(store, {**hook_input, "prompt": "What timezone does the parser use?"})
        hook_runner.default_record_stop(store, hook_input)
        # A tool notification and a flush must not make a repeated Stop look like a new answer.
        store.record_event(repo(), "s1", "tool_result", {"tool": "Read"})
        packet, _ = store.prepare_flush(repo(), "s1", "session-end")
        store.update_flush(packet, status="succeeded")
        hook_runner.default_record_stop(store, hook_input)

    responses = store.conn.execute("SELECT payload_json FROM events WHERE kind = 'assistant_stop'").fetchall()
    assert [json.loads(row[0])["text"] for row in responses] == ["The parser uses UTC."] * 2
    assert not store.has_unflushed_events(repo().identity, "s1")
    store.close()


def test_concurrent_response_hooks_record_one_answer(isolated_env):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    store = memory_core.EvidenceStore()
    store.record_event(repo(), "s1", "user_prompt", {"text": "Inspect the parser."})
    ready = Barrier(2)

    def capture():
        connection = memory_core.EvidenceStore(store.path)
        try:
            ready.wait(timeout=5)
            connection.record_assistant_response(repo(), "s1", "The parser uses UTC.")
        finally:
            connection.close()

    with ThreadPoolExecutor(max_workers=2) as workers:
        list(workers.map(lambda _: capture(), range(2)))
    assert store.conn.execute("SELECT COUNT(*) FROM events WHERE kind = 'assistant_stop'").fetchone()[0] == 1
    store.record_assistant_response(repo(), "s1", "The serializer also uses UTC.")
    store.record_assistant_response(repo(), "s2", "The parser uses UTC.")
    assert store.conn.execute("SELECT COUNT(*) FROM events WHERE kind = 'assistant_stop'").fetchone()[0] == 3
    store.close()


def test_flush_worker_restores_harness_identity_from_child_environment(isolated_env, monkeypatch):
    import flush_worker

    handoff_path = Path(os.environ["NEATMEM_CODE_DATA_DIR"]) / "pending" / "kimi.running"
    handoff_path.parent.mkdir(parents=True, exist_ok=True)
    handoff_path.write_text(
        json.dumps({"hook_input": {"session_id": "s1", "cwd": "/tmp/repo"}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("NEATMEM_PLUGIN_HARNESS", "kimi")
    monkeypatch.setenv("NEATMEM_PLUGIN_ENV_PREFIX", "NEATMEM_KIMI")
    monkeypatch.setenv("NEATMEM_PLUGIN_DATA_DIR_NAME", "kimi-plugin")
    monkeypatch.setenv("NEATMEM_PLUGIN_SOURCE_TAG", "kimi_plugin")
    monkeypatch.setattr("sys.argv", ["flush_worker.py", str(handoff_path)])

    with (
        patch.object(flush_worker, "configure_harness", wraps=memory_core.configure_harness) as configure,
        patch.object(flush_worker, "checkpoint_session", return_value={"status": "nothing-to-flush"}),
    ):
        flush_worker.main()

    configure.assert_called_once_with(
        "kimi",
        env_prefix="NEATMEM_KIMI",
        data_dir_name="kimi-plugin",
        source_tag="kimi_plugin",
    )
    memory_core.configure_harness(
        "claude-code", data_dir_name="claude-code-plugin", source_tag="claude_code_plugin"
    )


def test_launch_handoff_resolves_flush_worker_under_core(isolated_env):
    """hook.py lives in adapters/claude/; flush_worker.py lives in core/.

    _launch_handoff must compute the worker path relative to core/, not as a
    sibling of hook.py, or every background flush launches against a
    nonexistent path.
    """
    import hook_runner

    data_dir_path = Path(os.environ["NEATMEM_CODE_DATA_DIR"])
    pending_dir = data_dir_path / "pending"
    pending_dir.mkdir(parents=True, exist_ok=True)
    handoff_path = pending_dir / "test-handoff.json"
    handoff_path.write_text("{}", encoding="utf-8")

    with patch.object(hook_runner.subprocess, "Popen") as popen:
        assert hook_runner._launch_handoff(handoff_path) is True

    launched_args = popen.call_args.args[0]
    worker_path = Path(launched_args[1])
    assert worker_path == CORE / "flush_worker.py"
    assert worker_path.is_file()
    child_env = popen.call_args.kwargs["env"]
    assert child_env["NEATMEM_PLUGIN_HARNESS"] == "claude-code"
    assert child_env["NEATMEM_PLUGIN_DATA_DIR_NAME"] == "claude-code-plugin"
    assert child_env["NEATMEM_PLUGIN_SOURCE_TAG"] == "claude_code_plugin"
    assert child_env["NEATMEM_CODE_DATA_DIR"] == os.environ["NEATMEM_CODE_DATA_DIR"]


def test_flush_worker_detaches_on_windows_as_well_as_posix(monkeypatch):
    """start_new_session is a POSIX no-op on Windows, where Claude Code also ships.

    Without a creationflags fallback the worker stays attached to Claude's
    console on Windows and background extraction can die with the session.
    """
    import hook_runner

    assert hook_runner.detached_process_kwargs("darwin") == {"start_new_session": True}
    assert hook_runner.detached_process_kwargs("linux") == {"start_new_session": True}

    monkeypatch.setattr(subprocess, "DETACHED_PROCESS", 0x00000008, raising=False)
    monkeypatch.setattr(
        subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200, raising=False
    )
    windows_kwargs = hook_runner.detached_process_kwargs("win32")
    assert windows_kwargs == {"creationflags": 0x00000208}
    assert "start_new_session" not in windows_kwargs


def test_launch_handoff_always_requests_detachment(isolated_env):
    import hook_runner

    pending_dir = Path(os.environ["NEATMEM_CODE_DATA_DIR"]) / "pending"
    pending_dir.mkdir(parents=True, exist_ok=True)
    handoff_path = pending_dir / "detach-handoff.json"
    handoff_path.write_text("{}", encoding="utf-8")

    with patch.object(hook_runner.subprocess, "Popen") as popen:
        assert hook_runner._launch_handoff(handoff_path) is True

    assert popen.call_args.kwargs["start_new_session"] is True


def test_pending_recovery_is_capped_and_leaves_the_rest_for_next_session(
    isolated_env,
):
    """One worker per packet at once can pile up after a long API outage.

    Each worker can block for NEATMEM_CODE_EXTRACTION_WAIT_SECONDS, so an
    uncapped launch is a thundering herd at session start.
    """
    import hook_runner

    pending_dir = Path(os.environ["NEATMEM_CODE_DATA_DIR"]) / "pending"
    pending_dir.mkdir(parents=True, exist_ok=True)
    for number in range(hook_runner.PENDING_LAUNCH_LIMIT + 3):
        (pending_dir / f"packet-{number}.json").write_text("{}", encoding="utf-8")

    with patch.object(hook_runner.subprocess, "Popen"):
        launched = hook_runner.recover_pending_handoffs()

    assert launched == hook_runner.PENDING_LAUNCH_LIMIT
    assert len(list(pending_dir.glob("*.running"))) == hook_runner.PENDING_LAUNCH_LIMIT
    assert len(list(pending_dir.glob("*.json"))) == 3


def test_pending_recovery_drops_packets_past_the_expiry_window(isolated_env):
    import hook_runner

    pending_dir = Path(os.environ["NEATMEM_CODE_DATA_DIR"]) / "pending"
    pending_dir.mkdir(parents=True, exist_ok=True)
    expired = pending_dir / "expired.json"
    expired.write_text("{}", encoding="utf-8")
    fresh = pending_dir / "fresh.json"
    fresh.write_text("{}", encoding="utf-8")

    stale_time = time.time() - hook_runner.PENDING_EXPIRY_SECONDS - 60
    os.utime(expired, (stale_time, stale_time))

    with patch.object(hook_runner.subprocess, "Popen"):
        launched = hook_runner.recover_pending_handoffs()

    assert launched == 1
    assert not expired.exists()
    assert fresh.with_suffix(".running").exists()


def test_session_end_promotes_an_inflight_periodic_checkpoint(isolated_env):
    store = memory_core.EvidenceStore()
    for number in range(1, 6):
        _record_exchange(store, number)

    packet_id, _ = store.prepare_flush(repo(), "s1", "periodic")
    store.update_flush(
        packet_id, status="semantic-queued", semantic_event_id="event-1"
    )
    prepared_id, _ = store.prepare_flush(repo(), "s1", "session-end")
    record = store.flush_record(packet_id)

    assert prepared_id == packet_id
    assert record["reason"] == "session-end"
    store.close()


def _record_complete_session(
    store: memory_core.EvidenceStore,
    session_id: str = "s1",
    repo_context: memory_core.RepoContext | None = None,
) -> None:
    r = repo_context or repo()
    store.record_event(
        r,
        session_id,
        "session_start",
        {"source": "startup", "model": "", "branch": "main", "head_sha": "abc123"},
    )
    store.record_event(
        r,
        session_id,
        "user_prompt",
        {"text": "Fix the ODS datetime serialization bug."},
    )
    store.record_event(
        r,
        session_id,
        "tool_result",
        {"tool": "Read", "path": "/tmp/repo/src/ods.py", "failed": False},
    )
    store.record_event(
        r,
        session_id,
        "tool_result",
        {
            "tool": "Edit",
            "path": "/tmp/repo/src/ods.py",
            "failed": False,
            "mutation_chars": 42,
        },
    )
    store.record_event(
        r,
        session_id,
        "tool_result",
        {
            "tool": "Bash",
            "command": "pytest tests/test_ods.py",
            "command_kind": "test",
            "failed": False,
            "result_preview": "3 passed",
        },
    )
    store.record_event(
        r,
        session_id,
        "assistant_stop",
        {"text": "Fixed serialization in src/ods.py and the focused tests pass."},
    )


def test_flush_submits_only_session_dialogue_with_shared_identity(
    isolated_env, monkeypatch
):
    store = memory_core.EvidenceStore()
    _record_complete_session(store)
    store.record_event(
        repo(),
        "s1",
        "user_prompt",
        {"text": "Remember that ODS dates must remain timezone-naive."},
    )
    store.record_event(
        repo(),
        "s1",
        "assistant_stop",
        {"text": "The serializer now preserves timezone-naive ODS dates."},
    )
    store.record_event(
        repo(),
        "s1",
        "subagent_stop",
        {
            "agent_id": "agent-1",
            "agent_type": "neatmem:sidekick",
            "final_message": "Confirmed that src/ods.py is the only writer.",
        },
    )
    hook_input = {
        "session_id": "s1",
        "cwd": "/tmp/repo",
        "task": "Canonical issue: preserve ODS datetime formatting.",
        "task_outcome": "Submission PASSED (eval status: completed).",
        "instance_id": "example__repo-123",
    }

    with (
        patch.object(memory_core, "resolve_repo", return_value=repo()),
        patch.object(
            memory_core, "_request_json", side_effect=_flush_server(2)
        ) as request,
    ):
        result = memory_core.flush_session(store, hook_input, "session-end")

    assert result["status"] == "succeeded"
    assert result["extracted_count"] == 2
    assert result["message_count"] == 4
    assert request.call_count == 2

    (sent_body,) = _add_bodies(request)
    assert set(sent_body) == {"messages", "user_id", "agent_id", "run_id"}
    assert sent_body["agent_id"] == "code-example"
    assert sent_body["user_id"] == "test-user"
    assert sent_body["run_id"] == "s1"
    flush_body = request.call_args_list[-1].args[2]
    assert request.call_args_list[-1].args[0].endswith("/v1/messages/flush/")
    assert flush_body == {
        "user_id": "test-user",
        "agent_id": "code-example",
        "run_id": "s1",
    }

    assert [message["role"] for message in sent_body["messages"]] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    serialized = json.dumps(sent_body)
    assert "Canonical issue: preserve ODS datetime formatting." not in serialized
    assert "Fix the ODS datetime serialization bug." in serialized
    assert "Remember that ODS dates must remain timezone-naive." in serialized
    assert "Fixed serialization in src/ods.py" in serialized
    assert "The serializer now preserves timezone-naive ODS dates." in serialized
    assert "Sidekick outcome" not in serialized
    assert "Confirmed that src/ods.py is the only writer." not in serialized
    assert "pytest tests/test_ods.py" not in serialized
    assert "Submission PASSED" not in serialized
    assert "Files read" not in serialized
    assert "FEEDBACK FROM PREVIOUS ACTION" not in serialized
    assert "task_episode" not in serialized
    operation = store.conn.execute(
        "SELECT item_count FROM operations WHERE operation = 'flush'"
    ).fetchone()
    assert operation["item_count"] == 2
    store.close()


def test_non_git_memory_metadata_omits_branch_and_commit(isolated_env, monkeypatch):
    store = memory_core.EvidenceStore()
    non_git_repo = memory_core.RepoContext(
        cwd="/tmp/workspace",
        root="/tmp/workspace",
        identity="local:/tmp/workspace",
        app_id="workspace",
        branch="detached",
        head_sha="",
        project_id="local-workspace-0123456789",
    )
    store.record_event(
        non_git_repo,
        "s1",
        "user_prompt",
        {"text": "Explain how this workspace loads benchmark personas."},
    )
    store.record_event(
        non_git_repo,
        "s1",
        "assistant_stop",
        {"text": "The persona loader reads definitions from personas.json."},
    )

    with (
        patch.object(memory_core, "resolve_repo", return_value=non_git_repo),
        patch.object(
            memory_core, "_request_json", side_effect=_flush_server(2)
        ) as request,
    ):
        result = memory_core.flush_session(
            store, {"session_id": "s1", "cwd": "/tmp/workspace"}, "session-end"
        )

    assert result["status"] == "succeeded"
    (sent_body,) = _add_bodies(request)
    # The write carries the shared identity only; no branch/commit metadata.
    assert set(sent_body) == {"messages", "user_id", "agent_id", "run_id"}
    assert sent_body["agent_id"] == "local-workspace-0123456789"
    store.close()


def test_large_extraction_keeps_agent_assignment_and_response_together():
    messages = [
        {"role": "user", "content": "Investigate the repository."},
        {
            "role": "assistant",
            "content": "Subagent assignment (Explore):\nFind the parser.",
        },
        {
            "role": "assistant",
            "content": "Subagent response (Explore):\n" + ("parser details " * 80),
        },
        {
            "role": "assistant",
            "content": "Subagent assignment (Explore):\nFind the callers.",
        },
        {
            "role": "assistant",
            "content": "Subagent response (Explore):\n" + ("caller details " * 80),
        },
        {"role": "assistant", "content": "Main Claude response:\nConclusion."},
    ]

    batches = memory_core.extraction_message_batches(messages, max_tokens=550)

    flattened = [message for batch in batches for message in batch]
    assert flattened == messages
    assert len(batches) > 1
    locations = {
        message["content"]: batch_index
        for batch_index, batch in enumerate(batches)
        for message in batch
    }
    assert locations[messages[1]["content"]] == locations[messages[2]["content"]]
    assert locations[messages[3]["content"]] == locations[messages[4]["content"]]
    assert flattened[-1]["content"].startswith("Main Claude response:")


def test_large_extraction_submits_every_batch_before_flushing(
    isolated_env, monkeypatch
):
    store = memory_core.EvidenceStore()
    _record_complete_session(store)
    messages = [
        {"role": "user", "content": "A" * 20000},
        {"role": "assistant", "content": "B" * 20000},
        {"role": "user", "content": "C" * 20000},
        {"role": "assistant", "content": "D" * 20000},
    ]
    calls = []

    def request(url, key, body, timeout):
        calls.append((url, body))
        if url.endswith("/v1/messages/flush/"):
            return {"extracted_count": 3}, 100, 20
        return {}, 100, 20

    with (
        patch.object(memory_core, "resolve_repo", return_value=repo()),
        patch.object(memory_core, "build_extraction_messages", return_value=messages),
        patch.object(memory_core, "_request_json", side_effect=request),
    ):
        result = memory_core.flush_session(
            store,
            {"session_id": "s1", "cwd": "/tmp/repo"},
            "session-end",
        )

    urls = [url for url, _ in calls]
    assert [url.rsplit("/", 3)[-2] for url in urls] == ["add", "add", "flush"]
    posted = [body for url, body in calls if url.endswith("/v1/messages/add/")]
    assert len(posted) == 2
    posted_messages = [message for body in posted for message in body["messages"]]
    # flush attaches the packet-level event_at to every uploaded message
    event_ats = {message.get("event_at") for message in posted_messages}
    assert len(event_ats) == 1 and None not in event_ats and "" not in event_ats
    assert [
        {k: v for k, v in message.items() if k != "event_at"}
        for message in posted_messages
    ] == messages
    for _, body in calls:
        assert body["agent_id"] == "code-example"
        assert body["user_id"] == "test-user"
        assert body["run_id"] == "s1"
    assert result["status"] == "succeeded"
    assert result["message_count"] == 4
    assert result["extracted_count"] == 3
    store.close()


def test_failed_checkpoint_reuses_the_same_packet_on_retry(isolated_env, monkeypatch):
    store = memory_core.EvidenceStore()
    _record_complete_session(store)
    hook_input = {"session_id": "s1", "cwd": "/tmp/repo"}

    with (
        patch.object(memory_core, "resolve_repo", return_value=repo()),
        patch.object(
            memory_core, "_request_json", side_effect=OSError("temporary network error")
        ),
    ):
        first = memory_core.flush_session(store, hook_input, "session-end")

    with (
        patch.object(memory_core, "resolve_repo", return_value=repo()),
        patch.object(memory_core, "_request_json", side_effect=_flush_server(1)),
    ):
        second = memory_core.flush_session(store, hook_input, "session-end")

    assert first["status"] == "error"
    assert second["status"] == "succeeded"
    assert first["packet_id"] == second["packet_id"]
    assert store.conn.execute("SELECT COUNT(*) FROM flushes").fetchone()[0] == 1
    store.close()


def test_session_end_flushes_events_recorded_after_an_earlier_packet_was_queued(
    isolated_env, monkeypatch
):
    store = memory_core.EvidenceStore()
    store.record_event(
        repo(), "s1", "user_prompt", {"text": "Inspect the serializer."}
    )
    first_packet, _ = store.prepare_flush(repo(), "s1", "pre-compact")
    store.update_flush(first_packet, status="succeeded")
    store.record_event(
        repo(),
        "s1",
        "assistant_stop",
        {"text": "The serializer preserves timezone-naive dates."},
    )

    with (
        patch.object(memory_core, "resolve_repo", return_value=repo()),
        patch.object(
            memory_core, "_request_json", side_effect=_flush_server(1)
        ) as request,
    ):
        result = memory_core.flush_session(
            store, {"session_id": "s1", "cwd": "/tmp/repo"}, "session-end"
        )

    assert result["status"] == "succeeded"
    posted = json.dumps(_add_bodies(request))
    assert "Inspect the serializer." not in posted
    assert "timezone-naive dates" in posted
    assert (
        store.conn.execute(
            "SELECT COUNT(*) FROM events WHERE flush_id IS NULL"
        ).fetchone()[0]
        == 0
    )
    assert store.conn.execute("SELECT COUNT(*) FROM flushes").fetchone()[0] == 2
    store.close()


def test_completed_flush_does_not_consume_a_resumed_invocation(
    isolated_env, monkeypatch
):
    store = memory_core.EvidenceStore()
    store.record_event(
        repo(), "s1", "user_prompt", {"text": "Inspect the serializer."}
    )
    first_packet, _ = store.prepare_flush(repo(), "s1", "session-end")
    store.update_flush(first_packet, status="prepared")
    store.record_event(repo(), "s1", "session_start", {"source": "resume"})
    store.record_event(
        repo(), "s1", "user_prompt", {"text": "Confirm the serializer."}
    )
    store.record_event(
        repo(),
        "s1",
        "assistant_stop",
        {"text": "The serializer preserves timezone-naive dates."},
    )

    with (
        patch.object(memory_core, "resolve_repo", return_value=repo()),
        patch.object(
            memory_core, "_request_json", side_effect=_flush_server(1)
        ) as request,
    ):
        result = memory_core.flush_session(
            store, {"session_id": "s1", "cwd": "/tmp/repo"}, "session-end"
        )

    assert result["status"] == "succeeded"
    posted = json.dumps(_add_bodies(request))
    assert "Inspect the serializer." in posted
    # The resumed session's events stay queued for a later flush.
    assert "Confirm the serializer." not in posted
    assert (
        store.conn.execute(
            "SELECT COUNT(*) FROM events WHERE flush_id IS NULL"
        ).fetchone()[0]
        == 3
    )
    store.close()


def test_legacy_memory_write_event_does_not_bypass_production_extraction(
    isolated_env, monkeypatch
):
    store = memory_core.EvidenceStore()
    _record_complete_session(store)
    store.record_event(
        repo(),
        "s1",
        "memory_write",
        {"kind": "learned", "fact_count": 3, "status": "stored", "memory_count": 3},
    )

    with (
        patch.object(memory_core, "resolve_repo", return_value=repo()),
        patch.object(
            memory_core, "_request_json", side_effect=_flush_server(2)
        ) as request,
    ):
        first = memory_core.flush_session(
            store, {"session_id": "s1", "cwd": "/tmp/repo"}, "session-end"
        )

    assert first["status"] == "succeeded"
    assert "fact_count" not in json.dumps(_add_bodies(request))
    store.close()


def test_flush_succeeds_without_an_api_key(isolated_env):
    store = memory_core.EvidenceStore()
    _record_complete_session(store)

    with (
        patch.object(memory_core, "resolve_repo", return_value=repo()),
        patch.object(
            memory_core, "_request_json", side_effect=_flush_server(2)
        ) as request,
    ):
        result = memory_core.flush_session(
            store, {"session_id": "s1", "cwd": "/tmp/repo"}, "session-end"
        )

    assert result["status"] == "succeeded"
    assert all(call.args[1] == "" for call in request.call_args_list)
    store.close()


def test_search_is_repo_scoped_and_does_not_reinject_seen_results(
    isolated_env, monkeypatch
):
    store = memory_core.EvidenceStore()
    captured_payloads = []

    def fake_request(url, key, payload, timeout):
        captured_payloads.append(payload)
        return (
            [
                {
                    "id": "raw-episode",
                    "memory": "Verified coding-session episode with raw commands.",
                    "metadata": {"record_kind": "task_episode"},
                },
                {
                    "id": "memory-1",
                    "memory": "The ODS serializer is in src/ods.py.",
                    "score": 0.82,
                    "metadata": {"record_kind": "durable_candidate"},
                },
                {
                    "id": "lower-score",
                    "memory": "A weakly related repository fact.",
                    "score": 0.2,
                    "metadata": {"record_kind": "durable_candidate"},
                },
                {
                    "id": "missing-score",
                    "memory": "An unscored repository fact.",
                    "metadata": {"record_kind": "durable_candidate"},
                },
            ],
            100,
            100,
        )

    with patch.object(memory_core, "_request_json", side_effect=fake_request):
        first_result = memory_core.search_memories(
            store, repo(), "s1", "Where is ODS serialization implemented?"
        )
        second_result = memory_core.search_memories(
            store, repo(), "s1", "Where is ODS serialization implemented?"
        )

    first = first_result.memories
    assert [memory["id"] for memory in first] == [
        "raw-episode",
        "memory-1",
        "lower-score",
    ]
    rendered = memory_core.format_context(first)
    assert "[durable_candidate]" not in rendered
    assert "2. The ODS serializer is in src/ods.py." in rendered
    assert second_result.memories == []
    assert second_result.already_shown_count == 3
    assert set(captured_payloads[0]) == {"query", "filters", "top_k"}
    assert captured_payloads[0]["filters"] == {"agent_id": "code-example"}
    assert captured_payloads[0]["top_k"] == 3
    retrieval = store.conn.execute(
        "SELECT rank, score, memory_text, context_chars FROM retrievals "
        "WHERE memory_id = 'memory-1'"
    ).fetchone()
    assert dict(retrieval) == {
        "rank": 2,
        "score": 0.82,
        "memory_text": "The ODS serializer is in src/ods.py.",
        "context_chars": 36,
    }
    assert store.conn.execute("SELECT COUNT(*) FROM retrievals").fetchone()[0] == 3
    store.close()


def test_format_context_labels_memories_from_non_main_branches():
    rendered = memory_core.format_context(
        [
            {
                "id": "from-main",
                "memory": "The default parser is in parser.py.",
                "metadata": {"branch": "main"},
            },
            {
                "id": "from-feature",
                "memory": "The feature parser adds strict mode.",
                "metadata": {"branch": "feat/strict-parser"},
            },
            {
                "id": "from-unknown",
                "memory": "The fallback parser accepts plain text.",
                "metadata": {"branch": "unknown"},
            },
        ]
    )

    assert "1. The default parser is in parser.py." in rendered
    assert (
        "2. The feature parser adds strict mode. "
        "[learnt on branch feat/strict-parser]" in rendered
    )
    assert "3. The fallback parser accepts plain text." in rendered
    assert "branch unknown" not in rendered
    assert "[mem0:" not in rendered


def test_sidekick_reuses_parent_memory_once_and_records_lifecycle(
    isolated_env, monkeypatch
):
    store = memory_core.EvidenceStore()
    store.mark_injected(
        "s1",
        repo().identity,
        [
            {
                "id": "memory-1",
                "memory": "ODS serialization lives in src/tablib/formats/_ods.py.",
                "score": 0.91,
            }
        ],
    )
    start_input = {
        "session_id": "s1",
        "cwd": "/tmp/repo/.claude/worktrees/sidekick-worker",
        "agent_id": "agent-123",
        "agent_type": "neatmem:sidekick",
    }

    with patch.object(memory_core, "resolve_repo", return_value=repo()):
        first = memory_core.record_subagent_start(store, start_input)
        repeated = memory_core.record_subagent_start(store, start_input)
        memory_core.record_subagent_stop(
            store,
            {
                **start_input,
                "agent_transcript_path": "/tmp/transcript.jsonl",
                "last_assistant_message": "Outcome: fixed and tested.",
            },
        )

    assert "src/tablib/formats/_ods.py" in first
    assert repeated == ""
    run = store.conn.execute("SELECT * FROM sidekick_runs").fetchone()
    assert run["agent_id"] == "agent-123"
    assert run["context_chars"] == len(first)
    assert run["stopped_at"]
    assert run["final_message"] == "Outcome: fixed and tested."
    events = [
        row[0]
        for row in store.conn.execute("SELECT kind FROM events ORDER BY id").fetchall()
    ]
    assert events == ["subagent_start", "subagent_start", "subagent_stop"]
    store.close()


def test_subagent_stop_without_agent_id_closes_latest_matching_run(isolated_env):
    store = memory_core.EvidenceStore()
    start_input = {
        "session_id": "s1",
        "cwd": "/tmp/repo",
        "agent_id": "agent-123",
        "agent_type": "sidekick",
    }

    with patch.object(memory_core, "resolve_repo", return_value=repo()):
        memory_core.record_subagent_start(store, start_input)
        memory_core.record_subagent_stop(
            store,
            {
                "session_id": "s1",
                "cwd": "/tmp/repo",
                "agent_type": "sidekick",
                "last_assistant_message": "Done.",
            },
        )

    rows = store.conn.execute("SELECT agent_id, stopped_at, final_message FROM sidekick_runs").fetchall()
    store.close()

    assert len(rows) == 1
    assert rows[0]["agent_id"] == "agent-123"
    assert rows[0]["stopped_at"]
    assert rows[0]["final_message"] == "Done."


def test_search_once_per_session_avoids_repeated_remote_searches(
    isolated_env, monkeypatch
):
    monkeypatch.setenv("NEATMEM_API_KEY", "m0-test-key")
    monkeypatch.setenv("NEATMEM_CODE_SEARCH_ONCE_PER_SESSION", "true")
    store = memory_core.EvidenceStore()

    with patch.object(
        memory_core,
        "_request_json",
        return_value=({"results": []}, 100, 20),
    ) as request:
        memory_core.search_memories(
            store, repo(), "s1", "Find the parser implementation"
        )
        memory_core.search_memories(
            store, repo(), "s1", "Now inspect its callers and tests"
        )

    request.assert_called_once()
    store.close()


def test_plugin_top_k_option_does_not_enable_threshold_or_reranking(
    isolated_env, monkeypatch
):
    monkeypatch.setenv("CLAUDE_PLUGIN_OPTION_API_KEY", "m0-test-key")
    monkeypatch.setenv("CLAUDE_PLUGIN_OPTION_TOP_K", "3")
    monkeypatch.setenv("CLAUDE_PLUGIN_OPTION_THRESHOLD", "0.6")
    monkeypatch.setenv("NEATMEM_CODE_RERANK", "true")
    store = memory_core.EvidenceStore()

    with patch.object(
        memory_core,
        "_request_json",
        return_value=({"results": []}, 100, 20),
    ) as request:
        memory_core.search_memories(
            store, repo(), "s1", "Find the parser implementation"
        )

    payload = request.call_args.args[2]
    assert payload["top_k"] == 3
    assert set(payload) == {"query", "filters", "top_k"}
    store.close()


def test_search_retries_one_transient_network_failure(isolated_env, monkeypatch):
    monkeypatch.setenv("NEATMEM_API_KEY", "m0-test-key")
    store = memory_core.EvidenceStore()

    with (
        patch.object(
            memory_core,
            "_request_json",
            side_effect=[
                memory_core.urllib.error.URLError("temporary DNS failure"),
                ({"results": []}, 100, 20),
            ],
        ) as request,
        patch.object(memory_core.time, "sleep") as sleep,
    ):
        result = memory_core.search_memories(
            store, repo(), "s1", "Find the parser implementation"
        )

    assert result.memories == []
    assert result.succeeded is True
    assert request.call_count == 2
    sleep.assert_called_once_with(0.25)
    operation = store.conn.execute(
        "SELECT success, error FROM operations WHERE operation = 'search'"
    ).fetchone()
    assert dict(operation) == {"success": 1, "error": ""}
    store.close()


def test_search_result_separates_memories_from_matches_already_shown(
    isolated_env, monkeypatch
):
    monkeypatch.setenv("NEATMEM_API_KEY", "m0-test-key")
    store = memory_core.EvidenceStore()
    existing = {
        "id": "existing-memory",
        "memory": "The parser entrypoint is src/parser.py.",
        "metadata": {"record_kind": "durable_candidate"},
    }
    new = {
        "id": "new-memory",
        "memory": "Parser regression tests live in tests/test_parser.py.",
        "metadata": {"record_kind": "durable_candidate"},
    }
    store.mark_injected("s1", repo().identity, [existing])

    with patch.object(
        memory_core,
        "_request_json",
        return_value=({"results": [existing, new]}, 100, 200),
    ):
        result = memory_core.search_memories(
            store, repo(), "s1", "Where is the parser implemented and tested?"
        )

    assert result.succeeded is True
    assert result.matched_count == 2
    assert result.already_shown_count == 1
    assert [memory["id"] for memory in result.memories] == ["new-memory"]
    store.close()


def test_stateless_search_returns_the_requested_memories_each_time(
    isolated_env, monkeypatch
):
    monkeypatch.setenv("NEATMEM_API_KEY", "m0-test-key")
    memory = {
        "id": "memory-1",
        "memory": "The parser entrypoint is src/parser.py.",
        "metadata": {"record_kind": "durable_candidate"},
    }

    with patch.object(
        memory_core,
        "_request_json",
        return_value=({"results": [memory]}, 100, 200),
    ):
        first = memory_core.search_memories(None, repo(), None, "parser")
        second = memory_core.search_memories(None, repo(), None, "parser")

    assert first.memories == [memory]
    assert second.memories == [memory]
    assert first.already_shown_count == 0
    assert second.already_shown_count == 0


def test_explicit_search_formats_memories_and_empty_results():
    new_result = memory_core.MemorySearchResult(
        succeeded=True,
        matched_count=2,
        already_shown_count=1,
        memories=[
            {
                "id": "new-memory",
                "memory": "Parser tests live in tests/test_parser.py.",
            }
        ],
    )
    zero_result = memory_core.MemorySearchResult(
        succeeded=True,
        matched_count=0,
        already_shown_count=0,
        memories=[],
    )
    empty_result = memory_core.MemorySearchResult(
        succeeded=True,
        matched_count=3,
        already_shown_count=3,
        memories=[],
    )

    assert memory_core.format_search_result(new_result) == (
        "1. Parser tests live in tests/test_parser.py."
    )
    assert memory_core.format_search_result(zero_result) == "No matching memories found."
    assert memory_core.format_search_result(empty_result) == "No matching memories found."


def test_search_output_uses_one_budget_without_per_memory_cutoff(
    isolated_env, monkeypatch
):
    monkeypatch.setenv("NEATMEM_CODE_MAX_CONTEXT_CHARS", "1000")
    first_text = "A" * 700
    second_text = "B" * 400

    rendered = memory_core.format_search_result(
        memory_core.MemorySearchResult(
            succeeded=True,
            matched_count=2,
            already_shown_count=0,
            memories=[
                {"id": "one", "memory": first_text, "metadata": {}},
                {"id": "two", "memory": second_text, "metadata": {}},
            ],
        )
    )

    assert first_text in rendered
    assert "B" not in rendered
    assert len(rendered) <= 1000
    assert "category" not in rendered
    assert "id" not in rendered


def test_mcp_tool_contract_is_small_and_read_only():
    response = mcp_server.handle_request(
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
    )
    tool = response["result"]["tools"][0]

    assert tool["name"] == "search_memories"
    assert set(tool["inputSchema"]["properties"]) == {
        "query",
        "top_k",
        "scope",
        "run_id",
    }
    assert tool["inputSchema"]["required"] == ["query"]
    assert tool["inputSchema"]["properties"]["top_k"]["minimum"] == 1
    assert tool["inputSchema"]["properties"]["top_k"]["maximum"] == 20
    assert tool["inputSchema"]["properties"]["scope"]["enum"] == ["repo", "mine"]
    assert tool["annotations"]["readOnlyHint"] is True
    assert tool["annotations"]["idempotentHint"] is True
    assert "Claude" not in tool["description"]


def test_plugin_mcp_config_starts_the_local_server():
    config = json.loads((PLUGIN_ROOT / ".mcp.json").read_text())
    server = config["mcpServers"]["neatmem"]

    assert server == {
        "command": "python3",
        "args": ["${CLAUDE_PLUGIN_ROOT}/core/mcp_server.py"],
        "env": {"NEATMEM_CODE_DATA_DIR": "${CLAUDE_PLUGIN_DATA}"},
    }

    requests = "\n".join(
        [
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {"protocolVersion": "2024-11-05"},
                }
            ),
            json.dumps(
                {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
            ),
            "",
        ]
    )
    result = subprocess.run(
        [sys.executable, str(CORE / "mcp_server.py")],
        input=requests,
        text=True,
        capture_output=True,
        check=False,
    )
    responses = [json.loads(line) for line in result.stdout.splitlines()]

    assert result.returncode == 0
    assert result.stderr == ""
    assert responses[0]["result"]["serverInfo"]["name"] == "neatmem"
    assert responses[1]["result"]["tools"][0]["name"] == "search_memories"


def test_mcp_tool_is_session_independent_and_returns_plain_memories(
    isolated_env, monkeypatch
):
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", "/tmp/repo")
    result = memory_core.MemorySearchResult(
        succeeded=True,
        matched_count=1,
        already_shown_count=0,
        memories=[
            {
                "id": "memory-1",
                "memory": "The parser entrypoint is src/parser.py.",
                "score": 0.91,
                "categories": ["project_knowledge"],
                "metadata": {
                    "branch": "feat/parser",
                    "category": "project_knowledge",
                },
            }
        ],
    )

    with (
        patch.object(mcp_server, "resolve_repo", return_value=repo()) as resolve,
        patch.object(mcp_server, "search_memories", return_value=result) as search,
    ):
        rendered = mcp_server.call_search_memories(
            {
                "query": "parser",
                "top_k": 5,
            }
        )

    resolve.assert_called_once_with("/tmp/repo")
    assert search.call_args.args == (None, repo(), None, "parser")
    assert search.call_args.kwargs == {
        "top_k": 5,
        "scope": None,
        "run_id": None,
        "operation": "mcp-search",
    }
    assert rendered == (
        "1. The parser entrypoint is src/parser.py. "
        "[learnt on branch feat/parser]"
    )
    assert "memory-1" not in rendered
    assert "0.91" not in rendered
    assert "project_knowledge" not in rendered


@pytest.mark.parametrize("scope", ["repo", "mine"])
def test_mcp_search_passes_run_id_for_every_scope(isolated_env, monkeypatch, scope):
    from dataclasses import replace

    monkeypatch.setenv("NEATMEM_API_KEY", "test-key")
    project = replace(repo(), directory="src", project_id="code-example-hash")
    monkeypatch.setattr(mcp_server, "resolve_repo", lambda cwd: project)
    with patch.object(
        memory_core, "_request_json_with_network_retry", return_value=({"results": []}, 0, 0)
    ) as request:
        mcp_server.call_search_memories({"query": "parser", "scope": scope})
        across_sessions = request.call_args.args[2]["filters"]
        mcp_server.call_search_memories({"query": "parser", "scope": scope, "run_id": "session-42"})
        assert request.call_args.args[2]["filters"] == {**across_sessions, "run_id": "session-42"}


def test_mcp_tool_uses_codex_workspace_metadata(isolated_env):
    result = memory_core.MemorySearchResult(
        succeeded=True, matched_count=0, already_shown_count=0, memories=[]
    )
    request = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "search_memories",
            "arguments": {"query": "parser"},
            "_meta": {
                "x-codex-turn-metadata": {
                    "workspaces": {"/tmp/active-repository": {"has_changes": False}}
                }
            },
        },
    }

    with (
        patch.object(mcp_server, "resolve_repo", return_value=repo()) as resolve,
        patch.object(mcp_server, "search_memories", return_value=result),
    ):
        response = mcp_server.handle_request(request)

    resolve.assert_called_once_with("/tmp/active-repository")
    assert response["result"]["isError"] is False


@pytest.mark.parametrize(
    "arguments, message",
    [
        ({"query": ""}, "non-empty"),
        ({"query": "parser", "top_k": 0}, "1 to 20"),
        ({"query": "parser", "top_k": True}, "1 to 20"),
        ({"query": "parser", "category": "other"}, "Unknown search argument"),
        ({"query": "parser", "threshold": 0.5}, "Unknown search argument"),
        ({"query": "parser", "run_id": " "}, "run_id must be a non-empty string"),
        ({"query": "parser", "run_id": 42}, "run_id must be a non-empty string"),
    ],
)
def test_mcp_tool_rejects_invalid_arguments(arguments, message):
    with pytest.raises(mcp_server.ToolInputError, match=message):
        mcp_server._validate_arguments(arguments)


def test_search_skill_describes_memory_as_optional_starting_knowledge():
    prompt = (PLUGIN_ROOT / "skills" / "search" / "SKILL.md").read_text()
    normalized = " ".join(prompt.split())

    assert "earlier work may already explain" in normalized
    assert "avoid repeating file reads, searches, or experiments" in normalized
    assert "Call `search_memories` with the user's question" in normalized
    assert "Return the tool's result directly" in normalized
    assert "Run the search before repository exploration" not in normalized
    assert "run_id" in normalized
    assert "--run-id" in normalized
    assert "Omit `run_id` to search across sessions" in normalized


def test_control_skills_exposed():
    names = sorted(p.name for p in (PLUGIN_ROOT / "skills").iterdir() if p.is_dir())
    assert names == ["pause", "resume", "search", "status", "unpause"]


def test_status_skill_runs_cli_and_surfaces_server_failures():
    text = (PLUGIN_ROOT / "skills" / "status" / "SKILL.md").read_text()
    assert "memory_cli.py" in text and "doctor" in text
    assert "status --json" in text
    assert "unreachable" in text  # the skill must distinguish server-down from empty


def test_status_skill_checklist_matches_json_output_fields():
    text = (PLUGIN_ROOT / "skills" / "status" / "SKILL.md").read_text()
    normalized = " ".join(text.split())
    assert "pending flush packets" not in normalized  # status --json has no such field
    assert "user" in normalized.lower()
    assert "repo_id" in text
    assert "completed flushes" in normalized.lower() or "completed flush" in normalized.lower()


def test_pause_skill_points_to_dedicated_unpause_command():
    text = (PLUGIN_ROOT / "skills" / "pause" / "SKILL.md").read_text()
    assert "pause" in text
    assert "/neatmem:unpause" in text


def test_unpause_skill_runs_cli_and_is_a_dedicated_command():
    text = (PLUGIN_ROOT / "skills" / "unpause" / "SKILL.md").read_text()
    assert "disable-model-invocation: true" in text
    assert '--harness "claude-code" --plugin-data-dir "${CLAUDE_PLUGIN_DATA}" resume' in text


def test_resume_skill_briefs_from_memory_search():
    text = (PLUGIN_ROOT / "skills" / "resume" / "SKILL.md").read_text()
    assert "disable-model-invocation: true" in text
    assert "search_memories" in text


def test_status_output_explains_memory_activity_in_plain_language(capsys):
    memory_cli._print_status(
        {
            "paused": False,
            "repo_id": "https://github.com/example/repo",
            "data_dir": "/tmp/mem0-plugin-data",
            "api_key_configured": True,
            "events": 8,
            "flushes": 2,
            "retrievals": 3,
            "subagent_runs": 1,
            "last_operation": {
                "operation": "flush",
                "success": 1,
                "duration_ms": 123.4,
                "item_count": 2,
            },
            "last_subagent": {
                "stopped_at": "now",
                "context_chars": 240,
                "agent_id": "agent-1",
            },
        }
    )

    output = capsys.readouterr().out
    assert "Saved on this computer: 8 session details, 2 memory updates" in output
    assert "3 memories returned, 1 subagent runs" in output
    assert "Last memory update: succeeded" in output
    assert "Last subagent: finished, received 240 characters of memory" in output
    assert "checkpoint" not in output.lower()
    assert "injected" not in output.lower()


def test_legacy_plugin_api_key_environment_name_remains_supported(
    isolated_env, monkeypatch
):
    monkeypatch.setenv("CLAUDE_PLUGIN_OPTION_NEATMEM_API_KEY", "m0-legacy-key")

    assert memory_core.api_key() == "m0-legacy-key"


def test_user_id_defaults_to_the_local_account(isolated_env, monkeypatch):
    monkeypatch.delenv("NEATMEM_CODE_USER_ID", raising=False)
    monkeypatch.delenv("NEATMEM_USER_ID", raising=False)
    monkeypatch.delenv("NEATMEM_RESOLVED_USER_ID", raising=False)
    monkeypatch.delenv("CLAUDE_PLUGIN_OPTION_USER_ID", raising=False)
    monkeypatch.setattr(memory_core.getpass, "getuser", lambda: "local-account")

    assert memory_core.user_id() == "local-account"


def test_no_external_command_action(isolated_env):
    """The undocumented external-command hook action must not ship."""
    assert not hasattr(memory_core, "record_external_command")

    result = _run_hook(isolated_env, "external-command", {})

    assert result.returncode != 0
    assert "invalid choice" in result.stderr


def test_command_shaped_mcp_tool_captures_bounded_command_and_result():
    payload = memory_core.tool_payload(
        {
            "tool_name": "mcp__repo__exec",
            "tool_input": {"command": "python -m pytest tests/test_parser.py -q"},
            "tool_response": {
                "content": [{"type": "text", "text": "exit_code: 0\n1 passed"}]
            },
        }
    )

    assert payload["tool"] == "mcp__repo__exec"
    assert payload["command"] == "python -m pytest tests/test_parser.py -q"
    assert payload["command_kind"] == "test"
    assert "1 passed" in payload["result_preview"]


def _run_hook(
    tmp_path: Path, action: str, payload: dict, *extra: str
) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["NEATMEM_CODE_DATA_DIR"] = str(tmp_path / "hook-data")
    env["NEATMEM_CODE_SYNC_FLUSH"] = "1"
    # Keep hook subprocesses hermetic: no real server, no idle-flush workers.
    env["NEATMEM_API_URL"] = "http://127.0.0.1:1"
    env["NEATMEM_CODE_IDLE_FLUSH_SECONDS"] = "0"
    env.pop("NEATMEM_API_KEY", None)
    env.pop("CLAUDE_PLUGIN_OPTION_API_KEY", None)
    env.pop("CLAUDE_PLUGIN_OPTION_NEATMEM_API_KEY", None)
    return subprocess.run(
        [sys.executable, str(ADAPTER), action, *extra],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )


def test_offline_hook_flow_records_evidence_without_remote_writes(isolated_env):
    cwd = str(PLUGIN_ROOT)
    session_id = "hook-session"
    calls = [
        (
            "session-start",
            {"session_id": session_id, "cwd": cwd, "source": "startup"},
            (),
        ),
        (
            "user-prompt",
            {
                "session_id": session_id,
                "cwd": cwd,
                "prompt": "Fix the parser regression.",
            },
            (),
        ),
        (
            "post-tool",
            {
                "session_id": session_id,
                "cwd": cwd,
                "tool_name": "Read",
                "tool_input": {"file_path": f"{cwd}/README.md"},
                "tool_response": {"success": True},
            },
            (),
        ),
        (
            "stop",
            {
                "session_id": session_id,
                "cwd": cwd,
                "last_assistant_message": "Inspected the parser.",
            },
            (),
        ),
        ("flush", {"session_id": session_id, "cwd": cwd}, ("--reason", "session-end")),
    ]

    for action, payload, extra in calls:
        result = _run_hook(isolated_env, action, payload, *extra)
        assert result.returncode == 0
        assert result.stderr == ""
        if action == "session-start":
            # The unreachable test server makes the config pull fail; the
            # hook must say so explicitly instead of staying silent.
            notice = json.loads(result.stdout)
            context = notice["hookSpecificOutput"]["additionalContext"]
            assert "built-in defaults" in context
        if action == "flush":
            # No key gate: the flush is attempted and fails against the
            # unreachable test server; the packet stays retryable.
            assert json.loads(result.stdout)["status"] == "error"

    db_path = isolated_env / "hook-data" / "evidence.sqlite3"
    connection = sqlite3.connect(db_path)
    kinds = [
        row[0] for row in connection.execute("SELECT kind FROM events ORDER BY id")
    ]
    flush_statuses = [
        row[0] for row in connection.execute("SELECT status FROM flushes")
    ]
    connection.close()

    assert kinds == ["session_start", "user_prompt", "tool_result", "assistant_stop"]
    assert flush_statuses == ["error"]


def test_session_end_captures_final_transcript_before_stop_hook(
    isolated_env, tmp_path
):
    cwd = str(PLUGIN_ROOT)
    session_id = "session-end-before-stop"
    transcript = tmp_path / "session.jsonl"
    _write_transcript(
        transcript,
        session_id,
        [
            {
                "type": "user",
                "origin": {"kind": "human"},
                "message": {"role": "user", "content": "Inspect the parser."},
            },
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "The parser reads JSONL."}
                    ],
                },
            },
        ],
    )
    prompt = _run_hook(
        isolated_env,
        "user-prompt",
        {"session_id": session_id, "cwd": cwd, "prompt": "Inspect the parser."},
    )
    session_end = _run_hook(
        isolated_env,
        "flush",
        {
            "session_id": session_id,
            "cwd": cwd,
            "transcript_path": str(transcript),
        },
        "--reason",
        "session-end",
    )
    late_stop = _run_hook(
        isolated_env,
        "stop",
        {
            "session_id": session_id,
            "cwd": cwd,
            "transcript_path": str(transcript),
            "last_assistant_message": "The parser reads JSONL.",
        },
    )

    assert prompt.returncode == 0
    assert session_end.returncode == 0
    assert late_stop.returncode == 0
    connection = sqlite3.connect(isolated_env / "hook-data" / "evidence.sqlite3")
    rows = connection.execute(
        "SELECT payload_json FROM events WHERE kind = 'assistant_stop'"
    ).fetchall()
    connection.close()

    assert len(rows) == 1
    messages = json.loads(rows[0][0])["transcript_messages"]
    assert [message["content"] for message in messages] == [
        "Inspect the parser.",
        "Main Claude response:\nThe parser reads JSONL.",
    ]


def test_post_tool_hook_is_silent_after_recording_output(isolated_env):
    result = _run_hook(
        isolated_env,
        "post-tool",
        {
            "session_id": "s1",
            "cwd": str(PLUGIN_ROOT),
            "tool_use_id": "toolu-1",
            "tool_name": "Bash",
            "tool_input": {"command": "pytest -q"},
            "tool_response": {
                "stdout": "x" * 20000,
                "stderr": "",
                "interrupted": False,
                "isImage": False,
            },
        },
    )

    assert result.returncode == 0
    assert result.stdout == ""


def test_first_user_prompt_searches_verbatim_and_returns_five_memories(
    isolated_env, monkeypatch
):
    monkeypatch.setenv("NEATMEM_API_KEY", "m0-test-key")
    store = memory_core.EvidenceStore()
    prompt = (
        "How does the ODS serializer preserve date formatting, and where is that "
        "behavior implemented?"
    )
    results = [
        {
            "id": f"memory-{index}",
            "memory": f"Repository fact {index}.",
            "score": 1 - index / 10,
            "metadata": {"record_kind": "durable_candidate"},
        }
        for index in range(1, 7)
    ]

    import hook_runner

    with (
        patch.object(memory_core, "resolve_repo", return_value=repo()),
        patch.object(
            memory_core,
            "_request_json",
            return_value=({"results": results}, 200, 600),
        ) as request,
    ):
        output = hook_runner.prompt_memory_output(
            store,
            {"session_id": "s1", "cwd": "/tmp/repo", "prompt": prompt},
        )

    payload = request.call_args.args[2]
    assert payload["query"] == prompt
    # Embargo is active for prompt-search: over-fetch 2x to absorb filtering.
    assert payload["top_k"] == 10
    assert set(payload) == {"query", "filters", "top_k"}
    assert "systemMessage" not in output
    context = output["hookSpecificOutput"]["additionalContext"]
    assert context.startswith(
        "NeatMem found these relevant memories from earlier work in this repository:\n"
    )
    assert "Repository fact 1." in context
    assert "Repository fact 5." in context
    assert "Repository fact 6." not in context
    operation = store.conn.execute(
        "SELECT operation, item_count FROM operations"
    ).fetchone()
    assert dict(operation) == {"operation": "prompt-search", "item_count": 5}
    store.close()


def test_later_user_prompts_do_not_search_automatically(isolated_env, monkeypatch):
    monkeypatch.setenv("NEATMEM_API_KEY", "m0-test-key")
    store = memory_core.EvidenceStore()

    import hook_runner

    with (
        patch.object(memory_core, "resolve_repo", return_value=repo()),
        patch.object(
            memory_core,
            "_request_json",
            return_value=({"results": []}, 100, 20),
        ) as request,
    ):
        first = hook_runner.prompt_memory_output(
            store,
            {
                "session_id": "s1",
                "cwd": "/tmp/repo",
                "prompt": "Where is ODS date formatting implemented?",
            },
        )
        second = hook_runner.prompt_memory_output(
            store,
            {
                "session_id": "s1",
                "cwd": "/tmp/repo",
                "prompt": "Now update its tests.",
            },
        )

    assert first == {}
    assert second == {}
    assert request.call_count == 1
    store.close()


def test_manual_search_remains_available_after_automatic_search(
    isolated_env, monkeypatch
):
    monkeypatch.setenv("NEATMEM_API_KEY", "m0-test-key")
    store = memory_core.EvidenceStore()

    import hook_runner

    with (
        patch.object(memory_core, "resolve_repo", return_value=repo()),
        patch.object(
            memory_core,
            "_request_json",
            side_effect=[
                ({"results": []}, 100, 20),
                (
                    {
                        "results": [
                            {
                                "id": "manual-memory",
                                "memory": "The serializer lives in src/ods.py.",
                                "metadata": {"record_kind": "durable_candidate"},
                            }
                        ]
                    },
                    100,
                    100,
                ),
            ],
        ) as request,
    ):
        hook_runner.prompt_memory_output(
            store,
            {
                "session_id": "s1",
                "cwd": "/tmp/repo",
                "prompt": "Fix ODS date formatting in exported spreadsheets.",
            },
        )
        result = memory_core.search_memories(
            store,
            repo(),
            "s1",
            "Where is ODS date formatting implemented?",
        )

    assert [memory["id"] for memory in result.memories] == ["manual-memory"]
    assert request.call_count == 2
    # prompt-search over-fetches 2x; explicit search does not.
    assert request.call_args_list[0].args[2]["top_k"] == 10
    assert request.call_args_list[1].args[2]["top_k"] == 3
    store.close()


def test_first_prompt_is_silent_when_all_matches_were_already_provided(
    isolated_env, monkeypatch
):
    monkeypatch.setenv("NEATMEM_API_KEY", "m0-test-key")
    store = memory_core.EvidenceStore()
    existing = {
        "id": "existing-memory",
        "memory": "The parser entrypoint is src/parser.py.",
        "metadata": {"record_kind": "durable_candidate"},
    }
    store.mark_injected("s1", repo().identity, [existing])

    import hook_runner

    with (
        patch.object(memory_core, "resolve_repo", return_value=repo()),
        patch.object(
            memory_core,
            "_request_json",
            return_value=({"results": [existing]}, 100, 100),
        ),
    ):
        output = hook_runner.prompt_memory_output(
            store,
            {
                "session_id": "s1",
                "cwd": "/tmp/repo",
                "prompt": "Where is the parser implemented?",
            },
        )

    assert output == {}
    store.close()


def test_user_prompt_search_failure_still_records_evidence_and_emits_no_context(
    isolated_env,
):
    env = os.environ.copy()
    env["NEATMEM_CODE_DATA_DIR"] = str(isolated_env / "hook-data")
    env["NEATMEM_API_KEY"] = "m0-test-key"
    env["NEATMEM_API_URL"] = "http://127.0.0.1:1"
    result = subprocess.run(
        [sys.executable, str(ADAPTER), "user-prompt"],
        input=json.dumps(
            {
                "session_id": "s1",
                "cwd": str(PLUGIN_ROOT),
                "prompt": "Fix the parser regression.",
            }
        ),
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout == ""
    connection = sqlite3.connect(isolated_env / "hook-data" / "evidence.sqlite3")
    assert connection.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 1
    operation = connection.execute(
        "SELECT operation, success FROM operations"
    ).fetchone()
    assert operation == ("prompt-search", 0)
    connection.close()


def _config_server(inject_timing="every", min_query_chars=7, embargo=1800):
    """Fake GET /v1/config/ carrying a client_policy payload."""

    def fake_get_json(url, key, timeout):
        assert url.endswith("/v1/config/")
        return (
            {
                "client_policy": {
                    "inject_timing": inject_timing,
                    "min_query_chars": min_query_chars,
                    "same_session_embargo_seconds": embargo,
                }
            },
            80,
        )

    return fake_get_json


def test_fetch_client_policy_caches_the_server_values(isolated_env):
    store = memory_core.EvidenceStore()
    with patch.object(memory_core, "_get_json", side_effect=_config_server("every", 7)):
        policy = memory_core.fetch_client_policy(store)
    assert policy == {
        "inject_timing": "every",
        "min_query_chars": 7,
        "same_session_embargo_seconds": 1800,
        "source": "server",
        "error": "",
    }
    # Cached: a later read needs no HTTP.
    with patch.object(memory_core, "_get_json", side_effect=AssertionError("HTTP called")):
        assert memory_core.client_policy(store) == policy
    store.close()


def test_fetch_client_policy_falls_back_to_defaults_with_an_explicit_error(isolated_env):
    store = memory_core.EvidenceStore()
    with patch.object(
        memory_core, "_get_json", side_effect=OSError("connection refused")
    ):
        policy = memory_core.fetch_client_policy(store)
    assert policy["source"] == "default"
    assert policy["inject_timing"] == "first"
    assert policy["min_query_chars"] == 20
    assert "connection refused" in policy["error"]
    store.close()


def test_fetch_client_policy_rejects_an_unknown_server_timing(isolated_env):
    store = memory_core.EvidenceStore()
    with patch.object(
        memory_core, "_get_json", side_effect=_config_server("sometimes", 7)
    ):
        policy = memory_core.fetch_client_policy(store)
    assert policy["source"] == "server"
    assert policy["inject_timing"] == "first"
    store.close()


def test_inject_timing_prefers_the_client_env_override(isolated_env, monkeypatch):
    store = memory_core.EvidenceStore()
    with patch.object(memory_core, "_get_json", side_effect=_config_server("off", 20)):
        memory_core.fetch_client_policy(store)
    monkeypatch.setenv("NEATMEM_INJECT_TIMING", "every")
    assert memory_core.inject_timing(store) == "every"
    monkeypatch.delenv("NEATMEM_INJECT_TIMING")
    # Server policy beats the built-in default.
    assert memory_core.inject_timing(store) == "off"
    # An invalid override is ignored.
    monkeypatch.setenv("NEATMEM_INJECT_TIMING", "bogus")
    assert memory_core.inject_timing(store) == "off"
    store.close()


def test_inject_timing_off_never_searches(isolated_env, monkeypatch):
    monkeypatch.setenv("NEATMEM_INJECT_TIMING", "off")
    store = memory_core.EvidenceStore()

    import hook_runner

    with (
        patch.object(memory_core, "resolve_repo", return_value=repo()),
        patch.object(
            memory_core, "_request_json", side_effect=AssertionError("search called")
        ) as request,
    ):
        for prompt in ("Where is the parser implemented?", "And its tests?"):
            output = hook_runner.prompt_memory_output(
                store, {"session_id": "s1", "cwd": "/tmp/repo", "prompt": prompt}
            )
            assert output == {}
    assert request.call_count == 0
    # Prompts are still recorded as evidence.
    assert store.conn.execute(
        "SELECT COUNT(*) FROM events WHERE kind = 'user_prompt'"
    ).fetchone()[0] == 2
    store.close()


def test_inject_timing_every_searches_each_prompt(isolated_env, monkeypatch):
    monkeypatch.setenv("NEATMEM_INJECT_TIMING", "every")
    store = memory_core.EvidenceStore()

    import hook_runner

    with (
        patch.object(memory_core, "resolve_repo", return_value=repo()),
        patch.object(
            memory_core, "_request_json", return_value=({"results": []}, 100, 20)
        ) as request,
    ):
        for prompt in ("Where is the parser implemented?", "And where are its tests kept?"):
            hook_runner.prompt_memory_output(
                store, {"session_id": "s1", "cwd": "/tmp/repo", "prompt": prompt}
            )
    assert request.call_count == 2
    store.close()


def test_min_query_chars_comes_from_the_server_policy(isolated_env):
    store = memory_core.EvidenceStore()
    with patch.object(memory_core, "_get_json", side_effect=_config_server("every", 30)):
        memory_core.fetch_client_policy(store)

    import hook_runner

    with (
        patch.object(memory_core, "resolve_repo", return_value=repo()),
        patch.object(
            memory_core, "_request_json", return_value=({"results": []}, 100, 20)
        ) as request,
    ):
        # 25 chars: above the built-in 20, below the server's 30.
        output = hook_runner.prompt_memory_output(
            store,
            {"session_id": "s1", "cwd": "/tmp/repo", "prompt": "Fix the parser regression"},
        )
    assert output == {}
    assert request.call_count == 0
    store.close()


def test_neatmem_enabled_zero_makes_every_hook_a_noop(isolated_env):
    env = os.environ.copy()
    env["NEATMEM_CODE_DATA_DIR"] = str(isolated_env / "hook-data")
    env["NEATMEM_ENABLED"] = "0"
    for action, payload in [
        ("session-start", {"session_id": "s1", "cwd": str(PLUGIN_ROOT)}),
        (
            "user-prompt",
            {
                "session_id": "s1",
                "cwd": str(PLUGIN_ROOT),
                "prompt": "Fix the parser regression.",
            },
        ),
        (
            "stop",
            {
                "session_id": "s1",
                "cwd": str(PLUGIN_ROOT),
                "last_assistant_message": "Done.",
            },
        ),
    ]:
        result = subprocess.run(
            [sys.executable, str(ADAPTER), action],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            env=env,
            check=False,
        )
        assert result.returncode == 0
        assert result.stdout == ""
    assert not (isolated_env / "hook-data" / "evidence.sqlite3").exists()


def test_session_start_with_reachable_server_caches_policy_without_a_notice(isolated_env):
    import http.server
    import threading

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps(
                {"client_policy": {"inject_timing": "off", "min_query_chars": 20}}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        env = os.environ.copy()
        env["NEATMEM_CODE_DATA_DIR"] = str(isolated_env / "hook-data")
        env["NEATMEM_API_URL"] = f"http://127.0.0.1:{server.server_port}"
        result = subprocess.run(
            [sys.executable, str(ADAPTER), "session-start"],
            input=json.dumps({"session_id": "s1", "cwd": str(PLUGIN_ROOT)}),
            text=True,
            capture_output=True,
            env=env,
            check=False,
        )
        assert result.returncode == 0
        assert result.stdout == ""
        connection = sqlite3.connect(isolated_env / "hook-data" / "evidence.sqlite3")
        cached = json.loads(
            connection.execute(
                "SELECT value FROM settings WHERE key = 'client_policy'"
            ).fetchone()[0]
        )
        connection.close()
        assert cached["inject_timing"] == "off"
        assert cached["source"] == "server"
    finally:
        server.shutdown()



def test_search_skill_wraps_native_tool_for_user_invocation():
    skill = (PLUGIN_ROOT / "skills" / "search" / "SKILL.md").read_text()

    assert "memories from earlier Claude Code sessions" in skill
    assert "avoid repeating file reads, searches, or experiments" in skill
    assert "disable-model-invocation: true" in skill
    assert "Call `search_memories`" in skill
    assert "```bash" not in skill
    assert "memory_cli.py" not in skill
    assert "--top-k" in skill
    assert "--scope" in skill
    assert "--category" not in skill


def test_plugin_entrypoints_share_explicit_claude_data_dir(tmp_path, monkeypatch):
    canonical_data = tmp_path / "claude-plugin-data"
    conflicting_data = tmp_path / "benchmark-override"
    monkeypatch.setenv("NEATMEM_CODE_DATA_DIR", str(canonical_data))
    monkeypatch.setenv("NEATMEM_CODE_USER_ID", "test-user")
    monkeypatch.setenv("NEATMEM_API_KEY", "m0-test-key")
    repo_context = memory_core.resolve_repo(str(PLUGIN_ROOT))
    store = memory_core.EvidenceStore()
    with patch.object(
        memory_core,
        "_request_json",
        return_value=(
            {
                "results": [
                    {
                        "id": "memory-1",
                        "memory": "Mem0 stores plugin evidence in one database.",
                        "score": 0.91,
                        "metadata": {"record_kind": "durable_candidate"},
                    }
                ]
            },
            100,
            120,
        ),
    ):
        result = memory_core.search_memories(
            store,
            repo_context,
            "shared-session",
            "Where does Mem0 store the plugin evidence database?",
        )
    assert len(result.memories) == 1
    store.close()

    env = os.environ.copy()
    env["NEATMEM_CODE_DATA_DIR"] = str(conflicting_data)
    status = subprocess.run(
        [
            sys.executable,
            str(CORE / "memory_cli.py"),
            "--plugin-data-dir",
            str(canonical_data),
            "status",
            "--json",
        ],
        cwd=PLUGIN_ROOT,
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )
    assert status.returncode == 0
    status_payload = json.loads(status.stdout)
    assert status_payload["data_dir"] == str(canonical_data)
    assert status_payload["retrievals"] == 1
    assert status_payload["subagent_runs"] == 0
    assert not (conflicting_data / "evidence.sqlite3").exists()


def test_paused_hook_does_not_capture(isolated_env):
    data_path = isolated_env / "hook-data" / "evidence.sqlite3"
    store = memory_core.EvidenceStore(data_path)
    store.set_setting("paused", "true")
    store.close()

    result = _run_hook(
        isolated_env,
        "user-prompt",
        {
            "session_id": "paused-session",
            "cwd": str(PLUGIN_ROOT),
            "prompt": "This should not be captured.",
        },
    )

    assert result.returncode == 0
    connection = sqlite3.connect(data_path)
    assert connection.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0
    connection.close()


def test_version_is_single_sourced():
    manifest = json.loads((PLUGIN_ROOT / ".claude-plugin" / "plugin.json").read_text())
    assert manifest["version"] == memory_core.PLUGIN_VERSION
    response = mcp_server.handle_request({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
    assert response["result"]["serverInfo"]["version"] == memory_core.PLUGIN_VERSION


def test_user_id_falls_back_to_default_only_when_the_os_account_is_unknown(monkeypatch):
    for name in (
        "CLAUDE_PLUGIN_OPTION_USER_ID",
        "NEATMEM_CODE_USER_ID",
        "NEATMEM_USER_ID",
        "NEATMEM_RESOLVED_USER_ID",
    ):
        monkeypatch.delenv(name, raising=False)

    def raise_os_error():
        raise OSError("no password database")

    monkeypatch.setattr(memory_core.getpass, "getuser", raise_os_error)
    assert memory_core.user_id() == "default"
    monkeypatch.setattr(memory_core.getpass, "getuser", lambda: "windows-account")
    assert memory_core.user_id() == "windows-account"


def test_user_id_uses_the_os_account_when_the_hook_env_is_empty(monkeypatch):
    """CC hook subprocesses have no USER/USERNAME; resolve the account anyway."""
    for name in (
        "CLAUDE_PLUGIN_OPTION_USER_ID",
        "PLUGIN_OPTION_USER_ID",
        "NEATMEM_CODE_USER_ID",
        "NEATMEM_USER_ID",
        "NEATMEM_RESOLVED_USER_ID",
        "USER",
        "USERNAME",
        "LOGNAME",
        "LNAME",
    ):
        monkeypatch.delenv(name, raising=False)

    import pwd

    assert memory_core.user_id() == pwd.getpwuid(os.getuid()).pw_name


def test_transcript_rows_resume_from_a_byte_offset(tmp_path):
    transcript = tmp_path / "session.jsonl"
    _write_transcript(
        transcript,
        "s1",
        [
            {
                "type": "user",
                "origin": {"kind": "human"},
                "message": {"role": "user", "content": "Inspect the parser."},
            },
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "The parser reads JSONL."}],
                },
            },
        ],
    )

    rows, end, resumed = transcript_mod._transcript_rows(str(transcript))
    assert [row["uuid"] for row in rows] == ["entry-1", "entry-2"]
    assert end == transcript.stat().st_size
    assert resumed is False

    first_size = end
    third_row = {
        "uuid": "entry-3",
        "parentUuid": "entry-2",
        "sessionId": "s1",
        "isSidechain": False,
        "type": "user",
        "message": {"role": "user", "content": "Where is that implemented?"},
    }
    with transcript.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(third_row) + "\n")
        handle.write('{"uuid": "partial-row"')

    rows, end, resumed = transcript_mod._transcript_rows(str(transcript), first_size)
    assert [row["uuid"] for row in rows] == ["entry-3"]
    assert resumed is True
    assert end < transcript.stat().st_size

    rows, _, resumed = transcript_mod._transcript_rows(
        str(transcript), transcript.stat().st_size + 100
    )
    assert [row["uuid"] for row in rows] == ["entry-1", "entry-2", "entry-3"]
    assert resumed is False


def test_record_stop_reads_the_transcript_from_the_stored_offset(
    isolated_env, tmp_path
):
    transcript = tmp_path / "session.jsonl"
    first_entries = [
        {
            "type": "user",
            "origin": {"kind": "human"},
            "message": {"role": "user", "content": "Inspect the parser."},
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "The parser reads JSONL."}],
            },
        },
    ]
    _write_transcript(transcript, "s1", first_entries)
    store = memory_core.EvidenceStore()

    with patch.object(memory_core, "resolve_repo", return_value=repo()):
        transcript_mod.record_stop(
            store,
            {
                "session_id": "s1",
                "cwd": "/tmp/repo",
                "transcript_path": str(transcript),
                "last_assistant_message": "The parser reads JSONL.",
            },
        )

    first_size = transcript.stat().st_size
    payload = json.loads(
        store.conn.execute(
            "SELECT payload_json FROM events WHERE kind = 'assistant_stop' "
            "ORDER BY id DESC LIMIT 1"
        ).fetchone()["payload_json"]
    )
    assert payload["transcript_path"] == str(transcript)
    assert payload["transcript_offset"] == first_size

    second_entries = first_entries + [
        {
            "type": "user",
            "origin": {"kind": "human"},
            "message": {"role": "user", "content": "Where is that implemented?"},
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "In memory_core.py."}],
            },
        },
    ]
    _write_transcript(transcript, "s1", second_entries)
    original_rows = transcript_mod._transcript_rows
    offsets = []

    def spying_rows(path, offset=0):
        offsets.append(offset)
        return original_rows(path, offset)

    with (
        patch.object(memory_core, "resolve_repo", return_value=repo()),
        patch.object(transcript_mod, "_transcript_rows", side_effect=spying_rows),
    ):
        transcript_mod.record_stop(
            store,
            {
                "session_id": "s1",
                "cwd": "/tmp/repo",
                "transcript_path": str(transcript),
                "last_assistant_message": "In memory_core.py.",
            },
        )
    store.close()

    assert offsets == [first_size]


def test_flush_gives_up_after_repeated_failures_and_later_events_still_flush(
    isolated_env, monkeypatch
):
    store = memory_core.EvidenceStore()
    _record_complete_session(store)
    hook_input = {"session_id": "s1", "cwd": "/tmp/repo"}

    with (
        patch.object(memory_core, "resolve_repo", return_value=repo()),
        patch.object(memory_core, "_request_json", side_effect=OSError("api down")),
    ):
        for _ in range(memory_core.MAX_FLUSH_ATTEMPTS):
            failed = memory_core.flush_session(store, hook_input, "session-end")
            assert failed["status"] == "error"
        given_up = memory_core.flush_session(store, hook_input, "session-end")

    record = store.conn.execute("SELECT status, attempts FROM flushes").fetchone()
    assert given_up["status"] == "nothing-to-flush"
    assert record["status"] == "gave-up"
    assert record["attempts"] == memory_core.MAX_FLUSH_ATTEMPTS

    _record_exchange(store, 99)
    with (
        patch.object(memory_core, "resolve_repo", return_value=repo()),
        patch.object(memory_core, "_request_json", side_effect=_flush_server(1)),
    ):
        later = memory_core.flush_session(store, hook_input, "session-end")

    assert later["status"] == "succeeded"
    assert store.conn.execute("SELECT COUNT(*) FROM flushes").fetchone()[0] == 2
    store.close()


def test_flush_touches_the_worker_heartbeat(isolated_env, monkeypatch, tmp_path):
    handoff = tmp_path / "packet.running"
    handoff.write_text("{}", encoding="utf-8")
    os.utime(handoff, (1000, 1000))
    monkeypatch.setenv("NEATMEM_CODE_HANDOFF_PATH", str(handoff))
    store = memory_core.EvidenceStore()
    _record_complete_session(store)

    with (
        patch.object(memory_core, "resolve_repo", return_value=repo()),
        patch.object(memory_core, "_request_json", side_effect=_flush_server(1)),
    ):
        result = memory_core.flush_session(
            store, {"session_id": "s1", "cwd": "/tmp/repo"}, "session-end"
        )

    assert result["status"] == "succeeded"
    assert handoff.stat().st_mtime > 1000

    monkeypatch.delenv("NEATMEM_CODE_HANDOFF_PATH")
    memory_core.touch_handoff_heartbeat()
    store.close()


def test_paused_session_start_keeps_pending_packets_fresh(isolated_env):
    import hook_runner

    pending_dir = Path(os.environ["NEATMEM_CODE_DATA_DIR"]) / "pending"
    pending_dir.mkdir(parents=True, exist_ok=True)
    held = pending_dir / "held.json"
    held.write_text("{}", encoding="utf-8")
    claimed = pending_dir / "claimed.running"
    claimed.write_text("{}", encoding="utf-8")
    stale_time = time.time() - hook_runner.PENDING_EXPIRY_SECONDS - 60
    for path in (held, claimed):
        os.utime(path, (stale_time, stale_time))

    hook_runner.refresh_pending_handoffs()

    assert held.stat().st_mtime > time.time() - 60
    assert claimed.stat().st_mtime > time.time() - 60


def test_corrupt_database_is_quarantined_and_capture_restarts(isolated_env):
    database = Path(os.environ["NEATMEM_CODE_DATA_DIR"]) / "evidence.sqlite3"
    database.parent.mkdir(parents=True, exist_ok=True)
    database.write_text("this is not a sqlite database", encoding="utf-8")

    store = memory_core.EvidenceStore()
    store.record_event(repo(), "s1", "user_prompt", {"text": "Still capturing."})
    count = store.conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    store.close()

    quarantined = list(database.parent.glob("evidence.sqlite3.corrupt-*"))
    assert count == 1
    assert len(quarantined) == 1
    assert quarantined[0].read_text(encoding="utf-8") == "this is not a sqlite database"


def test_stale_cached_api_key_is_cleared_when_config_is_removed(
    isolated_env, monkeypatch
):
    monkeypatch.setenv("CLAUDE_PLUGIN_OPTION_API_KEY", "m0-cached-key")
    assert memory_core.cache_plugin_api_key() is True
    assert memory_core.clear_stale_api_key_cache() is False
    assert memory_core.api_key() == "m0-cached-key"

    monkeypatch.delenv("CLAUDE_PLUGIN_OPTION_API_KEY")
    assert memory_core.clear_stale_api_key_cache() is True
    assert memory_core.api_key() == ""
    assert memory_core.clear_stale_api_key_cache() is False


def test_resolve_repo_caches_git_lookups_in_process(isolated_env, monkeypatch):
    calls = []

    def fake_git(cwd, *args):
        calls.append(args)
        return {
            ("rev-parse", "--show-toplevel"): "/tmp/repo",
            ("config", "--get", "remote.origin.url"): "https://github.com/example/repo.git",
            ("branch", "--show-current"): "main",
            ("rev-parse", "HEAD"): "abc123",
        }.get(args, "")

    monkeypatch.setattr(memory_core, "_git", fake_git)
    memory_core._resolve_repo_cached.cache_clear()

    first = memory_core.resolve_repo("/tmp/repo")
    calls_after_first = len(calls)
    second = memory_core.resolve_repo("/tmp/repo")

    assert first == second
    assert calls_after_first > 0
    assert len(calls) == calls_after_first

    memory_core._resolve_repo_cached.cache_clear()
    memory_core.resolve_repo("/tmp/repo")
    assert len(calls) > calls_after_first


def test_resolve_repo_keeps_the_directory_when_cwd_is_reached_through_a_symlink(isolated_env, tmp_path):
    real_root = tmp_path / "real"
    (real_root / "services" / "billing").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(real_root)], check=True)
    link = tmp_path / "link"
    link.symlink_to(real_root)
    memory_core._resolve_repo_cached.cache_clear()

    repo = memory_core.resolve_repo(str(link / "services" / "billing"))

    assert repo.directory == "services/billing"
    assert repo.root == str(real_root.resolve())


SCOPE_ENV_VARS = (
    "CLAUDE_PLUGIN_OPTION_USER_ID",
    "NEATMEM_CODE_USER_ID",
    "NEATMEM_USER_ID",
    "NEATMEM_RESOLVED_USER_ID",
    "USER",
    "USERNAME",
)


@pytest.mark.parametrize("wildcard", ["*", "**", " * "])
def test_a_wildcard_user_id_never_becomes_the_scope(monkeypatch, wildcard):
    for name in SCOPE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    monkeypatch.setenv("NEATMEM_CODE_USER_ID", wildcard)
    monkeypatch.setattr(memory_core.getpass, "getuser", lambda: "os-account")
    assert memory_core.user_id() == "os-account"


@pytest.mark.parametrize("wildcard", ["*", "**", " * "])
def test_a_wildcard_user_id_falls_through_to_the_account_env(monkeypatch, wildcard):
    for name in SCOPE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    monkeypatch.setenv("NEATMEM_CODE_USER_ID", wildcard)
    monkeypatch.setenv("USER", "real-account")
    assert memory_core.user_id() == "real-account"


def test_a_wildcard_project_id_falls_through_to_the_git_remote(monkeypatch, tmp_path):
    monkeypatch.setenv("NEATMEM_PROJECT_ID", "*")
    resolved = memory_core._legacy_project_id(
        str(tmp_path), str(tmp_path), "https://github.com/mem0ai/mem0.git", "mem0ai/mem0"
    )
    assert resolved == "mem0ai-mem0"


def test_forget_refuses_an_unscoped_delete(monkeypatch, isolated_env):
    monkeypatch.setenv("NEATMEM_API_KEY", "test-key")
    monkeypatch.setenv("NEATMEM_CODE_USER_ID", "*")
    repo = memory_core.RepoContext(
        cwd="/x", root="/x", identity="x", app_id="*", branch="main", head_sha="abc", project_id="*"
    )
    with patch.object(memory_core, "_request_json") as request:
        result = memory_core.forget_remote_repo(repo)
    request.assert_not_called()
    assert result["status"] == "error"
    assert "wildcard" in result["error"]


def test_forget_deletes_each_memory_by_id(monkeypatch, isolated_env):
    monkeypatch.setenv("NEATMEM_API_KEY", "test-key")
    repo = memory_core.RepoContext(
        cwd="/x", root="/x", identity="x", app_id="repo-a", branch="main", head_sha="abc", project_id="repo-a"
    )
    listed = {"results": [{"id": "m1", "app_id": "repo-a"}, {"id": "m2", "app_id": "repo-a/sub"}]}
    deleted = []

    with patch.object(memory_core, "_request_json", return_value=(listed, 0, 0)) as request:
        with patch.object(
            memory_core, "_delete_memory", side_effect=lambda *args: deleted.append(args[2]) is None
        ):
            result = memory_core.forget_remote_repo(repo)

    url, _, payload, _ = request.call_args[0]
    assert "/v2/memories/" in url
    assert payload["filters"] == {
        "user_id": "test-user", "agent_id": "repo-a"
    }
    assert deleted == ["m1", "m2"]
    assert result == {"status": "deleted", "deleted": 2}


def test_forget_only_touches_shared_project_memory_when_asked(monkeypatch, isolated_env):
    monkeypatch.setenv("NEATMEM_API_KEY", "test-key")
    repo = memory_core.RepoContext(
        cwd="/x", root="/x", identity="x", app_id="repo-a", branch="main", head_sha="abc", project_id="repo-a"
    )
    listed = [({"results": []}, 0, 0), ({"results": [{"id": "shared-1", "app_id": "repo-a"}]}, 0, 0)]
    with patch.object(memory_core, "_request_json", side_effect=listed) as request:
        with patch.object(memory_core, "_delete_memory", return_value=True):
            result = memory_core.forget_remote_repo(repo, include_project_memory=True)

    assert [call.args[2]["filters"] for call in request.call_args_list] == [
        {"user_id": "test-user", "agent_id": "repo-a"},
        {"agent_id": "repo-a"},
    ]
    assert result == {"status": "deleted", "deleted": 1}


@pytest.mark.parametrize("include_project", [False, True])
def test_forget_includes_shared_project_memories_only_when_requested(monkeypatch, isolated_env, include_project):
    from dataclasses import replace

    monkeypatch.setenv("NEATMEM_API_KEY", "test-key")
    project = replace(repo(), project_id="code-example-hash")
    user_filter = {"user_id": "test-user", "agent_id": project.project_id}
    project_filter = {"agent_id": project.project_id}
    listed = {
        json.dumps(user_filter, sort_keys=True): [{"id": "personal"}],
        json.dumps(project_filter, sort_keys=True): [{"id": "current"}],
    }

    def request(url, key, payload, timeout):
        return {"results": listed.get(json.dumps(payload["filters"], sort_keys=True), [])}, 0, 0

    with (
        patch.object(memory_core, "_request_json", side_effect=request),
        patch.object(memory_core, "_delete_memory", return_value=True) as delete,
    ):
        result = memory_core.forget_remote_repo(project, include_project_memory=include_project)

    expected = {"personal", "current"} if include_project else {"personal"}
    assert {call.args[2] for call in delete.call_args_list} == expected
    assert result == {"status": "deleted", "deleted": len(expected)}


def test_forget_reports_partial_failures(monkeypatch, isolated_env):
    monkeypatch.setenv("NEATMEM_API_KEY", "test-key")
    repo = memory_core.RepoContext(
        cwd="/x", root="/x", identity="x", app_id="repo-a", branch="main", head_sha="abc", project_id="repo-a"
    )
    listed = {"results": [{"id": "m1", "app_id": "repo-a"}, {"id": "m2", "app_id": "repo-a"}]}

    with patch.object(memory_core, "_request_json", return_value=(listed, 0, 0)):
        with patch.object(memory_core, "_delete_memory", side_effect=[True, False]):
            result = memory_core.forget_remote_repo(repo)

    assert result["status"] == "partial"
    assert result["deleted"] == 1
    assert result["failed"] == 1


def test_an_empty_session_is_never_posted_to_the_api(isolated_env, monkeypatch):
    monkeypatch.setenv("NEATMEM_API_KEY", "m0-test-key")
    store = memory_core.EvidenceStore()
    _record_complete_session(store)

    with (
        patch.object(memory_core, "resolve_repo", return_value=repo()),
        patch.object(memory_core, "build_extraction_messages", return_value=[]),
        patch.object(memory_core, "_request_json") as request,
    ):
        result = memory_core.flush_session(
            store, {"session_id": "s1", "cwd": "/tmp/repo"}, "session-end"
        )

    request.assert_not_called()
    assert result["status"] == "nothing-to-flush"
    store.close()


def test_search_refuses_a_wildcard_repository_scope(isolated_env, monkeypatch):
    monkeypatch.setenv("NEATMEM_API_KEY", "m0-test-key")
    repo = memory_core.RepoContext(
        cwd="/x", root="/x", identity="x", app_id="*", branch="main", head_sha="abc"
    )

    with patch.object(memory_core, "_request_json") as request:
        result = memory_core.search_memories(None, repo, None, "anything")

    request.assert_not_called()
    assert result.succeeded is False


def test_flush_refuses_to_write_without_a_session_id(isolated_env, monkeypatch):
    monkeypatch.setenv("NEATMEM_API_KEY", "m0-test-key")
    store = memory_core.EvidenceStore()
    _record_complete_session(store)

    with (
        patch.object(memory_core, "resolve_repo", return_value=repo()),
        patch.object(memory_core, "_request_json") as request,
    ):
        result = memory_core.flush_session(store, {"cwd": "/tmp/repo"}, "session-end")

    request.assert_not_called()
    assert result == {"status": "error", "reason": "no-session-id"}
    store.close()


def test_flush_refuses_a_wildcard_repository_scope(isolated_env, monkeypatch):
    monkeypatch.setenv("NEATMEM_API_KEY", "m0-test-key")
    store = memory_core.EvidenceStore()
    _record_complete_session(store)

    with (
        patch.object(memory_core, "resolve_repo", return_value=repo()),
        patch.object(memory_core, "user_id", return_value="*"),
        patch.object(memory_core, "_request_json") as request,
    ):
        result = memory_core.flush_session(
            store, {"session_id": "s1", "cwd": "/tmp/repo"}, "session-end"
        )

    request.assert_not_called()
    assert result == {"status": "error", "reason": "wildcard-scope"}
    store.close()


def _payments(directory=""):
    return memory_core.RepoContext(
        cwd="/x/" + directory if directory else "/x",
        root="/x",
        identity="https://github.com/acme/payments-api",
        app_id="payments-api",
        branch="main",
        head_sha="s",
        project_id="payments-api",
        directory=directory,
    )


def test_search_filters_repo_scope_is_the_shared_project():
    assert memory_core._search_filters("priya", _payments("services/billing"), "repo") == {
        "agent_id": "payments-api"
    }


def test_search_filters_mine_scope_narrows_the_project_to_the_user():
    assert memory_core._search_filters("priya", _payments(), "mine") == {
        "user_id": "priya", "agent_id": "payments-api"
    }


def test_directory_app_id_is_the_repository_at_the_root_and_nested_below():
    assert memory_core.directory_app_id(_payments()) == "payments-api"
    assert memory_core.directory_app_id(_payments("services/billing")) == "payments-api/services/billing"


def test_no_search_scope_wildcards_the_user():
    for scope in memory_core.SEARCH_SCOPES:
        assert '"user_id": "*"' not in json.dumps(
            memory_core._search_filters("priya", _payments("apps/web"), scope)
        ), scope


def _flatten_filter_values(node: dict) -> dict:
    result = {}
    for key, value in node.items():
        if key in ("AND", "OR") and isinstance(value, list):
            for child in value:
                if isinstance(child, dict):
                    result.update(_flatten_filter_values(child))
        else:
            result[key] = value
    return result


def test_every_search_scope_pins_an_owned_identity():
    for scope in memory_core.SEARCH_SCOPES:
        for directory in ("", "apps/web"):
            filters = memory_core._search_filters("priya", _payments(directory), scope)
            values = _flatten_filter_values(filters)
            assert values.get("user_id") == "priya" or values.get("agent_id") == "payments-api", (
                f"{scope} has an unpinned branch: {filters}"
            )


def test_search_scope_env_override_and_fallback(monkeypatch):
    monkeypatch.setenv("NEATMEM_CODE_SEARCH_SCOPE", "mine")
    assert memory_core.search_scope() == "mine"
    monkeypatch.setenv("NEATMEM_CODE_SEARCH_SCOPE", "team")
    assert memory_core.search_scope() == "repo"
    monkeypatch.setenv("NEATMEM_CODE_SEARCH_SCOPE", "all")
    assert memory_core.search_scope() == "repo"
    monkeypatch.delenv("NEATMEM_CODE_SEARCH_SCOPE")
    assert memory_core.search_scope() == "repo"


def test_search_memories_rejects_unknown_scope(monkeypatch):
    monkeypatch.setenv("NEATMEM_API_KEY", "test-key")
    monkeypatch.setenv("NEATMEM_CODE_USER_ID", "priya")
    repo = _payments()
    with pytest.raises(ValueError, match="Unknown search scope"):
        memory_core.search_memories(None, repo, None, "q", scope="everything")


def test_search_memories_sends_repo_filters_by_default(monkeypatch):
    monkeypatch.setenv("NEATMEM_API_KEY", "test-key")
    monkeypatch.setenv("NEATMEM_CODE_USER_ID", "priya")
    sent = {}

    def fake_request(url, key, payload, timeout):
        sent.update(payload)
        return {"results": []}, 0, 0

    monkeypatch.setattr(
        memory_core, "_request_json_with_network_retry", fake_request
    )
    memory_core.search_memories(None, _payments(), None, "q")
    assert sent["filters"] == {"agent_id": "payments-api"}


def _record_failed_then_recovered_command(store, session_id="s1"):
    r = repo()
    store.record_event(
        r,
        session_id,
        "tool_failure",
        {
            "tool": "Bash",
            "command": "pytest tests/",
            "command_kind": "test",
            "failed": True,
            "result_preview": "error: NEATMEM_API_KEY is not set",
        },
    )
    store.record_event(
        r,
        session_id,
        "tool_result",
        {
            "tool": "Bash",
            "command": "NEATMEM_API_KEY=x pytest tests/",
            "command_kind": "test",
            "failed": False,
            "result_preview": "136 passed",
        },
    )


def test_flush_sends_unified_body_with_both_agent_and_user_id(isolated_env, monkeypatch):
    store = memory_core.EvidenceStore()
    _record_complete_session(store)

    with (
        patch.object(memory_core, "resolve_repo", return_value=repo()),
        patch.object(
            memory_core, "_request_json", side_effect=_flush_server(2)
        ) as request,
    ):
        memory_core.flush_session(
            store, {"session_id": "s1", "cwd": "/tmp/repo"}, "session-end"
        )

    (sent_body,) = _add_bodies(request)
    assert set(sent_body) == {"messages", "user_id", "agent_id", "run_id"}
    assert sent_body["agent_id"] == "code-example"
    assert sent_body["user_id"] == "test-user"
    assert sent_body["run_id"] == "s1"
    store.close()


def test_a_failed_command_appears_in_extraction_evidence(isolated_env, monkeypatch):
    store = memory_core.EvidenceStore()
    _record_complete_session(store)
    _record_failed_then_recovered_command(store)

    with (
        patch.object(memory_core, "resolve_repo", return_value=repo()),
        patch.object(
            memory_core, "_request_json", side_effect=_flush_server(2)
        ) as request,
    ):
        result = memory_core.flush_session(
            store, {"session_id": "s1", "cwd": "/tmp/repo"}, "session-end"
        )

    assert result["status"] == "succeeded"
    (sent_body,) = _add_bodies(request)
    assert sent_body["agent_id"] == "code-example"
    assert sent_body["user_id"] == "test-user"
    assert "Commands run in this session:" in sent_body["messages"][-1]["content"]
    assert "[failed/test]" in sent_body["messages"][-1]["content"]
    store.close()


def test_a_folder_without_a_remote_shares_project_memory_under_a_path_hashed_namespace(isolated_env, monkeypatch):
    monkeypatch.setenv("NEATMEM_USER_ID", "maya")
    store = memory_core.EvidenceStore()
    folder = _local_folder()
    _record_complete_session(store, repo_context=folder)

    with (
        patch.object(memory_core, "resolve_repo", return_value=folder),
        patch.object(
            memory_core, "_request_json", side_effect=_flush_server(2)
        ) as request,
    ):
        memory_core.flush_session(
            store, {"session_id": "s1", "cwd": folder.cwd}, "session-end"
        )

    (sent_body,) = _add_bodies(request)
    assert sent_body["agent_id"] == folder.project_id
    assert sent_body["agent_id"].startswith("local-marketing-")
    assert sent_body["user_id"] == "maya"
    store.close()


def test_same_named_folders_at_different_paths_get_different_namespaces():
    a = memory_core._project_id("/home/maya/marketing", "local:/home/maya/marketing", "marketing")
    b = memory_core._project_id("/home/raj/marketing", "local:/home/raj/marketing", "marketing")
    assert a != b
    assert a.startswith("local-marketing-") and b.startswith("local-marketing-")
    remote_id = memory_core._project_id("/x", "https://github.com/acme/api", "acme-api")
    assert remote_id.startswith("acme-api-") and remote_id != "acme-api"


def test_relative_directory_is_empty_at_the_root_and_posix_below():
    assert memory_core._relative_directory("/x", "/x") == ""
    assert memory_core._relative_directory("/x/apps/web", "/x") == "apps/web"
    assert memory_core._relative_directory("/elsewhere", "/x") == ""


def test_results_are_ranked_by_score_alone():
    results = [
        {"id": "a", "memory": "x", "score": 0.9, "metadata": {"lane": "project"}},
        {"id": "b", "memory": "y", "score": 0.8, "metadata": {"lane": "project"}},
        {"id": "c", "memory": "z", "score": 0.7, "metadata": {"lane": "personal"}},
    ]
    with patch.object(
        memory_core,
        "_request_json_with_network_retry",
        return_value=({"results": results}, 0, 0),
    ), patch.dict(os.environ, {"NEATMEM_API_KEY": "test-key"}):
        found = memory_core.search_memories(None, repo(), None, "q", top_k=2)

    assert [m["id"] for m in found.memories] == ["a", "b"]


def test_branch_is_labelled_in_injected_context():
    context = memory_core.format_context(
        [
            {"memory": "The parser lives in src/ods.py.", "metadata": {"lane": "project"}},
            {
                "memory": "pytest needs NEATMEM_API_KEY set.",
                "metadata": {"lane": "project", "branch": "feature/x"},
            },
        ]
    )

    assert "1. The parser lives in src/ods.py." in context
    assert "2. pytest needs NEATMEM_API_KEY set. [learnt on branch feature/x]" in context


def _search_payload(monkeypatch, repo, **kwargs):
    sent = {}

    def fake_request(url, key, payload, timeout):
        sent.update(payload)
        return {"results": []}, 0, 0

    monkeypatch.setattr(memory_core, "_request_json_with_network_retry", fake_request)
    memory_core.search_memories(None, repo, None, "q", **kwargs)
    return sent


def _local_folder():
    return memory_core.RepoContext(
        cwd="/home/maya/marketing",
        root="/home/maya/marketing",
        identity="local:/home/maya/marketing",
        app_id="marketing",
        branch="",
        head_sha="",
        project_id="local-marketing-abcdef0123",
    )


def test_search_scope_resolves_to_the_configured_default(monkeypatch):
    monkeypatch.setenv("NEATMEM_API_KEY", "test-key")
    monkeypatch.setenv("NEATMEM_CODE_USER_ID", "maya")

    assert memory_core.resolve_search_scope(None) == "repo"
    assert memory_core.resolve_search_scope(" MINE ") == "mine"
    # The dir scope is gone; a stale configured value falls back to the default.
    monkeypatch.setenv("NEATMEM_CODE_SEARCH_SCOPE", "dir")
    assert memory_core.resolve_search_scope(None) == "repo"

    for scope in memory_core.SEARCH_SCOPES:
        sent = _search_payload(monkeypatch, _local_folder(), scope=scope)
        assert '"user_id": "*"' not in json.dumps(sent["filters"]), scope


def test_run_id_narrows_the_search_to_one_session(monkeypatch):
    monkeypatch.setenv("NEATMEM_API_KEY", "test-key")
    monkeypatch.setenv("NEATMEM_CODE_USER_ID", "priya")

    sent = _search_payload(monkeypatch, repo(), scope="repo", run_id="session-42")

    assert sent["filters"] == {
        "agent_id": "code-example",
        "run_id": "session-42",
    }


def test_all_matches_returned_regardless_of_score(monkeypatch):
    monkeypatch.setenv("NEATMEM_API_KEY", "test-key")
    results = [
        {"id": "strong", "memory": "a", "score": 0.4},
        {"id": "weak", "memory": "b", "score": 0.08},
        {"id": "unscored", "memory": "c"},
    ]
    monkeypatch.setattr(
        memory_core,
        "_request_json_with_network_retry",
        lambda *args: ({"results": results}, 0, 0),
    )

    found = memory_core.search_memories(None, repo(), None, "q")
    assert [m["id"] for m in found.memories] == ["strong", "weak", "unscored"]


def test_doctor_flags_a_wildcard_user_id(isolated_env, monkeypatch):
    monkeypatch.setenv("NEATMEM_USER_ID", "*")

    with (
        patch.object(memory_core, "resolve_repo", return_value=repo()),
        patch.object(memory_core, "_get_json", return_value=({}, 10)),
    ):
        result = memory_core.doctor("/tmp/repo")

    assert result["ok"] is False
    assert result["checks"]["user_id"]["ok"] is False
    assert "wildcard" in result["checks"]["user_id"]["detail"]


def test_every_write_carries_the_session_run_id(isolated_env, monkeypatch):
    store = memory_core.EvidenceStore()
    _record_complete_session(store)
    _record_failed_then_recovered_command(store)

    with (
        patch.object(memory_core, "resolve_repo", return_value=repo()),
        patch.object(
            memory_core, "_request_json", side_effect=_flush_server(2)
        ) as request,
    ):
        memory_core.flush_session(store, {"session_id": "s1", "cwd": "/tmp/repo"}, "session-end")

    bodies = [call.args[2] for call in request.call_args_list]
    assert len(bodies) == 2
    assert all(body["run_id"] == "s1" for body in bodies)
    store.close()


def test_shared_transcripts_redact_without_truncating_before_extraction():
    secret = '"password": "plain-private-value" '
    prompt = secret + 'x' * 20000
    answer = secret + '界' * 40000
    events = [
        {"kind": "user_prompt", "payload": {"text": prompt}},
        {"kind": "assistant_stop", "payload": {"transcript_messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ]}},
    ]
    _, structured = memory_core.build_episode(repo(), "s1", "packet", events)
    messages = memory_core.build_extraction_messages(structured)
    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert "plain-private-value" not in json.dumps(structured)
    assert [message["content"] for message in messages] == [memory_core.redact(prompt), memory_core.redact(answer)]


def test_prepare_flush_locks_before_reading_events(isolated_env):
    store = memory_core.EvidenceStore()
    store.record_event(repo(), "s1", "user_prompt", {"text": "Explain this."})
    statements = []
    store.conn.set_trace_callback(statements.append)
    store.prepare_flush(repo(), "s1", "session-end")
    assert statements[0] == "BEGIN IMMEDIATE"
    assert not store.conn.in_transaction
    store.close()


def test_extraction_batches_bound_individual_messages_without_losing_text():
    content = '界\\"\n' * 1000
    messages = [{"role": "assistant", "content": content}]
    batches = memory_core.extraction_message_batches(messages, max_tokens=200)
    assert all(memory_core._message_tokens(batch) <= 200 for batch in batches)
    assert ''.join(message['content'] for batch in batches for message in batch) == content


def test_overlapping_subagent_stop_without_id_does_not_guess(isolated_env):
    store = memory_core.EvidenceStore()
    store.start_subagent(repo(), "s1", "first", "sidekick", 0)
    store.start_subagent(repo(), "s1", "second", "sidekick", 0)
    stopped = store.stop_subagent(repo(), "s1", "", "sidekick", "", "Uncorrelated response")
    assert stopped not in {"first", "second"}
    rows = store.conn.execute("SELECT agent_id, stopped_at, final_message FROM sidekick_runs").fetchall()
    assert all(row["stopped_at"] is None for row in rows if row["agent_id"] in {"first", "second"})
    assert next(row for row in rows if row["agent_id"] == stopped)["final_message"] == "Uncorrelated response"
    store.close()


def test_delayed_handoff_survives_failed_atomic_rewrite(isolated_env, monkeypatch):
    import flush_worker

    path = isolated_env / "checkpoint.running"
    original = {"delay_seconds": 10, "hook_input": {"session_id": "s1"}}
    path.write_text(json.dumps(original))
    monkeypatch.setattr(sys, "argv", ["flush_worker.py", str(path)])
    replace = Path.replace

    def fail_temporary(source, target):
        if source.suffix == ".tmp":
            raise OSError("simulated disk failure")
        return replace(source, target)

    with patch.object(Path, "replace", fail_temporary), patch.object(flush_worker, "checkpoint_session") as checkpoint:
        with pytest.raises(OSError, match="simulated disk failure"):
            flush_worker.main()
    assert json.loads(path.with_suffix(".json").read_text()) == original
    assert not list(isolated_env.glob("*.tmp"))
    checkpoint.assert_not_called()


def test_json_secret_redaction_handles_escaped_quotes():
    value = json.dumps({"password": 'prefix"private suffix'})
    assert json.loads(memory_core.redact(value)) == {"password": "[REDACTED]"}


# --- Same-session embargo (prompt-search only) ---

from datetime import datetime, timedelta, timezone  # noqa: E402


def _embargo_memory(mid, run_id, age_seconds, *, event_ts=None):
    created = (datetime.now(timezone.utc) - timedelta(seconds=age_seconds)).isoformat()
    memory = {
        "id": mid,
        "memory": f"memory text {mid}",
        "run_id": run_id,
        "created_at": created,
        "metadata": {},
    }
    if event_ts is not None:
        memory["metadata"]["timestamp"] = event_ts
    return memory


def _embargo_search(store, memories, monkeypatch, operation="prompt-search"):
    monkeypatch.setenv("NEATMEM_API_KEY", "m0-test-key")
    with (
        patch.object(memory_core, "resolve_repo", return_value=repo()),
        patch.object(
            memory_core,
            "_request_json",
            return_value=({"results": memories}, 100, 500),
        ),
    ):
        return memory_core.search_memories(
            store, repo(), "s1", "a query long enough to pass", operation=operation
        )


def test_embargo_suppresses_fresh_same_session_memory(isolated_env, monkeypatch):
    store = memory_core.EvidenceStore()
    memories = [_embargo_memory("m1", "s1", 600)]
    result = _embargo_search(store, memories, monkeypatch)
    assert result.memories == []
    store.close()


def test_embargo_passes_old_same_session_memory(isolated_env, monkeypatch):
    store = memory_core.EvidenceStore()
    memories = [_embargo_memory("m1", "s1", 7200)]
    result = _embargo_search(store, memories, monkeypatch)
    assert [m["id"] for m in result.memories] == ["m1"]
    store.close()


def test_embargo_passes_other_session_memory(isolated_env, monkeypatch):
    store = memory_core.EvidenceStore()
    memories = [_embargo_memory("m1", "other-session", 600)]
    result = _embargo_search(store, memories, monkeypatch)
    assert [m["id"] for m in result.memories] == ["m1"]
    store.close()


def test_embargo_compact_exemption_by_event_ts(isolated_env, monkeypatch):
    store = memory_core.EvidenceStore()
    compact_at = datetime.now(timezone.utc) - timedelta(seconds=300)
    store.set_setting("last_compact_at:s1", compact_at.isoformat())
    # created_at is AFTER compact (late flush) but event time is BEFORE compact:
    # exemption must judge by event time.
    event_ts = (compact_at - timedelta(seconds=600)).isoformat()
    memories = [_embargo_memory("m1", "s1", 120, event_ts=event_ts)]
    result = _embargo_search(store, memories, monkeypatch)
    assert [m["id"] for m in result.memories] == ["m1"]
    store.close()


def test_embargo_compact_marker_does_not_exempt_post_compact_memory(
    isolated_env, monkeypatch
):
    store = memory_core.EvidenceStore()
    compact_at = datetime.now(timezone.utc) - timedelta(seconds=600)
    store.set_setting("last_compact_at:s1", compact_at.isoformat())
    memories = [_embargo_memory("m1", "s1", 120)]  # created after compact
    result = _embargo_search(store, memories, monkeypatch)
    assert result.memories == []
    store.close()


def test_embargo_suppressed_memory_is_not_marked_injected(isolated_env, monkeypatch):
    store = memory_core.EvidenceStore()
    memories = [_embargo_memory("m1", "s1", 600)]
    _embargo_search(store, memories, monkeypatch)
    rows = store.conn.execute("SELECT memory_id FROM retrievals").fetchall()
    assert rows == []
    # 31 minutes later the same memory must be injectable.
    old = _embargo_memory("m1", "s1", 600 + 31 * 60)
    result = _embargo_search(store, [old], monkeypatch)
    assert [m["id"] for m in result.memories] == ["m1"]
    store.close()


def test_embargo_does_not_apply_to_explicit_search(isolated_env, monkeypatch):
    store = memory_core.EvidenceStore()
    memories = [_embargo_memory("m1", "s1", 600)]
    result = _embargo_search(store, memories, monkeypatch, operation="mcp-search")
    assert [m["id"] for m in result.memories] == ["m1"]
    store.close()


def test_embargo_disabled_by_zero(isolated_env, monkeypatch):
    monkeypatch.setenv("NEATMEM_CODE_SAME_SESSION_EMBARGO_SECONDS", "0")
    store = memory_core.EvidenceStore()
    memories = [_embargo_memory("m1", "s1", 60)]
    result = _embargo_search(store, memories, monkeypatch)
    assert [m["id"] for m in result.memories] == ["m1"]
    store.close()


def test_embargo_invalid_policy_falls_back_to_default(isolated_env, monkeypatch):
    store = memory_core.EvidenceStore()
    store.set_setting(
        "client_policy",
        json.dumps({"same_session_embargo_seconds": "bogus"}),
    )
    assert (
        memory_core.same_session_embargo_seconds(store)
        == memory_core.DEFAULT_SAME_SESSION_EMBARGO_SECONDS
    )
    store.close()


def test_embargo_overfetch_returns_remaining_after_filtering(
    isolated_env, monkeypatch
):
    store = memory_core.EvidenceStore()
    memories = [
        *[_embargo_memory(f"fresh-{i}", "s1", 300) for i in range(3)],
        *[_embargo_memory(f"old-{i}", "s1", 7200) for i in range(2)],
    ]
    with (
        patch.object(memory_core, "resolve_repo", return_value=repo()),
        patch.object(
            memory_core, "_request_json", return_value=({"results": memories}, 100, 500)
        ) as request,
    ):
        monkeypatch.setenv("NEATMEM_API_KEY", "m0-test-key")
        result = memory_core.search_memories(
            store, repo(), "s1", "a query long enough to pass",
            top_k=5, operation="prompt-search",
        )
    # over-fetch requested 10, embargo dropped 3 fresh, 2 old survive
    assert request.call_args.args[2]["top_k"] == 10
    assert sorted(m["id"] for m in result.memories) == ["old-0", "old-1"]
    suppressed = store.conn.execute(
        "SELECT item_count FROM operations WHERE operation='embargo-suppressed'"
    ).fetchone()
    assert suppressed and suppressed["item_count"] == 3
    store.close()


def test_embargo_falls_back_to_created_at_when_timestamp_invalid(
    isolated_env, monkeypatch
):
    store = memory_core.EvidenceStore()
    memory = _embargo_memory("m1", "s1", 7200, event_ts="not-a-date")
    result = _embargo_search(store, [memory], monkeypatch)
    assert [m["id"] for m in result.memories] == ["m1"]
    # Same invalid timestamp on a fresh memory stays suppressed via created_at.
    fresh = _embargo_memory("m2", "s1", 120, event_ts="not-a-date")
    result = _embargo_search(store, [fresh], monkeypatch)
    assert result.memories == []
    store.close()


def test_embargo_event_ts_missing_uses_created_at(isolated_env, monkeypatch):
    store = memory_core.EvidenceStore()
    memory = _embargo_memory("m1", "s1", 600)
    memory["metadata"] = None  # server always sends a dict; be defensive anyway
    result = _embargo_search(store, [memory], monkeypatch)
    assert result.memories == []
    store.close()


def test_record_session_start_writes_compact_marker(isolated_env):
    store = memory_core.EvidenceStore()
    memory_core.record_session_start(
        store,
        {"session_id": "s1", "cwd": "/tmp/repo", "source": "compact"},
    )
    marker = store.setting("last_compact_at:s1", "")
    assert marker and memory_core._parse_iso_datetime(marker) is not None
    # startup must NOT write the marker
    memory_core.record_session_start(
        store,
        {"session_id": "s2", "cwd": "/tmp/repo", "source": "startup"},
    )
    assert store.setting("last_compact_at:s2", "") == ""
    store.close()
