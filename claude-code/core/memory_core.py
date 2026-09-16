#!/usr/bin/env python3
"""Shared core for the NeatMem agent plugin.

Hooks record small session details locally. When the agent compacts or ends the
session, the plugin sends the useful parts to the local NeatMem server so it can
create memories. The agent can search those memories during later work in the
repository.
"""

from __future__ import annotations

import functools
import getpass
import hashlib
import json
import math
import os
import re
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_API_URL = "http://127.0.0.1:8000"
PLUGIN_VERSION = "0.1.0"

_harness_name: str = "generic"
_harness_env_prefix: str = "NEATMEM_PLUGIN"
_harness_data_dir_name: str = "neatmem-plugin"
_harness_source_tag: str = "neatmem_plugin"


def configure_harness(
    name: str,
    env_prefix: str = "",
    data_dir_name: str = "",
    source_tag: str = "",
) -> None:
    global _harness_name, _harness_env_prefix, _harness_data_dir_name, _harness_source_tag
    _harness_name = name
    _harness_env_prefix = env_prefix or f"NEATMEM_{name.upper().replace('-', '_')}"
    _harness_data_dir_name = data_dir_name or f"{name}-plugin"
    _harness_source_tag = source_tag or f"{name.replace('-', '_')}_plugin"


def harness_config() -> dict[str, str]:
    return {
        "name": _harness_name,
        "env_prefix": _harness_env_prefix,
        "data_dir_name": _harness_data_dir_name,
        "source_tag": _harness_source_tag,
    }


MAX_COMMAND_CHARS = 2000
MAX_RESULT_CHARS = 2500
MAX_EPISODE_CHARS = 12000
CHECKPOINT_EXCHANGES = 5
CHECKPOINT_MESSAGES = 10
CHECKPOINT_SOURCE_CHARS = 40000
DEFAULT_MAX_CONTEXT_CHARS = 4000
MAX_EXTRACTION_INPUT_TOKENS = 24000
MAX_FLUSH_ATTEMPTS = 5
FORGET_PAGE_SIZE = 100
FORGET_MAX_PAGES = 50

TEST_COMMAND_RE = re.compile(
    r"(?:^|\s)(?:pytest|py\.test|jest|vitest|go\s+test|cargo\s+test|"
    r"npm\s+(?:run\s+)?test|pnpm\s+(?:run\s+)?test|yarn\s+test|"
    r"mvn\s+test|gradle\s+test|make\s+test)(?:\s|$)",
    re.IGNORECASE,
)
BUILD_COMMAND_RE = re.compile(
    r"(?:^|\s)(?:npm|pnpm|yarn)\s+(?:run\s+)?build(?:\s|$)|"
    r"(?:^|\s)(?:cargo|go|mvn|gradle|make)\s+build(?:\s|$)",
    re.IGNORECASE,
)

SECRET_PATTERNS = [
    re.compile(r"(?i)(authorization\s*[:=]\s*(?:bearer|token)\s+)[^\s\"']+"),
    re.compile(
        r"(?i)((?:api[_-]?key|secret[_-]?access[_-]?key|session[_-]?token)\s*[:=]\s*)[^\s\"']+"
    ),
    re.compile(
        r"(?i)((?:access[_-]?token|refresh[_-]?token|password|credential)"
        r"\s*[:=]\s*)[^\s&\"']+"
    ),
    re.compile(r"\b(?:sk|m0|mem0_sk|psk)-[A-Za-z0-9_\-]{12,}\b"),
    re.compile(r"\b(?:ASIA|AKIA)[A-Z0-9]{12,}\b"),
    re.compile(r"\b(?:ghp_|github_pat_|xox[baprs]-)[A-Za-z0-9_\-]{12,}\b"),
    re.compile(
        r"-----BEGIN [^-]*PRIVATE KEY-----.*?-----END [^-]*PRIVATE KEY-----",
        re.DOTALL,
    ),
    re.compile(
        r'(?i)("(?:api[_-]?key|password|secret(?:[_-]?access[_-]?key)?'
        r'|(?:access|refresh|session)[_-]?token|token|authorization|credential'
        r')"\s*:\s*")(?:\\.|[^"\\])*'
    ),
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def redact(value: Any) -> str:
    text = (
        value
        if isinstance(value, str)
        else json.dumps(value, ensure_ascii=False, default=str)
    )
    for pattern in SECRET_PATTERNS:
        if pattern.groups:
            text = pattern.sub(r"\1[REDACTED]", text)
        else:
            text = pattern.sub("[REDACTED]", text)
    return text


def bounded(value: Any, limit: int) -> str:
    text = redact(value).strip()
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} chars]"


def _git(cwd: str, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", cwd, *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=0.5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def _normalize_remote(remote: str) -> str:
    remote = remote.strip()
    if remote.startswith("git@") and ":" in remote:
        host_path = remote[4:].replace(":", "/", 1)
        remote = f"https://{host_path}"
    if remote.endswith(".git"):
        remote = remote[:-4]
    if "://" in remote:
        parsed = urllib.parse.urlsplit(remote)
        hostname = parsed.hostname or ""
        if parsed.port:
            hostname = f"{hostname}:{parsed.port}"
        remote = urllib.parse.urlunsplit(
            (parsed.scheme, hostname, parsed.path, parsed.query, parsed.fragment)
        )
    return remote.rstrip("/")


_WILDCARD_SCOPE = re.compile(r"^\*+$")


def _scope_value(raw: str | None) -> str:
    """Reject wildcards as identities: they are filter syntax and would widen the scope."""
    value = (raw or "").strip()
    return "" if _WILDCARD_SCOPE.match(value) else value


SEARCH_SCOPES = ("repo", "mine")
DEFAULT_SEARCH_SCOPE = "repo"

def directory_app_id(repo: RepoContext) -> str:
    """The app_id of the directory this session runs in: the repository at the root, repository/path below it."""
    return f"{repo.app_id}/{repo.directory}" if repo.directory else repo.app_id


def _search_filters(user: str, repo: RepoContext, scope: str) -> dict[str, Any]:
    """Build the scope filter. Every write carries user_id + agent_id(project),
    so the repo lane is the project agent and mine narrows it to this user.
    Filters stay flat: the server requires a top-level user_id/agent_id/run_id
    key, and flat keys are implicitly ANDed by the vector store."""
    if scope == "mine":
        return {"user_id": user, "agent_id": repo.project_id}
    return {"agent_id": repo.project_id}


def search_scope() -> str:
    configured = (
        _plugin_option("search_scope", "NEATMEM_CODE_SEARCH_SCOPE") or ""
    ).strip().lower()
    return configured if configured in SEARCH_SCOPES else DEFAULT_SEARCH_SCOPE


def resolve_search_scope(scope: str | None) -> str:
    value = (scope or search_scope()).strip().lower()
    if value not in SEARCH_SCOPES:
        raise ValueError(f"Unknown search scope: {value}")
    return value


def _legacy_project_id(cwd: str, root: str, raw_remote: str, identity: str) -> str:
    """Human-readable repository namespace (owner-repo), the base of project_id."""
    configured = _scope_value(os.environ.get("NEATMEM_PROJECT_ID"))
    if configured:
        return configured

    remote = raw_remote or ("" if identity.startswith("local:") else identity)
    remote = remote.strip().removesuffix(".git")
    for prefix in ("https://", "http://", "ssh://", "git://"):
        if remote.startswith(prefix):
            remote = remote[len(prefix) :]
            break
    else:
        remote = re.sub(r"^git@", "", remote)
    parts = [part for part in remote.replace(":", "/", 1).split("/") if part]
    if len(parts) >= 2:
        return f"{parts[-2]}-{parts[-1]}".replace("/", "-").replace(":", "-")
    if parts:
        return parts[-1].replace("/", "-").replace(":", "-")
    return os.path.basename(root or cwd) or "unknown"


@dataclass(frozen=True)
class RepoContext:
    cwd: str
    root: str
    identity: str
    app_id: str
    branch: str
    head_sha: str
    project_id: str = ""
    directory: str = ""


def _project_id(root: str, identity: str, app_id: str) -> str:
    """The shared namespace: includes a host hash so repos with the same owner/name on different hosts stay apart."""
    if not identity.startswith("local:"):
        return f"{app_id}-{hashlib.sha256(identity.encode()).hexdigest()[:10]}"
    return f"local-{app_id}-{hashlib.sha256(root.encode()).hexdigest()[:10]}"


def _relative_directory(cwd: str, root: str) -> str:
    relative = os.path.relpath(cwd, root)
    return "" if relative == "." or relative.startswith("..") else relative.replace(os.sep, "/")


@dataclass(frozen=True)
class MemorySearchResult:
    succeeded: bool
    matched_count: int
    already_shown_count: int
    memories: list[dict[str, Any]]


@functools.lru_cache(maxsize=64)
def _resolve_repo_cached(cwd: str) -> RepoContext:
    given_cwd = cwd
    cwd = os.path.realpath(cwd)
    given_root = _git(cwd, "rev-parse", "--show-toplevel") or given_cwd
    root = os.path.realpath(given_root)
    raw_remote = _git(root, "config", "--get", "remote.origin.url")
    remote = _normalize_remote(raw_remote)
    identity = remote or f"local:{root}"
    app_id = _legacy_project_id(given_cwd, given_root, raw_remote, identity)
    return RepoContext(
        cwd=cwd,
        root=root,
        identity=identity,
        app_id=app_id,
        branch=_git(root, "branch", "--show-current") or "detached",
        head_sha=_git(root, "rev-parse", "HEAD"),
        project_id=_project_id(root, identity, app_id),
        directory=_relative_directory(cwd, root),
    )


def resolve_repo(cwd: str | None) -> RepoContext:
    return _resolve_repo_cached(os.path.abspath(cwd or os.getcwd()))


def api_key() -> str:
    configured = (
        os.environ.get("NEATMEM_API_KEY")
        or os.environ.get("PLUGIN_OPTION_API_KEY")
        or os.environ.get("CLAUDE_PLUGIN_OPTION_API_KEY")
        or os.environ.get("CLAUDE_PLUGIN_OPTION_NEATMEM_API_KEY")
        or ""
    ).strip()
    if configured:
        return configured
    try:
        return (data_dir() / "api-key").read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def cache_plugin_api_key() -> bool:
    """Bridge host's hook-only sensitive config into plugin-owned storage."""
    configured = (
        os.environ.get("PLUGIN_OPTION_API_KEY")
        or os.environ.get("CLAUDE_PLUGIN_OPTION_API_KEY")
        or os.environ.get("CLAUDE_PLUGIN_OPTION_NEATMEM_API_KEY")
        or ""
    ).strip()
    if not configured:
        return False

    directory = data_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "api-key"
    temporary = directory / f"api-key.{os.getpid()}.tmp"
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(configured)
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return True


def clear_stale_api_key_cache() -> bool:
    """Drop the cached key file once every configured key source is gone."""
    configured = (
        os.environ.get("NEATMEM_API_KEY")
        or os.environ.get("PLUGIN_OPTION_API_KEY")
        or os.environ.get("CLAUDE_PLUGIN_OPTION_API_KEY")
        or os.environ.get("CLAUDE_PLUGIN_OPTION_NEATMEM_API_KEY")
        or ""
    ).strip()
    if configured:
        return False
    path = data_dir() / "api-key"
    if not path.exists():
        return False
    try:
        path.unlink()
    except OSError:
        return False
    return True


def detached_process_kwargs(platform: str | None = None) -> dict:
    """Keep a spawned worker alive after the coding agent exits, on POSIX and Windows."""
    if (platform or sys.platform) == "win32":
        return {
            "creationflags": subprocess.DETACHED_PROCESS
            | subprocess.CREATE_NEW_PROCESS_GROUP
        }
    return {"start_new_session": True}


def _plugin_option(name: str, fallback: str = "") -> str:
    return (
        os.environ.get(f"PLUGIN_OPTION_{name.upper()}")
        or os.environ.get(f"CLAUDE_PLUGIN_OPTION_{name.upper()}")
        or os.environ.get(fallback)
        or ""
    ).strip()


def api_url() -> str:
    """NeatMem server base URL: CC userConfig bridge or env, then default."""
    return (_plugin_option("api_url", "NEATMEM_API_URL") or DEFAULT_API_URL).rstrip("/")


def user_id() -> str:
    configured = _scope_value(_plugin_option("user_id", "NEATMEM_USER_ID"))
    if configured:
        return configured
    # Hook subprocesses start with a scrubbed environment (no USER/USERNAME),
    # so resolve the account through getpass, which falls back to the OS
    # password database instead of env alone.
    try:
        account = getpass.getuser()
    except (KeyError, OSError):
        account = ""
    return _scope_value(account) or "default"


def data_dir() -> Path:
    configured = (
        os.environ.get("NEATMEM_CODE_DATA_DIR")
        or os.environ.get("NEATMEM_PLUGIN_DATA_DIR")
        or os.environ.get("PLUGIN_DATA")
        or os.environ.get("CLAUDE_PLUGIN_DATA")
    )
    return (
        Path(configured).expanduser() if configured else Path.home() / ".neatmem" / _harness_data_dir_name
    )


def _bool_option(name: str, fallback: str, default: bool = False) -> bool:
    value = _plugin_option(name, fallback)
    if not value:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _int_option(name: str, fallback: str, default: int) -> int:
    value = _plugin_option(name, fallback)
    try:
        return int(value) if value else default
    except ValueError:
        return default


INJECT_TIMINGS = ("off", "first", "every")
DEFAULT_INJECT_TIMING = "first"
DEFAULT_MIN_QUERY_CHARS = 20
CLIENT_POLICY_TIMEOUT_SECONDS = 2

_DEFAULT_CLIENT_POLICY = {
    "inject_timing": DEFAULT_INJECT_TIMING,
    "min_query_chars": DEFAULT_MIN_QUERY_CHARS,
    "source": "default",
    "error": "",
}


def client_policy(store: "EvidenceStore") -> dict[str, Any]:
    """Client policy cached by the last SessionStart fetch, else defaults."""
    policy = dict(_DEFAULT_CLIENT_POLICY)
    raw = store.setting("client_policy", "")
    if raw:
        try:
            cached = json.loads(raw)
        except json.JSONDecodeError:
            cached = None
        if isinstance(cached, dict):
            for key in policy:
                if key in cached:
                    policy[key] = cached[key]
    try:
        policy["min_query_chars"] = max(int(policy["min_query_chars"]), 1)
    except (TypeError, ValueError):
        policy["min_query_chars"] = DEFAULT_MIN_QUERY_CHARS
    if policy["inject_timing"] not in INJECT_TIMINGS:
        policy["inject_timing"] = DEFAULT_INJECT_TIMING
    return policy


def fetch_client_policy(store: "EvidenceStore") -> dict[str, Any]:
    """Pull the client policy from GET /v1/config/ and cache it for the session.

    The server only serves values; enforcement happens here. On failure the
    built-in defaults are cached with an error note so the degradation is
    surfaced to the session instead of failing silently.
    """
    policy = dict(_DEFAULT_CLIENT_POLICY)
    try:
        payload, _ = _get_json(
            f"{api_url()}/v1/config/", api_key(), CLIENT_POLICY_TIMEOUT_SECONDS
        )
        remote = payload.get("client_policy") if isinstance(payload, dict) else None
        if not isinstance(remote, dict):
            raise ValueError("server config has no client_policy")
        timing = str(remote.get("inject_timing") or "").strip()
        if timing in INJECT_TIMINGS:
            policy["inject_timing"] = timing
        try:
            policy["min_query_chars"] = max(int(remote.get("min_query_chars")), 1)
        except (TypeError, ValueError):
            pass
        policy["source"] = "server"
    except Exception as exc:  # hooks must fail open
        policy["error"] = bounded(str(exc), 300)
    store.set_setting("client_policy", json.dumps(policy))
    return policy


def inject_timing(store: "EvidenceStore") -> str:
    """Effective inject timing: client env override > server policy > default."""
    override = os.environ.get("NEATMEM_INJECT_TIMING", "").strip()
    if override in INJECT_TIMINGS:
        return override
    return str(client_policy(store)["inject_timing"])


def plugin_enabled() -> bool:
    """Process-level master switch: NEATMEM_ENABLED=0 turns all hooks off."""
    return os.environ.get("NEATMEM_ENABLED", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }



def _checkpoint_message(event: dict[str, Any]) -> str:
    kind = event.get("kind")
    payload = event.get("payload") or {}
    if kind == "user_prompt":
        return redact(payload.get("text", "")).strip()
    if kind == "assistant_stop":
        transcript_messages = payload.get("transcript_messages") or []
        if isinstance(transcript_messages, list):
            text = "\n".join(
                str(message.get("content") or "")
                for message in transcript_messages
                if isinstance(message, dict) and message.get("content")
            )
            if text:
                return text
        return redact(payload.get("text", "")).strip()
    if kind in {"subagent_stop", "sidekick_stop"}:
        return redact(payload.get("final_message", "")).strip()
    return ""


def checkpoint_stats(events: list[dict[str, Any]]) -> tuple[int, int, int]:
    """Return completed exchanges, messages, and source characters."""
    completed = sum(event.get("kind") == "assistant_stop" for event in events)
    contents = [content for event in events if (content := _checkpoint_message(event))]
    return completed, len(contents), sum(len(content) for content in contents)


def select_checkpoint_events(
    events: list[dict[str, Any]], *, force: bool
) -> list[dict[str, Any]]:
    """Select one ordered extraction block without splitting an exchange."""
    for index, event in enumerate(events):
        if event.get("kind") != "assistant_stop":
            continue
        candidate = events[: index + 1]
        completed, messages, source_chars = checkpoint_stats(candidate)
        if (
            completed >= CHECKPOINT_EXCHANGES
            or messages >= CHECKPOINT_MESSAGES
            or source_chars >= CHECKPOINT_SOURCE_CHARS
        ):
            return candidate
    return events if force else []


class EvidenceStore:
    def __init__(self, path: Path | None = None):
        directory = data_dir() if path is None else path.parent
        directory.mkdir(parents=True, exist_ok=True)
        self.path = path or directory / "evidence.sqlite3"
        try:
            self._open()
        except sqlite3.DatabaseError:
            self._quarantine()
            self._open()

    def _open(self) -> None:
        self.conn = sqlite3.connect(self.path, timeout=10)
        self.conn.row_factory = sqlite3.Row
        try:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA busy_timeout=10000")
            self._migrate()
        except sqlite3.DatabaseError:
            self.conn.close()
            raise

    def _quarantine(self) -> None:
        """Move an unreadable database aside so capture restarts cleanly."""
        stamp = int(time.time())
        for suffix in ("", "-wal", "-shm"):
            source = Path(f"{self.path}{suffix}")
            try:
                source.replace(f"{self.path}.corrupt-{stamp}{suffix}")
            except FileNotFoundError:
                continue
            except OSError:
                try:
                    source.unlink()
                except OSError:
                    pass

    def close(self) -> None:
        self.conn.close()

    def _migrate(self) -> None:
        # Keep the legacy table name so existing databases and in-flight workers remain compatible.
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                repo_id TEXT NOT NULL,
                app_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                kind TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                flush_id TEXT
            );
            CREATE INDEX IF NOT EXISTS events_session_idx
                ON events(repo_id, session_id, flush_id, id);

            CREATE TABLE IF NOT EXISTS session_scopes (
                session_id TEXT PRIMARY KEY,
                repo_id TEXT NOT NULL,
                app_id TEXT NOT NULL,
                root TEXT NOT NULL,
                branch TEXT NOT NULL,
                head_sha TEXT NOT NULL,
                created_at TEXT NOT NULL,
                directory TEXT NOT NULL DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS flushes (
                packet_id TEXT PRIMARY KEY,
                repo_id TEXT NOT NULL,
                app_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                reason TEXT NOT NULL,
                event_start INTEGER NOT NULL,
                event_end INTEGER NOT NULL,
                status TEXT NOT NULL,
                episode_event_id TEXT,
                semantic_event_id TEXT,
                error TEXT,
                attempts INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS retrievals (
                session_id TEXT NOT NULL,
                repo_id TEXT NOT NULL,
                memory_id TEXT NOT NULL,
                injected_at TEXT NOT NULL,
                rank INTEGER,
                score REAL,
                memory_text TEXT,
                context_chars INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY(session_id, repo_id, memory_id)
            );

            CREATE TABLE IF NOT EXISTS operations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                repo_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                operation TEXT NOT NULL,
                duration_ms REAL NOT NULL,
                success INTEGER NOT NULL,
                item_count INTEGER NOT NULL DEFAULT 0,
                request_chars INTEGER NOT NULL DEFAULT 0,
                response_chars INTEGER NOT NULL DEFAULT 0,
                error TEXT
            );

            CREATE TABLE IF NOT EXISTS sidekick_runs (
                repo_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                agent_type TEXT NOT NULL,
                started_at TEXT NOT NULL,
                stopped_at TEXT,
                transcript_path TEXT,
                context_chars INTEGER NOT NULL DEFAULT 0,
                final_message TEXT,
                PRIMARY KEY(repo_id, session_id, agent_id)
            );
            CREATE INDEX IF NOT EXISTS sidekick_runs_repo_idx
                ON sidekick_runs(repo_id, started_at);

            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            """
        )
        # Remove the pre-0.1.1 snapshot implementation.
        self.conn.executescript(
            """
            DROP TABLE IF EXISTS sidekick_calls;
            DROP TABLE IF EXISTS sidekick_snapshots;
            DROP TABLE IF EXISTS sidekick_state;
            DROP TABLE IF EXISTS sidekick_packets;
            """
        )
        self._ensure_column("retrievals", "rank", "INTEGER")
        self._ensure_column("retrievals", "score", "REAL")
        self._ensure_column("retrievals", "memory_text", "TEXT")
        self._ensure_column("retrievals", "context_chars", "INTEGER NOT NULL DEFAULT 0")
        self._ensure_column("flushes", "attempts", "INTEGER NOT NULL DEFAULT 0")
        self._ensure_column("session_scopes", "directory", "TEXT NOT NULL DEFAULT ''")
        self.conn.commit()

    def _ensure_column(self, table: str, column: str, declaration: str) -> None:
        columns = {
            str(row["name"])
            for row in self.conn.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column not in columns:
            self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")

    def record_event(
        self,
        repo: RepoContext,
        session_id: str,
        kind: str,
        payload: dict[str, Any],
    ) -> int:
        cursor = self.conn.execute(
            """INSERT INTO events
               (repo_id, app_id, session_id, created_at, kind, payload_json)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                repo.identity,
                repo.app_id,
                session_id,
                utc_now(),
                kind,
                json.dumps(payload, ensure_ascii=False, sort_keys=True),
            ),
        )
        self.conn.commit()
        return int(cursor.lastrowid)

    def record_assistant_response(self, repo: RepoContext, session_id: str, message: str) -> None:
        """Ignore repeated response hooks until another prompt or a different answer arrives."""
        with self.conn:
            # Serialize the check and insert across concurrent Stop and SessionEnd hooks.
            self.conn.execute("BEGIN IMMEDIATE")
            previous = self.conn.execute(
                """SELECT kind, payload_json FROM events
                   WHERE repo_id = ? AND session_id = ? AND kind IN ('user_prompt', 'assistant_stop')
                   ORDER BY id DESC LIMIT 1""",
                (repo.identity, session_id),
            ).fetchone()
            if (
                previous is not None
                and previous["kind"] == "assistant_stop"
                and json.loads(previous["payload_json"]).get("text") == message
            ):
                return
            self.record_event(repo, session_id, "assistant_stop", {"text": message})

    def repo_for_session(self, session_id: str, cwd: str | None) -> RepoContext:
        """Keep one project scope for every hook in a coding-agent session."""
        current = resolve_repo(cwd)
        if session_id == "unknown-session":
            return current

        with self.conn:
            self.conn.execute(
                """INSERT OR IGNORE INTO session_scopes
                   (session_id, repo_id, app_id, root, branch, head_sha, created_at, directory)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    current.identity,
                    current.app_id,
                    current.root,
                    current.branch,
                    current.head_sha,
                    utc_now(),
                    current.directory,
                ),
            )
        scope = self.conn.execute(
            "SELECT * FROM session_scopes WHERE session_id = ?", (session_id,)
        ).fetchone()
        same_git_repo = (
            current.identity == scope["repo_id"] and bool(current.head_sha)
        )
        pinned = current if same_git_repo else resolve_repo(str(scope["root"]))
        return RepoContext(
            cwd=current.cwd,
            root=pinned.root,
            identity=str(scope["repo_id"]),
            app_id=str(scope["app_id"]),
            branch=pinned.branch,
            head_sha=pinned.head_sha,
            project_id=pinned.project_id,
            directory=str(scope["directory"] or ""),
        )

    def prepare_flush(
        self, repo: RepoContext, session_id: str, reason: str
    ) -> tuple[str, list[dict[str, Any]]] | None:
        with self.conn:
            self.conn.execute("BEGIN IMMEDIATE")
            existing = self.conn.execute(
                """SELECT * FROM flushes
                   WHERE repo_id = ? AND session_id = ?
                     AND status NOT IN ('succeeded', 'gave-up')
                   ORDER BY created_at LIMIT 1""",
                (repo.identity, session_id),
            ).fetchone()
            if existing and int(existing["attempts"] or 0) >= MAX_FLUSH_ATTEMPTS:
                self.conn.execute(
                    "UPDATE flushes SET status = 'gave-up', updated_at = ? WHERE packet_id = ?",
                    (utc_now(), existing["packet_id"]),
                )
                existing = None
            if existing:
                if reason != "periodic" and existing["reason"] == "periodic":
                    self.conn.execute(
                        "UPDATE flushes SET reason = ?, updated_at = ? WHERE packet_id = ?",
                        (reason, utc_now(), existing["packet_id"]),
                    )
                existing_rows = self.conn.execute(
                    "SELECT * FROM events WHERE flush_id = ? ORDER BY id",
                    (existing["packet_id"],),
                ).fetchall()
                if existing_rows:
                    return str(existing["packet_id"]), [
                        {
                            "id": row["id"],
                            "created_at": row["created_at"],
                            "kind": row["kind"],
                            "payload": json.loads(row["payload_json"]),
                        }
                        for row in existing_rows
                    ]

            rows = self.conn.execute(
                """SELECT * FROM events
                   WHERE repo_id = ? AND session_id = ? AND flush_id IS NULL
                   ORDER BY id""",
                (repo.identity, session_id),
            ).fetchall()
            if not rows:
                return None

            events = [
                {
                    "id": row["id"],
                    "created_at": row["created_at"],
                    "kind": row["kind"],
                    "payload": json.loads(row["payload_json"]),
                }
                for row in rows
            ]
            events = select_checkpoint_events(events, force=reason != "periodic")
            if not events:
                return None
            event_start, event_end = events[0]["id"], events[-1]["id"]
            packet_material = f"{repo.identity}\0{session_id}\0{event_start}\0{event_end}"
            packet_id = hashlib.sha256(packet_material.encode()).hexdigest()[:32]
            now = utc_now()

            self.conn.execute(
                """INSERT OR IGNORE INTO flushes
                   (packet_id, repo_id, app_id, session_id, reason, event_start,
                    event_end, status, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 'prepared', ?, ?)""",
                (
                    packet_id,
                    repo.identity,
                    repo.app_id,
                    session_id,
                    reason,
                    event_start,
                    event_end,
                    now,
                    now,
                ),
            )
            event_ids = [event["id"] for event in events]
            placeholders = ", ".join("?" for _ in event_ids)
            self.conn.execute(
                f"UPDATE events SET flush_id = ? "
                f"WHERE id IN ({placeholders}) AND flush_id IS NULL",
                (packet_id, *event_ids),
            )
            return packet_id, events

    def checkpoint_due(self, repo_id: str, session_id: str) -> bool:
        if self.has_inflight_flush(repo_id, session_id):
            return False
        rows = self.conn.execute(
            """SELECT * FROM events
               WHERE repo_id = ? AND session_id = ? AND flush_id IS NULL
               ORDER BY id""",
            (repo_id, session_id),
        ).fetchall()
        events = [
            {
                "id": row["id"],
                "created_at": row["created_at"],
                "kind": row["kind"],
                "payload": json.loads(row["payload_json"]),
            }
            for row in rows
        ]
        return bool(select_checkpoint_events(events, force=False))

    def has_inflight_flush(self, repo_id: str, session_id: str) -> bool:
        return (
            self.conn.execute(
                """SELECT 1 FROM flushes
                   WHERE repo_id = ? AND session_id = ?
                     AND status IN ('prepared')
                   LIMIT 1""",
                (repo_id, session_id),
            ).fetchone()
            is not None
        )

    def flush_record(self, packet_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM flushes WHERE packet_id = ?", (packet_id,)
        ).fetchone()
        return dict(row) if row else None

    def has_unflushed_events(self, repo_id: str, session_id: str) -> bool:
        return (
            self.conn.execute(
                """SELECT 1 FROM events
                   WHERE repo_id = ? AND session_id = ? AND flush_id IS NULL
                   LIMIT 1""",
                (repo_id, session_id),
            ).fetchone()
            is not None
        )

    def unflushed_starts_with_session_start(
        self, repo_id: str, session_id: str
    ) -> bool:
        row = self.conn.execute(
            """SELECT kind FROM events
               WHERE repo_id = ? AND session_id = ? AND flush_id IS NULL
               ORDER BY id LIMIT 1""",
            (repo_id, session_id),
        ).fetchone()
        return bool(row and row["kind"] == "session_start")

    def update_flush(self, packet_id: str, **fields: Any) -> None:
        allowed = {"status", "episode_event_id", "semantic_event_id", "error"}
        updates = {key: value for key, value in fields.items() if key in allowed}
        updates["updated_at"] = utc_now()
        clause = ", ".join(f"{key} = ?" for key in updates)
        failed = str(fields.get("status", "")) in {
            "error",
            "semantic-failed",
            "semantic-timeout",
            "semantic-missing",
        }
        if failed:
            clause += ", attempts = attempts + 1"
        with self.conn:
            self.conn.execute(
                f"UPDATE flushes SET {clause} WHERE packet_id = ?",
                [*updates.values(), packet_id],
            )

    def unseen(
        self, session_id: str, repo_id: str, memories: Iterable[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        seen = {
            row["memory_id"]
            for row in self.conn.execute(
                "SELECT memory_id FROM retrievals WHERE session_id = ? AND repo_id = ?",
                (session_id, repo_id),
            )
        }
        return [memory for memory in memories if str(memory.get("id", "")) not in seen]

    def mark_injected(
        self, session_id: str, repo_id: str, memories: Iterable[dict[str, Any]]
    ) -> None:
        now = utc_now()
        with self.conn:
            for rank, memory in enumerate(memories, start=1):
                memory_id = str(memory.get("id", ""))
                if memory_id:
                    memory_text = bounded(
                        memory.get("memory") or memory.get("text") or "",
                        4000,
                    )
                    try:
                        score = float(memory["score"])
                    except (KeyError, TypeError, ValueError):
                        score = None
                    self.conn.execute(
                        """INSERT OR IGNORE INTO retrievals
                           (session_id, repo_id, memory_id, injected_at, rank,
                            score, memory_text, context_chars)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            session_id,
                            repo_id,
                            memory_id,
                            now,
                            rank,
                            score,
                            memory_text,
                            len(memory_text),
                        ),
                    )

    def injected_memories(
        self, session_id: str, repo_id: str
    ) -> list[dict[str, Any]]:
        """Return the exact memories already supplied to the main conversation."""
        rows = self.conn.execute(
            """SELECT memory_id, rank, score, memory_text
               FROM retrievals
               WHERE session_id = ? AND repo_id = ?
               ORDER BY COALESCE(rank, 2147483647), injected_at, memory_id""",
            (session_id, repo_id),
        ).fetchall()
        return [
            {
                "id": row["memory_id"],
                "memory": row["memory_text"],
                "score": row["score"],
            }
            for row in rows
            if row["memory_text"]
        ]

    def start_subagent(
        self,
        repo: RepoContext,
        session_id: str,
        agent_id: str,
        agent_type: str,
        context_chars: int,
    ) -> bool:
        """Record one native subagent instance and whether context was first sent."""
        with self.conn:
            cursor = self.conn.execute(
                """INSERT OR IGNORE INTO sidekick_runs
                   (repo_id, session_id, agent_id, agent_type, started_at,
                    context_chars)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    repo.identity,
                    session_id,
                    agent_id,
                    agent_type,
                    utc_now(),
                    context_chars,
                ),
            )
        return int(cursor.rowcount) > 0

    def stop_subagent(
        self,
        repo: RepoContext,
        session_id: str,
        agent_id: str,
        agent_type: str,
        transcript_path: str,
        final_message: str,
    ) -> str:
        now = utc_now()
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            if not agent_id:
                rows = self.conn.execute(
                    """SELECT agent_id FROM sidekick_runs
                       WHERE repo_id = ? AND session_id = ? AND agent_type = ? AND stopped_at IS NULL
                       LIMIT 2""",
                    (repo.identity, session_id, agent_type),
                ).fetchall()
                # Without a host ID, overlapping runs cannot be correlated reliably.
                agent_id = rows[0]["agent_id"] if len(rows) == 1 else f"unknown-agent-{time.time_ns()}"
            self.conn.execute(
                """INSERT INTO sidekick_runs
                   (repo_id, session_id, agent_id, agent_type, started_at,
                    stopped_at, transcript_path, final_message)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(repo_id, session_id, agent_id) DO UPDATE SET
                     stopped_at = excluded.stopped_at,
                     transcript_path = excluded.transcript_path,
                     final_message = excluded.final_message""",
                (
                    repo.identity,
                    session_id,
                    agent_id,
                    agent_type,
                    now,
                    now,
                    bounded(transcript_path, 2000),
                    redact(final_message).strip(),
                ),
            )
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
        return agent_id

    def operation(
        self,
        repo: RepoContext,
        session_id: str,
        operation: str,
        duration_ms: float,
        success: bool,
        *,
        item_count: int = 0,
        request_chars: int = 0,
        response_chars: int = 0,
        error: str = "",
    ) -> None:
        with self.conn:
            self.conn.execute(
                """INSERT INTO operations
                   (created_at, repo_id, session_id, operation, duration_ms,
                    success, item_count, request_chars, response_chars, error)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    utc_now(),
                    repo.identity,
                    session_id,
                    operation,
                    duration_ms,
                    int(success),
                    item_count,
                    request_chars,
                    response_chars,
                    bounded(error, 1000),
                ),
            )

    def has_operation(self, repo_id: str, session_id: str, operation: str) -> bool:
        row = self.conn.execute(
            """SELECT 1 FROM operations
               WHERE repo_id = ? AND session_id = ? AND operation = ?
               LIMIT 1""",
            (repo_id, session_id, operation),
        ).fetchone()
        return row is not None

    def has_event(self, repo_id: str, session_id: str, kind: str) -> bool:
        row = self.conn.execute(
            """SELECT 1 FROM events
               WHERE repo_id = ? AND session_id = ? AND kind = ?
               LIMIT 1""",
            (repo_id, session_id, kind),
        ).fetchone()
        return row is not None

    def latest_event_payload(
        self, repo_id: str, session_id: str, kind: str
    ) -> dict[str, Any]:
        row = self.conn.execute(
            """SELECT payload_json FROM events
               WHERE repo_id = ? AND session_id = ? AND kind = ?
               ORDER BY id DESC LIMIT 1""",
            (repo_id, session_id, kind),
        ).fetchone()
        if not row:
            return {}
        try:
            payload = json.loads(row["payload_json"])
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}

    def setting(self, key: str, default: str = "") -> str:
        row = self.conn.execute(
            "SELECT value FROM settings WHERE key = ?", (key,)
        ).fetchone()
        return str(row["value"]) if row else default

    def set_setting(self, key: str, value: str) -> None:
        with self.conn:
            self.conn.execute(
                """INSERT INTO settings(key, value, updated_at)
                   VALUES (?, ?, ?)
                   ON CONFLICT(key) DO UPDATE SET
                     value = excluded.value,
                     updated_at = excluded.updated_at""",
                (key, value, utc_now()),
            )

    def is_paused(self) -> bool:
        return self.setting("paused", "false").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    def forget_local_repo(self, repo_id: str) -> dict[str, int]:
        tables = {
            "events": "repo_id",
            "session_scopes": "repo_id",
            "flushes": "repo_id",
            "retrievals": "repo_id",
            "operations": "repo_id",
            "sidekick_runs": "repo_id",
        }
        removed: dict[str, int] = {}
        with self.conn:
            for table, column in tables.items():
                cursor = self.conn.execute(
                    f"DELETE FROM {table} WHERE {column} = ?", (repo_id,)
                )
                removed["subagent_runs" if table == "sidekick_runs" else table] = max(int(cursor.rowcount), 0)
        return removed

    def status(self, repo_id: str) -> dict[str, Any]:
        def count(table: str) -> int:
            return int(
                self.conn.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE repo_id = ?", (repo_id,)
                ).fetchone()[0]
            )

        last_operation = self.conn.execute(
            """SELECT created_at, operation, duration_ms, success, item_count, error
               FROM operations WHERE repo_id = ? ORDER BY id DESC LIMIT 1""",
            (repo_id,),
        ).fetchone()
        last_subagent = self.conn.execute(
            """SELECT session_id, agent_id, agent_type, started_at, stopped_at,
                      context_chars
               FROM sidekick_runs WHERE repo_id = ?
               ORDER BY started_at DESC LIMIT 1""",
            (repo_id,),
        ).fetchone()
        return {
            "paused": self.is_paused(),
            "events": count("events"),
            "flushes": count("flushes"),
            "retrievals": count("retrievals"),
            "subagent_runs": count("sidekick_runs"),
            "last_operation": dict(last_operation) if last_operation else None,
            "last_subagent": dict(last_subagent) if last_subagent else None,
        }


def _session_id(hook_input: dict[str, Any]) -> str:
    return str(hook_input.get("session_id") or "unknown-session")


def record_session_start(store: EvidenceStore, hook_input: dict[str, Any]) -> None:
    session_id = _session_id(hook_input)
    repo = store.repo_for_session(session_id, hook_input.get("cwd"))
    store.record_event(
        repo,
        session_id,
        "session_start",
        {
            "source": hook_input.get("source", "startup"),
            "model": bounded(hook_input.get("model", ""), 200),
            "branch": repo.branch,
            "head_sha": repo.head_sha,
        },
    )


def record_user_prompt(
    store: EvidenceStore, hook_input: dict[str, Any]
) -> tuple[RepoContext, str, str, bool]:
    session_id = _session_id(hook_input)
    repo = store.repo_for_session(session_id, hook_input.get("cwd"))
    prompt = redact(hook_input.get("prompt", "")).strip()
    is_first_prompt = not store.has_event(repo.identity, session_id, "user_prompt")
    store.record_event(repo, session_id, "user_prompt", {"text": prompt})
    return repo, session_id, prompt, is_first_prompt


def _tool_result_preview(response: Any) -> str:
    if isinstance(response, dict):
        selected = {}
        for key in (
            "stdout",
            "stderr",
            "output",
            "content",
            "error",
            "filePath",
            "success",
            "interrupted",
        ):
            if key in response:
                selected[key] = response[key]
        response = selected or {"keys": sorted(response.keys())[:20]}
    return bounded(response, MAX_RESULT_CHARS)


def tool_payload(hook_input: dict[str, Any], *, failed: bool | None = False) -> dict[str, Any]:
    name = str(hook_input.get("tool_name") or "unknown")
    tool_input = hook_input.get("tool_input") or {}
    if not isinstance(tool_input, dict):
        tool_input = {}
    payload: dict[str, Any] = {
        "tool": name,
        "failed": failed,
        "duration_ms": hook_input.get("duration_ms"),
        "agent_role": "subagent" if hook_input.get("agent_id") else "main",
    }
    if hook_input.get("agent_id"):
        payload["agent_id"] = bounded(hook_input["agent_id"], 200)
    if hook_input.get("agent_type"):
        payload["agent_type"] = bounded(hook_input["agent_type"], 200)

    if name in {"Read", "Write", "Edit", "MultiEdit", "NotebookEdit"}:
        path = tool_input.get("file_path") or tool_input.get("notebook_path")
        if path:
            payload["path"] = bounded(path, 1000)
        if name in {"Write", "Edit", "MultiEdit", "NotebookEdit"}:
            payload["mutation_chars"] = sum(
                len(str(tool_input.get(key, "")))
                for key in ("content", "new_string", "new_source", "edits")
            )
    elif name == "Bash" or "command" in tool_input:
        command = bounded(tool_input.get("command", ""), MAX_COMMAND_CHARS)
        payload["command"] = command
        payload["command_kind"] = (
            "test"
            if TEST_COMMAND_RE.search(command)
            else "build"
            if BUILD_COMMAND_RE.search(command)
            else "shell"
        )
        response = (
            hook_input.get("error") if failed else hook_input.get("tool_response")
        )
        payload["result_preview"] = _tool_result_preview(response)
    elif name in {"Grep", "Glob", "WebSearch", "WebFetch"}:
        for key in ("pattern", "path", "query", "url"):
            if tool_input.get(key):
                payload[key] = bounded(tool_input[key], 1000)
    else:
        payload["input_keys"] = sorted(tool_input.keys())[:20]
        if failed:
            payload["error"] = bounded(hook_input.get("error", ""), MAX_RESULT_CHARS)

    if failed and "error" not in payload:
        payload["error"] = bounded(hook_input.get("error", ""), MAX_RESULT_CHARS)
    return payload


def record_tool(
    store: EvidenceStore, hook_input: dict[str, Any], *, failed: bool | None = False
) -> None:
    session_id = _session_id(hook_input)
    repo = store.repo_for_session(session_id, hook_input.get("cwd"))
    payload = tool_payload(hook_input, failed=failed)
    if payload.get("path"):
        payload["repo_path"] = _repo_relative_path(repo, str(payload["path"]))
    store.record_event(
        repo,
        session_id,
        "tool_failure" if failed else "tool_result",
        payload,
    )


def record_subagent_start(
    store: EvidenceStore, hook_input: dict[str, Any]
) -> str:
    """Record a native subagent and reuse the main turn's retrieved memories."""
    session_id = _session_id(hook_input)
    repo = store.repo_for_session(session_id, hook_input.get("cwd"))
    agent_id = bounded(hook_input.get("agent_id", "unknown-agent"), 200)
    agent_type = bounded(hook_input.get("agent_type", "unknown-agent"), 200)
    context = combine_context(
        format_context(store.injected_memories(session_id, repo.identity))
    )
    first_start = store.start_subagent(
        repo, session_id, agent_id, agent_type, len(context)
    )
    store.record_event(
        repo,
        session_id,
        "subagent_start",
        {
            "agent_id": agent_id,
            "agent_type": agent_type,
            "context_chars": len(context) if first_start else 0,
            "repo_root": bounded(repo.root, 2000),
        },
    )
    return context if first_start else ""


def record_subagent_stop(store: EvidenceStore, hook_input: dict[str, Any]) -> None:
    session_id = _session_id(hook_input)
    repo = store.repo_for_session(session_id, hook_input.get("cwd"))
    agent_type = bounded(hook_input.get("agent_type", "unknown-agent"), 200)
    agent_id = bounded(hook_input.get("agent_id", ""), 200)
    final_message = redact(hook_input.get("last_assistant_message", "")).strip()
    transcript_path = bounded(hook_input.get("agent_transcript_path", ""), 2000)
    agent_id = store.stop_subagent(
        repo,
        session_id,
        agent_id,
        agent_type,
        transcript_path,
        final_message,
    )
    store.record_event(
        repo,
        session_id,
        "subagent_stop",
        {
            "agent_id": agent_id,
            "agent_type": agent_type,
            "transcript_path": transcript_path,
            "final_message": final_message,
        },
    )


def _ordered_unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _repo_relative_path(repo: RepoContext, value: str) -> str:
    value = str(value or "").strip()
    if not value:
        return ""
    try:
        path = Path(value)
        if path.is_absolute():
            try:
                return path.resolve().relative_to(Path(repo.root).resolve()).as_posix()
            except ValueError:
                return ""
    except (OSError, ValueError):
        pass
    return bounded(value, 1000)


def _render_command_lines(commands: list[dict[str, str]]) -> list[str]:
    lines = []
    for command in commands:
        line = f"- [{command['status']}/{command['kind']}] {command['command']}"
        if command["result"]:
            line += f" — {bounded(command['result'], 500).replace(chr(10), ' ')}"
        lines.append(line)
    return lines


def build_episode(
    repo: RepoContext,
    session_id: str,
    packet_id: str,
    events: list[dict[str, Any]],
    *,
    canonical_task: str = "",
    task_outcome: str = "",
) -> tuple[str, dict[str, Any]]:
    prompts = [
        redact(e["payload"].get("text", "")).strip()
        for e in events
        if e["kind"] == "user_prompt" and e["payload"].get("text")
    ]
    assistant_conclusions = [
        redact(e["payload"].get("text", "")).strip()
        for e in events
        if e["kind"] == "assistant_stop" and e["payload"].get("text")
    ]
    subagent_outcomes = [
        redact(e["payload"].get("final_message", "")).strip()
        for e in events
        if e["kind"] in {"subagent_stop", "sidekick_stop"} and e["payload"].get("final_message")
    ]
    tools = [
        e["payload"] for e in events if e["kind"] in {"tool_result", "tool_failure"}
        and e["payload"].get("agent_role", "main") == "main"
    ]
    read_paths = _ordered_unique(
        _repo_relative_path(repo, str(t.get("repo_path") or t.get("path", "")))
        for t in tools
        if t.get("tool") == "Read"
    )
    modified_paths = _ordered_unique(
        _repo_relative_path(repo, str(t.get("repo_path") or t.get("path", "")))
        for t in tools
        if t.get("tool") in {"Write", "Edit", "MultiEdit", "NotebookEdit"}
    )
    searches = [
        {key: t[key] for key in ("tool", "pattern", "path", "query", "url") if key in t}
        for t in tools
        if t.get("tool") in {"Grep", "Glob", "WebSearch", "WebFetch"}
    ]
    commands = [
        {
            "command": t.get("command", ""),
            "kind": t.get("command_kind", "shell"),
            "status": "unknown" if t.get("failed", False) is None else "failed" if t.get("failed") else "succeeded",
            "result": t.get("result_preview", ""),
        }
        for t in tools
        if t.get("command")
    ]

    task = bounded(canonical_task or (prompts[0] if prompts else ""), 4000)
    conclusion = redact(assistant_conclusions[-1] if assistant_conclusions else "").strip()
    outcome = bounded(task_outcome, 2000)

    extraction_messages: list[dict[str, str]] = []
    pending_user_messages: list[dict[str, str]] = []
    if task and not prompts:
        pending_user_messages.append({"role": "user", "content": task})
    for event in events:
        if event["kind"] == "user_prompt" and event["payload"].get("text"):
            pending_user_messages.append(
                {
                    "role": "user",
                    "content": redact(event["payload"].get("text", "")).strip(),
                }
            )
        elif event["kind"] == "assistant_stop":
            transcript_messages = event["payload"].get("transcript_messages") or []
            if isinstance(transcript_messages, list) and transcript_messages:
                transcript_users = {
                    redact(message.get("content") or "").strip()
                    for message in transcript_messages
                    if isinstance(message, dict) and message.get("role") == "user"
                }
                extraction_messages.extend(
                    message
                    for message in pending_user_messages
                    if message["content"].strip() not in transcript_users
                )
                extraction_messages.extend(
                    {
                        "role": str(message.get("role") or ""),
                        "content": redact(message.get("content") or "").strip(),
                    }
                    for message in transcript_messages
                    if isinstance(message, dict)
                    and message.get("role") in {"user", "assistant"}
                    and message.get("content")
                )
            else:
                extraction_messages.extend(pending_user_messages)
                if event["payload"].get("text"):
                    extraction_messages.append(
                        {
                            "role": "assistant",
                            "content": redact(event["payload"].get("text", "")).strip(),
                        }
                    )
            pending_user_messages = []
        elif event["kind"] in {"subagent_stop", "sidekick_stop"}:
            pass
    extraction_messages.extend(pending_user_messages)

    structured = {
        "packet_id": packet_id,
        "repo": repo.identity,
        "app_id": repo.app_id,
        "session_id": session_id,
        "branch": repo.branch,
        "head_sha": repo.head_sha,
        "task": task,
        "task_outcome": outcome,
        "assistant_conclusion": conclusion,
        "user_messages": prompts,
        "assistant_outcomes": assistant_conclusions,
        "subagent_outcomes": subagent_outcomes,
        "extraction_messages": extraction_messages,
        "files_read": read_paths[:50],
        "files_modified": modified_paths[:50],
        "searches": searches[-30:],
        "commands": commands[-30:],
    }
    lines = ["Coding-session episode"]
    if task:
        lines.extend(["", "Task:", task])
    if modified_paths:
        lines.extend(
            ["", "Files modified:", *[f"- {path}" for path in modified_paths[:50]]]
        )
    if read_paths:
        lines.extend(["", "Files read:", *[f"- {path}" for path in read_paths[:50]]])
    if commands:
        lines.append("")
        lines.append("Observed commands:")
        lines.extend(_render_command_lines(commands[-30:]))
    if searches:
        lines.extend(
            [
                "",
                "Observed searches:",
                *[
                    f"- {json.dumps(item, ensure_ascii=False, sort_keys=True)}"
                    for item in searches[-20:]
                ],
            ]
        )
    if conclusion:
        lines.extend(["", "Agent conclusion:", conclusion])
    if outcome:
        lines.extend(["", "Task outcome:", outcome])
    lines.extend(
        [
            "",
            f"Provenance: repo={repo.identity}; branch={repo.branch}; head={repo.head_sha}; packet={packet_id}",
        ]
    )
    content = "\n".join(lines)
    return bounded(content, MAX_EPISODE_CHARS), structured


def build_semantic_evidence(structured: dict[str, Any]) -> str:
    """Format changed paths for memory extraction.

    Test and build results remain in the local evidence store for diagnostics,
    but are not useful repository knowledge by default and should not steer
    memory extraction toward transient verification details.
    """
    modified_paths = [
        bounded(path, 500) for path in structured.get("files_modified", [])[:20]
    ]
    commands = structured.get("commands") or []
    if not any(command.get("status") == "failed" for command in commands):
        commands = []

    if not modified_paths and not commands:
        return ""

    lines = ["Additional repository details from this session"]
    if modified_paths:
        lines.extend(
            [
                "",
                "Changed paths:",
                *[f"- {path}" for path in modified_paths],
            ]
        )
    if commands:
        lines.extend(["", "Commands run in this session:", *_render_command_lines(commands)])
    return bounded("\n".join(lines), 8000)


def build_extraction_messages(structured: dict[str, Any]) -> list[dict[str, str]]:
    """Build the session messages sent to the NeatMem server for extraction."""
    evidence = build_semantic_evidence(structured)
    messages = [
        {"role": message["role"], "content": redact(message["content"]).strip()}
        for message in structured.get("extraction_messages", [])
        if message.get("role") in {"user", "assistant"} and message.get("content")
    ]
    if evidence:
        for message in reversed(messages):
            if message["role"] == "assistant":
                message["content"] = f"{message['content']}\n\n{evidence}"
                break
        else:
            messages.append({"role": "assistant", "content": evidence})

    return messages


def _estimated_tokens(value: str) -> int:
    """Conservatively estimate tokens without adding a tokenizer dependency."""
    ascii_chars = sum(ord(char) < 128 for char in value)
    return math.ceil((ascii_chars * 0.4) + (len(value) - ascii_chars))


def _message_tokens(messages: list[dict[str, str]]) -> int:
    return _estimated_tokens(json.dumps(messages, ensure_ascii=False))


def _is_agent_assignment(message: dict[str, str]) -> bool:
    return message.get("role") == "assistant" and message.get(
        "content", ""
    ).startswith("Subagent assignment (")


def _is_agent_response(message: dict[str, str]) -> bool:
    return message.get("role") == "assistant" and message.get(
        "content", ""
    ).startswith("Subagent response (")


def extraction_message_batches(
    messages: list[dict[str, str]],
    *,
    max_tokens: int = MAX_EXTRACTION_INPUT_TOKENS,
) -> list[list[dict[str, str]]]:
    """Keep exchanges together when possible; split oversized messages to enforce the request budget."""
    if not messages or _message_tokens(messages) <= max_tokens:
        return [messages]

    exchanges: list[list[dict[str, str]]] = []
    exchange: list[dict[str, str]] = []
    for message in messages:
        if message.get("role") == "user" and exchange:
            exchanges.append(exchange)
            exchange = []
        exchange.append(message)
    if exchange:
        exchanges.append(exchange)

    units: list[list[dict[str, str]]] = []
    for exchange in exchanges:
        if _message_tokens(exchange) <= max_tokens:
            units.append(exchange)
            continue
        index = 0
        while index < len(exchange):
            message = exchange[index]
            if (
                _is_agent_assignment(message)
                and index + 1 < len(exchange)
                and _is_agent_response(exchange[index + 1])
            ):
                units.append(exchange[index : index + 2])
                index += 2
            else:
                units.append([message])
                index += 1

    bounded_units: list[list[dict[str, str]]] = []
    for unit in units:
        if _message_tokens(unit) <= max_tokens:
            bounded_units.append(unit)
            continue
        for message in unit:
            remaining = message["content"]
            while remaining:
                low, high = 0, len(remaining)
                while low < high:
                    middle = (low + high + 1) // 2
                    if _message_tokens([{**message, "content": remaining[:middle]}]) <= max_tokens:
                        low = middle
                    else:
                        high = middle - 1
                if low == 0:
                    raise ValueError("Extraction token budget cannot fit a message")
                bounded_units.append([{**message, "content": remaining[:low]}])
                remaining = remaining[low:]

    batches: list[list[dict[str, str]]] = []
    batch: list[dict[str, str]] = []
    for unit in bounded_units:
        candidate = [*batch, *unit]
        if batch and _message_tokens(candidate) > max_tokens:
            batches.append(batch)
            batch = list(unit)
        else:
            batch = candidate
    if batch:
        batches.append(batch)
    return batches


def _auth_headers(key: str) -> dict[str, str]:
    """NeatMem is a local server with no auth by default; send the key only
    when one is configured (e.g. deployments behind an authenticating proxy)."""
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def _request_json(
    url: str, key: str, payload: dict[str, Any], timeout: float
) -> tuple[dict[str, Any] | list[Any], int, int]:
    raw = json.dumps(payload, ensure_ascii=False).encode()
    request = urllib.request.Request(
        url,
        data=raw,
        headers=_auth_headers(key),
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        response_raw = response.read()
    parsed = json.loads(response_raw or b"{}")
    return parsed, len(raw), len(response_raw)


def _request_json_with_network_retry(
    url: str, key: str, payload: dict[str, Any], timeout: float
) -> tuple[dict[str, Any] | list[Any], int, int]:
    """Retry one transient connection failure without retrying API responses."""
    try:
        return _request_json(url, key, payload, timeout)
    except urllib.error.HTTPError:
        raise
    except (urllib.error.URLError, TimeoutError, OSError):
        time.sleep(0.25)
        return _request_json(url, key, payload, timeout)


def _get_json(
    url: str, key: str, timeout: float
) -> tuple[dict[str, Any] | list[Any], int]:
    request = urllib.request.Request(
        url,
        headers=_auth_headers(key),
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        response_raw = response.read()
    return json.loads(response_raw or b"{}"), len(response_raw)


def touch_handoff_heartbeat() -> None:
    """Mark the worker's handoff file alive so recovery does not relaunch it."""
    path = os.environ.get("NEATMEM_CODE_HANDOFF_PATH", "")
    if not path:
        return
    try:
        os.utime(path)
    except OSError:
        pass


def flush_session(
    store: EvidenceStore, hook_input: dict[str, Any], reason: str
) -> dict[str, Any]:
    """Send unflushed events to the NeatMem server and extract them now.

    Two-step sync flow: /v1/messages/add/ enqueues the session messages
    (store-only), then /v1/messages/flush/ forces immediate extraction for
    this session's scope. No event polling: flush returns when extraction
    is done.
    """
    session_id = _session_id(hook_input)
    if session_id == "unknown-session":
        return {"status": "error", "reason": "no-session-id"}
    repo = store.repo_for_session(session_id, hook_input.get("cwd"))
    prepared = store.prepare_flush(repo, session_id, reason)
    if prepared is None:
        return {"status": "nothing-to-flush"}
    packet_id, events = prepared

    _, structured = build_episode(
        repo,
        session_id,
        packet_id,
        events,
        canonical_task=bounded(hook_input.get("task", ""), 4000),
        task_outcome=bounded(hook_input.get("task_outcome", ""), 2000),
    )

    write_user = _scope_value(user_id())
    write_project = _scope_value(repo.project_id)
    if not write_user or not write_project:
        return {"status": "error", "reason": "wildcard-scope"}

    key = api_key()
    server_url = api_url()
    flush_timeout = float(os.environ.get("NEATMEM_FLUSH_TIMEOUT_SECONDS", "120"))

    started = time.perf_counter()
    try:
        message_batches = [
            batch
            for batch in extraction_message_batches(
                build_extraction_messages(structured)
            )
            if batch
        ]
        if not message_batches:
            store.update_flush(packet_id, status="succeeded", error="")
            return {"status": "nothing-to-flush", "packet_id": packet_id}

        request_chars = 0
        response_chars = 0
        message_count = 0
        for messages in message_batches:
            _, req, resp = _request_json_with_network_retry(
                f"{server_url}/v1/messages/add/",
                key,
                {
                    "messages": messages,
                    "user_id": write_user,
                    "agent_id": write_project,
                    "run_id": session_id,
                },
                15,
            )
            request_chars += req
            response_chars += resp
            message_count += len(messages)

        touch_handoff_heartbeat()
        flush_response, req, resp = _request_json_with_network_retry(
            f"{server_url}/v1/messages/flush/",
            key,
            {"user_id": write_user, "agent_id": write_project, "run_id": session_id},
            flush_timeout,
        )
        request_chars += req
        response_chars += resp
        # flush_scope returns {"batches", "extracted_count", "last_processed_seq"}
        # where extracted_count is the number of messages extracted.
        extracted_count = 0
        if isinstance(flush_response, dict):
            extracted_count = int(flush_response.get("extracted_count", 0) or 0)

        elapsed = (time.perf_counter() - started) * 1000
        store.update_flush(packet_id, status="succeeded", error="")
        store.operation(
            repo,
            session_id,
            "flush",
            elapsed,
            True,
            item_count=extracted_count,
            request_chars=request_chars,
            response_chars=response_chars,
        )
        effective_reason = str(
            (store.flush_record(packet_id) or {}).get("reason") or reason
        )
        if (
            store.has_unflushed_events(repo.identity, session_id)
            and not store.unflushed_starts_with_session_start(
                repo.identity, session_id
            )
            and (
                effective_reason != "periodic"
                or store.checkpoint_due(repo.identity, session_id)
            )
        ):
            return flush_session(store, hook_input, effective_reason)
        return {
            "status": "succeeded",
            "packet_id": packet_id,
            "message_count": message_count,
            "extracted_count": extracted_count,
            "duration_ms": round(elapsed, 2),
        }
    except Exception as exc:  # hooks must fail open
        elapsed = (time.perf_counter() - started) * 1000
        error = bounded(str(exc), 1000)
        store.update_flush(packet_id, status="error", error=error)
        store.operation(repo, session_id, "flush", elapsed, False, error=error)
        return {"status": "error", "packet_id": packet_id, "error": error}


def checkpoint_session(
    store: EvidenceStore, hook_input: dict[str, Any], reason: str
) -> dict[str, Any]:
    """Run remote extraction at a durable boundary."""
    return flush_session(store, hook_input, reason)



def search_memories(
    store: EvidenceStore | None,
    repo: RepoContext,
    session_id: str | None,
    query: str,
    *,
    top_k: int | None = None,
    scope: str | None = None,
    run_id: str | None = None,
    operation: str = "search",
    timeout: float = 5,
) -> MemorySearchResult:
    key = api_key()
    if not query.strip():
        return MemorySearchResult(False, 0, 0, [])
    search_once = os.environ.get(
        "NEATMEM_CODE_SEARCH_ONCE_PER_SESSION", "false"
    ).lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    track_session = store is not None and bool(session_id)
    if (
        search_once
        and track_session
        and store.has_operation(repo.identity, session_id, "search")
    ):
        return MemorySearchResult(False, 0, 0, [])

    result_limit = min(
        max(
            top_k
            if top_k is not None
            else _int_option("top_k", "NEATMEM_CODE_TOP_K", 3),
            1,
        ),
        20,
    )
    user, project = _scope_value(user_id()), _scope_value(repo.project_id)
    if not user or not project:
        return MemorySearchResult(False, 0, 0, [])
    filters = _search_filters(user, repo, resolve_search_scope(scope))
    if run_id:
        filters = {**filters, "run_id": run_id}
    payload = {
        "query": query,
        "filters": filters,
        "top_k": result_limit,
    }
    url = api_url() + "/v2/memories/search/"
    started = time.perf_counter()
    try:
        response, request_chars, response_chars = _request_json_with_network_retry(
            url, key, payload, timeout
        )
        memories = (
            response if isinstance(response, list) else response.get("results", [])
        )
        memories = [
            memory for memory in memories if isinstance(memory, dict)
        ][:result_limit]
        if track_session:
            returned_memories = store.unseen(session_id, repo.identity, memories)
            store.mark_injected(session_id, repo.identity, returned_memories)
            already_shown_count = len(memories) - len(returned_memories)
        else:
            returned_memories = memories
            already_shown_count = 0
        elapsed = (time.perf_counter() - started) * 1000
        if track_session:
            store.operation(
                repo,
                session_id,
                operation,
                elapsed,
                True,
                item_count=len(returned_memories),
                request_chars=request_chars,
                response_chars=response_chars,
            )
        return MemorySearchResult(
            succeeded=True,
            matched_count=len(memories),
            already_shown_count=already_shown_count,
            memories=returned_memories,
        )
    except Exception as exc:
        elapsed = (time.perf_counter() - started) * 1000
        if track_session:
            store.operation(repo, session_id, operation, elapsed, False, error=str(exc))
        return MemorySearchResult(False, 0, 0, [])


def format_context(
    memories: list[dict[str, Any]],
    heading: str = "Relevant repository memories:",
) -> str:
    if not memories:
        return ""
    limit = min(
        max(
            _int_option(
                "max_context_chars",
                "NEATMEM_CODE_MAX_CONTEXT_CHARS",
                DEFAULT_MAX_CONTEXT_CHARS,
            ),
            1000,
        ),
        10000,
    )
    lines = [heading] if heading else []
    for memory in memories:
        text = re.sub(
            r"\s+",
            " ",
            redact(memory.get("memory") or memory.get("text") or ""),
        )
        text = text.strip()
        if not text:
            continue
        branch = str((memory.get("metadata") or {}).get("branch") or "").strip()
        branch_label = (
            f" [learnt on branch {branch}]"
            if branch.casefold() not in {"", "main", "master", "unknown", "detached"}
            else ""
        )
        number = len(lines) if heading else len(lines) + 1
        entry = f"{number}. {text}{branch_label}"
        candidate = "\n".join([*lines, entry])
        if len(candidate) <= limit:
            lines.append(entry)
            continue
        if not lines or (heading and len(lines) == 1):
            prefix = f"{number}. "
            suffix = f"…{branch_label}"
            available = (
                limit
                - len("\n".join(lines))
                - (1 if lines else 0)
                - len(prefix)
                - len(suffix)
            )
            if available > 0:
                lines.append(prefix + text[:available].rstrip() + suffix)
        break
    minimum_lines = 2 if heading else 1
    return "\n".join(lines) if len(lines) >= minimum_lines else ""


def format_search_result(result: MemorySearchResult) -> str:
    """Return only the text the coding agent needs from an explicit memory search."""
    if not result.succeeded:
        return "Memory search failed."
    if result.memories:
        rendered = format_context(result.memories, heading="")
        if rendered:
            return rendered
    return "No matching memories found."


def combine_context(*contexts: str) -> str:
    """Combine memory sources under one hard budget without repeated lines."""
    seen: set[str] = set()
    lines: list[str] = []
    for context in contexts:
        for line in str(context or "").splitlines():
            normalized = re.sub(r"\s+", " ", line).strip().casefold()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            lines.append(line.rstrip())
    limit = min(
        max(
            _int_option(
                "max_context_chars",
                "NEATMEM_CODE_MAX_CONTEXT_CHARS",
                DEFAULT_MAX_CONTEXT_CHARS,
            ),
            1000,
        ),
        10000,
    )
    return bounded("\n".join(lines), limit) if lines else ""


def _scoped_memory_ids(
    api_url: str, key: str, user: str, repo: RepoContext, include_project: bool
) -> list[str]:
    """List this user's memory ids for this repository, plus the shared project memory when asked."""
    ids: list[str] = []
    seen: set[str] = set()
    _collect_memory_ids(
        api_url, key,
        {"user_id": user, "agent_id": repo.project_id},
        ids, seen,
    )
    if include_project:
        _collect_memory_ids(
            api_url, key, {"agent_id": repo.project_id}, ids, seen,
        )
    return ids


def _collect_memory_ids(
    api_url: str,
    key: str,
    filters: dict[str, Any],
    ids: list[str],
    seen: set[str],
) -> None:
    """Page through one list filter; the list endpoint returns nothing for an OR whose user branch has no memories."""
    for page in range(1, FORGET_MAX_PAGES + 1):
        parsed, _, _ = _request_json(
            f"{api_url}/v2/memories/?page={page}&page_size={FORGET_PAGE_SIZE}",
            key,
            {"filters": filters},
            15,
        )
        items = parsed.get("results") if isinstance(parsed, dict) else parsed
        if not isinstance(items, list) or not items:
            break
        for item in items:
            if not isinstance(item, dict):
                continue
            memory_id = str(item.get("id", ""))
            if not memory_id or memory_id in seen:
                continue
            seen.add(memory_id)
            ids.append(memory_id)
        if len(items) < FORGET_PAGE_SIZE:
            break


def _delete_memory(api_url: str, key: str, memory_id: str) -> bool:
    request = urllib.request.Request(
        f"{api_url}/v1/memories/{urllib.parse.quote(memory_id)}/",
        headers=_auth_headers(key),
        method="DELETE",
    )
    try:
        with urllib.request.urlopen(request, timeout=15):
            return True
    except Exception:
        return False


def forget_remote_repo(
    repo: RepoContext, *, include_project_memory: bool = False
) -> dict[str, Any]:
    """Delete this user's memories for this repository; project memory is shared, so only on request."""
    key = api_key()
    user = _scope_value(user_id())
    if not user or not _scope_value(repo.project_id):
        return {
            "status": "error",
            "error": "Refusing to forget: the user or repository scope is a wildcard",
        }
    server_url = api_url()
    try:
        memory_ids = _scoped_memory_ids(server_url, key, user, repo, include_project_memory)
    except Exception as exc:
        return {"status": "error", "error": bounded(str(exc), 1000)}
    deleted = sum(_delete_memory(server_url, key, memory_id) for memory_id in memory_ids)
    failed = len(memory_ids) - deleted
    if failed:
        return {
            "status": "partial",
            "deleted": deleted,
            "failed": failed,
            "error": f"{failed} of {len(memory_ids)} memories could not be deleted",
        }
    return {"status": "deleted", "deleted": deleted}


def _doctor_server_connection(repo: RepoContext) -> dict[str, Any]:
    """Verify the NeatMem server is reachable with a read-only ping."""
    url = api_url()
    started = time.perf_counter()
    try:
        _get_json(f"{url}/v1/ping/", api_key(), 5)
    except Exception as exc:
        return {"ok": False, "detail": bounded(str(exc), 300)}
    elapsed = (time.perf_counter() - started) * 1000
    return {"ok": True, "detail": f"connected ({elapsed:.0f} ms)"}


def _doctor_user_id() -> dict[str, Any]:
    """Flag a configured user ID the plugin refuses, since the silent fallback surprises people."""
    configured = _plugin_option("user_id", "NEATMEM_USER_ID")
    if configured and not _scope_value(configured):
        return {
            "ok": False,
            "detail": f"configured user_id {configured!r} is a wildcard; using {user_id()!r}",
        }
    return {"ok": True, "detail": user_id()}


def doctor(cwd: str | None = None) -> dict[str, Any]:
    repo = resolve_repo(cwd)
    directory = data_dir()
    directory.mkdir(parents=True, exist_ok=True)
    server_url = api_url()
    checks: dict[str, dict[str, Any]] = {
        "python": {
            "ok": tuple(sys.version_info[:2]) >= (3, 10),
            "detail": f"{sys.version_info.major}.{sys.version_info.minor}",
        },
        "data_directory": {
            "ok": os.access(directory, os.W_OK),
            "detail": str(directory),
        },
        "api_url": {
            "ok": True,
            "detail": server_url,
        },
        "repository": {
            "ok": bool(repo.identity),
            "detail": repo.identity,
        },
        "user_id": _doctor_user_id(),
        "server_connection": _doctor_server_connection(repo),
    }
    return {
        "ok": all(bool(value["ok"]) for value in checks.values()),
        "plugin_version": PLUGIN_VERSION,
        "repo_id": repo.identity,
        "app_id": repo.app_id,
        "user_id": user_id(),
        "checks": checks,
    }
