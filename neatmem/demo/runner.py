"""Demo runner: server lifecycle + case replay.

Spawns an internal `neatmem serve` child (in-memory qdrant + sqlite by
default, released on exit), replays the case, and collects mechanism
frames from the child's request-scoped log lines.

Replay semantics:
- existing_memories are written verbatim via add(infer=false)
- each session is replayed turn by turn (a turn ends at an assistant
  message); each turn issues one add with the session's messages up to
  that turn — the same cumulative-add shape as agent auto-capture
- reps share one server; each rep starts with a full wipe of the demo
  user's memories and message history
"""

import argparse
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

from neatmem.cli import add_serve_arguments, serve_flags_to_env
from neatmem.demo import output
from neatmem.demo.schema import build_inline_case, load_case

DEMO_USER_ID = "demo-user"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neatmem demo",
        description="Replay a demo case against a fresh in-memory memory server.",
    )
    parser.add_argument("source", nargs="?",
                        help="Case file path, or '-' to read the case JSON from stdin")
    parser.add_argument("--existing", action="extend", nargs="+", default=None,
                        metavar="TEXT", help="Pre-existing memory, written verbatim "
                        "(repeatable). Mutually exclusive with source.")
    parser.add_argument("--say", action="extend", nargs="+", default=None,
                        metavar="TEXT", help="User message; each value is one sequential "
                        "add through the full pipeline (repeatable). "
                        "Mutually exclusive with source.")
    parser.add_argument("--reps", type=int, default=1,
                        help="Repeat the whole case N times from a fresh store (default 1)")
    parser.add_argument("--output", metavar="PATH",
                        help="Write a markdown run record to exactly this path "
                        "(default: print only)")
    add_serve_arguments(parser)
    return parser


class _Server:
    """Internal serve child; log goes to a temp file read incrementally."""

    def __init__(self, env: dict, port: int):
        self.port = port
        self.base = f"http://127.0.0.1:{port}"
        self._tmpdir = tempfile.TemporaryDirectory(prefix="neatmem-demo-")
        self.log_path = Path(self._tmpdir.name) / "server.log"
        cmd = [sys.executable, "-m", "neatmem.cli", "serve",
               "--host", "127.0.0.1", "--port", str(port)]
        self._log = open(self.log_path, "ab")
        self.proc = subprocess.Popen(
            cmd, env=env, stdout=self._log, stderr=subprocess.STDOUT,
            start_new_session=True)
        self._offset = 0

    def wait_health(self, tries: int = 40, interval: float = 3.0) -> None:
        for _ in range(tries):
            try:
                r = httpx.get(f"{self.base}/health", timeout=5)
                if r.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            if self.proc.poll() is not None:
                tail = self.log_path.read_text(errors="replace")[-3000:]
                raise SystemExit(f"neatmem demo: server died during startup:\n{tail}")
            time.sleep(interval)
        tail = self.log_path.read_text(errors="replace")[-3000:]
        raise SystemExit(f"neatmem demo: server health check timeout:\n{tail}")

    def new_frames(self) -> list:
        """Request-scoped log lines appended since the last call."""
        self._log.flush()
        with open(self.log_path, "r", errors="replace") as f:
            f.seek(self._offset)
            text = f.read()
            self._offset = f.tell()
        lines = []
        for line in text.splitlines():
            # uvicorn/loguru prefix, keep only request-scoped entries
            if output.REQ_LINE_RE.search(line):
                lines.append(line.strip())
        return lines

    def stop(self) -> None:
        if self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait(timeout=10)
        self._log.close()
        self._tmpdir.cleanup()


def _wipe(client: httpx.Client, base: str) -> None:
    client.delete(f"{base}/v1/memories/", params={"user_id": DEMO_USER_ID}, timeout=30)
    client.post(f"{base}/v1/messages/delete/", json={"user_id": DEMO_USER_ID}, timeout=30)


def _write_existing(client: httpx.Client, base: str, existing: list) -> None:
    for em in existing:
        payload = {
            "messages": [{"role": "user", "content": em["text"]}],
            "user_id": DEMO_USER_ID,
            "infer": False,
        }
        if em.get("created_at"):
            payload["metadata"] = {"created_at": em["created_at"]}
        r = client.post(f"{base}/v1/memories/", json=payload, timeout=60)
        r.raise_for_status()


def _turns(messages: list) -> list:
    """Split a session into turns; a turn ends at an assistant message."""
    turns, current = [], []
    for msg in messages:
        current.append(msg)
        if msg.get("role") == "assistant":
            turns.append(current)
            current = []
    if current:
        turns.append(current)
    return turns


def _replay_session(client: httpx.Client, base: str, server: _Server,
                    session: dict) -> list:
    """Replay one session as cumulative per-turn adds; return step records."""
    server.new_frames()  # discard setup noise so frames align to adds
    steps = []
    turns = _turns(session["messages"])
    cumulative = []
    for t_idx, turn in enumerate(turns, 1):
        cumulative.extend(turn)
        r = client.post(f"{base}/v1/memories/",
                        json={"messages": cumulative, "user_id": DEMO_USER_ID},
                        timeout=300)
        r.raise_for_status()
        steps.append({
            "label": f"session {session['id']} turn {t_idx}/{len(turns)}",
            "frames": server.new_frames(),
            "results": r.json(),
        })
    return steps


def _final_store(client: httpx.Client, base: str) -> list:
    r = client.post(f"{base}/v2/memories/",
                    json={"filters": {"user_id": DEMO_USER_ID}, "page_size": 1000},
                    timeout=30)
    r.raise_for_status()
    data = r.json()
    return data.get("results", data if isinstance(data, list) else [])


def _rep_stats(steps: list, final: list) -> dict:
    return {
        "final_count": len(final),
        "added": sum(len(s["results"].get("results", [])) for s in steps),
        "updated": sum(len(s["results"].get("duplicates", [])) for s in steps),
        "merged": sum(len(s["results"].get("merged", [])) for s in steps),
    }


def run_demo(argv: list) -> None:
    args = _build_parser().parse_args(argv)

    inline = bool(args.existing or args.say)
    if args.source and inline:
        raise SystemExit("neatmem demo: error: source and --existing/--say are "
                         "mutually exclusive")
    if not args.source and not inline:
        raise SystemExit("neatmem demo: error: provide a case file, '-', or "
                         "--existing/--say")
    if args.reps < 1:
        raise SystemExit("neatmem demo: error: --reps must be >= 1")

    case = load_case(args.source) if args.source else build_inline_case(args.existing, args.say)

    arm_env = serve_flags_to_env(args)
    env = {**os.environ, **arm_env}
    if args.vector_db_path is None and args.vector_db_url is None:
        env["QDRANT_PATH"] = ":memory:"
        env["QDRANT_HOST"] = ""
    if args.history_db_path is None:
        env["HISTORY_DB_PATH"] = ":memory:"

    port = args.port or _free_port()
    output.print_plan(case, args.reps, arm_env)
    print(f"Server: internal, 127.0.0.1:{port} "
          f"({'memory' if env.get('QDRANT_PATH') == ':memory:' else 'custom db'})")

    server = _Server(env, port)
    try:
        server.wait_health()
        reps_data = []
        with httpx.Client() as client:
            for rep_idx in range(1, args.reps + 1):
                if args.reps > 1:
                    print(f"\n=== rep {rep_idx}/{args.reps} ===")
                _wipe(client, server.base)
                _write_existing(client, server.base, case.existing_memories)
                steps = []
                for session in case.sessions:
                    steps.extend(_replay_session(client, server.base, server, session))
                final = _final_store(client, server.base)
                rep = {"idx": rep_idx, "steps": steps, "final": final,
                       "stats": _rep_stats(steps, final)}
                reps_data.append(rep)
                if args.reps == 1:
                    for step in steps:
                        print(f"\n-- {step['label']} --")
                        output.print_frames(step["frames"], step["results"])
                    print()
                    output.print_final_store(final)
                else:
                    output.print_compact_rep(rep)
        if args.reps > 1:
            output.print_summary(reps_data)
        if args.output:
            from dotenv import dotenv_values
            file_env = dotenv_values(args.env_file or ".env")
            llm_label = env.get("LLM_MODEL") or file_env.get("LLM_MODEL") or "default"
            path = output.write_output(case, reps_data, arm_env, argv, llm_label, args.output)
            print(f"\nRecord written: {path}")
    finally:
        server.stop()
