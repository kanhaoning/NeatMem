"""Demo case schema: load/validate case documents and build inline cases.

Case document (single JSON):

    {
      "name": "...",                     # optional human title
      "description": "...",              # optional
      "existing_memories": [             # optional; written verbatim (infer=false)
        {"text": "...", "created_at": "2026-08-20"}
      ],
      "sessions": [                      # ordered; each replayed via the pipeline
        {"id": "s1",                     # optional, auto-generated s1/s2/...
         "messages": [{"role": "user", "content": "..."}]}
      ]
    }
"""

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_ALLOWED_ROLES = {"user", "assistant", "system"}


@dataclass
class DemoCase:
    slug: str
    name: str
    description: str
    existing_memories: list = field(default_factory=list)
    sessions: list = field(default_factory=list)  # [{"id": str, "messages": [...]}]


def _fail(msg: str) -> "SystemExit":
    return SystemExit(f"neatmem demo: error: {msg}")


def _validate_message(m: Any, where: str) -> dict:
    if not isinstance(m, dict) or "role" not in m or "content" not in m:
        raise _fail(f"{where}: message must be an object with role and content")
    if m["role"] not in _ALLOWED_ROLES:
        raise _fail(f"{where}: unsupported role {m['role']!r} (allowed: user/assistant/system)")
    if not isinstance(m["content"], str) or not m["content"].strip():
        raise _fail(f"{where}: message content must be a non-empty string")
    return {"role": m["role"], "content": m["content"]}


def _normalize(doc: dict, slug: str) -> DemoCase:
    if not isinstance(doc, dict):
        raise _fail("case document must be a JSON object")

    existing = []
    for i, em in enumerate(doc.get("existing_memories") or [], 1):
        if not isinstance(em, dict) or not isinstance(em.get("text"), str) or not em["text"].strip():
            raise _fail(f"existing_memories[{i}]: must be an object with non-empty text")
        item = {"text": em["text"]}
        if em.get("created_at"):
            item["created_at"] = str(em["created_at"])
        existing.append(item)

    sessions = []
    for i, s in enumerate(doc.get("sessions") or [], 1):
        if not isinstance(s, dict) or not s.get("messages"):
            raise _fail(f"sessions[{i}]: must be an object with a non-empty messages list")
        sid = s.get("id") or f"s{i}"
        sessions.append({
            "id": str(sid),
            "messages": [_validate_message(m, f"sessions[{i}]") for m in s["messages"]],
        })

    if not sessions and not existing:
        raise _fail("case has neither sessions nor existing_memories (nothing to run)")

    return DemoCase(
        slug=slug,
        name=str(doc.get("name") or slug),
        description=str(doc.get("description") or ""),
        existing_memories=existing,
        sessions=sessions,
    )


def load_case(source: str) -> DemoCase:
    """Load a case from a file path or '-' (stdin)."""
    if source == "-":
        slug = "stdin"
        try:
            doc = json.load(sys.stdin)
        except json.JSONDecodeError as e:
            raise _fail(f"stdin: invalid JSON: {e}")
    else:
        path = Path(source)
        if not path.exists():
            raise _fail(f"case file not found: {source}")
        slug = path.stem
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise _fail(f"{source}: invalid JSON: {e}")
    if not _SLUG_RE.match(slug):
        # slug lands in the --output filename; keep it filesystem-clean
        slug = re.sub(r"[^a-z0-9-]+", "-", slug.lower()).strip("-") or "case"
    return _normalize(doc, slug)


def build_inline_case(existing: list, say: list) -> DemoCase:
    """--existing/--say sugar: same document shape, built in memory.

    Each --say value becomes its own session (one sequential add per value);
    --existing values become existing_memories (verbatim writes)."""
    doc = {
        "name": "inline demo",
        "existing_memories": [{"text": t} for t in (existing or [])],
        "sessions": [
            {"messages": [{"role": "user", "content": t}]} for t in (say or [])
        ],
    }
    return _normalize(doc, "inline")
