"""Configure harness identity for the test suite."""

from __future__ import annotations

import sys
from pathlib import Path

HOST_ROOT = Path(__file__).resolve().parents[1]
_core = HOST_ROOT / "core"
sys.path.insert(0, str(_core))

import hook_runner  # noqa: E402,F401
from memory_core import configure_harness  # noqa: E402

configure_harness("claude-code", data_dir_name="claude-code-plugin", source_tag="claude_code_plugin")
