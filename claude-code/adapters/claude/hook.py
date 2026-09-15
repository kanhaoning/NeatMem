#!/usr/bin/env python3
"""Claude Code hooks for Mem0."""

from __future__ import annotations

import sys
from pathlib import Path

_here = Path(__file__).resolve()
_bundled_core = _here.parents[2] / "core"
_core_dir = _bundled_core if (_bundled_core / "memory_core.py").is_file() else _bundled_core / "python"
sys.path.insert(0, str(_core_dir))
sys.path.insert(0, str(_here.parent))

import hook_runner  # noqa: E402
from memory_core import (  # noqa: E402
    configure_harness,
    record_tool,
)
from transcript import record_stop  # noqa: E402

configure_harness("claude-code", data_dir_name="claude-code-plugin", source_tag="claude_code_plugin")


if __name__ == "__main__":
    hook_runner.entry_point(
        record_stop_fn=record_stop,
        extra_actions={
            "post-tool-failure": lambda s, h: record_tool(s, h, failed=True),
        },
        data_dir_env="NEATMEM_CODE_DATA_DIR",
        automatic_flush_reasons={"session-end", "pre-compact"},
    )
