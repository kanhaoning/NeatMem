#!/usr/bin/env python3
"""Claude Code hooks for NeatMem."""

from __future__ import annotations

import sys
from pathlib import Path

_here = Path(__file__).resolve()
_bundled_core = _here.parents[2] / "core"
_core_dir = _bundled_core if (_bundled_core / "memory_core.py").is_file() else _bundled_core / "python"
sys.path.insert(0, str(_core_dir))
sys.path.insert(0, str(_here.parent))

import hook_runner  # noqa: E402
from memory_core import record_tool  # noqa: E402
from transcript import record_stop, periodic_reminder_after_tool  # noqa: E402


if __name__ == "__main__":
    hook_runner.entry_point(
        record_stop_fn=record_stop,
        post_tool_after=periodic_reminder_after_tool,
        extra_actions={
            "post-tool-failure": lambda s, h: record_tool(s, h, failed=True),
        },
        data_dir_env="NEATMEM_CODE_DATA_DIR",
        automatic_flush_reasons={"session-end", "pre-compact"},
    )
