"""Shared test setup: make core/ importable."""

from __future__ import annotations

import sys
from pathlib import Path

HOST_ROOT = Path(__file__).resolve().parents[1]
_core = HOST_ROOT / "core"
sys.path.insert(0, str(_core))

import hook_runner  # noqa: E402,F401
