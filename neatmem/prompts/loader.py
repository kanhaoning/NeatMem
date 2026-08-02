"""Prompt override loader (id or file path, from_pretrained-style).

Every core prompt can be overridden via a ``*_PROMPT`` environment variable
(or the matching ``neatmem serve --*-prompt`` CLI flag). The value is either:

- a built-in variant id (see _BUILTIN_VARIANTS, resolved to the packaged
  example file), or
- a path to the user's own prompt file.

When the env var is not set, the built-in default prompt is returned
unchanged. Prompts are loaded once per process (first use, then cached) --
restart the server after editing a prompt file.
"""

import os
from importlib.resources import files as resource_files
from pathlib import Path
from typing import Dict, Tuple

# Built-in variants selectable by id: env var -> {id: example filename}.
# The packaged example files are the single source of truth for non-default
# variants (dedup_en = the v7_en prompt validated on 2026-07-24).
_BUILTIN_VARIANTS: Dict[str, Dict[str, str]] = {
    "DEDUP_PROMPT": {
        "zh": "dedup_zh.example.txt",
        "en": "dedup_en.example.txt",
    },
}

# Cache loaded prompts per env var; each prompt is read at most once per process.
_cache: Dict[str, str] = {}


def load_prompt(env_var: str, default: str, required_placeholders: Tuple[str, ...] = ()) -> str:
    """Return the prompt template selected by ``env_var``, or ``default``.

    Resolution order for the env var value:
    1. known built-in id -> packaged example file
    2. existing file path -> that file
    3. otherwise -> SystemExit listing valid ids and path requirements

    A missing required ``{placeholder}`` in the resolved text -> SystemExit.
    """
    if env_var in _cache:
        return _cache[env_var]

    value = os.environ.get(env_var)
    if not value:
        _cache[env_var] = default
        return default

    builtin = _BUILTIN_VARIANTS.get(env_var, {})
    if value in builtin:
        text = (resource_files("neatmem.prompts") / "examples" / builtin[value]).read_text(
            encoding="utf-8"
        )
    else:
        p = Path(value)
        if not p.is_file():
            ids = ", ".join(sorted(builtin)) or "(none)"
            raise SystemExit(
                f"ERROR: {env_var}={value!r} is neither a known built-in id nor an existing file.\n"
                f"  Built-in ids: {ids}\n"
                f"  Or pass a path to your own prompt file (absolute path recommended)."
            )
        text = p.read_text(encoding="utf-8")

    missing = [ph for ph in required_placeholders if "{" + ph + "}" not in text]
    if missing:
        need = ", ".join("{" + ph + "}" for ph in required_placeholders)
        miss = ", ".join("{" + m + "}" for m in missing)
        raise SystemExit(
            f"ERROR: {env_var} prompt {value!r} is missing required placeholder(s): {miss}\n"
            f"  This prompt requires: {need}\n"
            f"  Compare with the matching template in neatmem/prompts/examples/."
        )

    _cache[env_var] = text
    return text


def clear_prompt_cache() -> None:
    """Drop all cached prompts (mainly for tests)."""
    _cache.clear()
