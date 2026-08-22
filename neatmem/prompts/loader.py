"""Prompt override loader (id or file path, from_pretrained-style).

Every core prompt can be overridden via a ``*_PROMPT`` environment variable
(or the matching ``neatmem serve --*-prompt`` CLI flag). The value is either:

- a built-in variant id (see _BUILTIN_VARIANTS, resolved to the packaged
  example file), or
- a path to the user's own prompt file.

When the env var is not set, the default prompt is used: either the
``default`` string passed by the caller, or the packaged txt file named by
``default_file`` (prompts migrated to white-box txt files; the txt is the
single source of truth, no in-code copy). A missing default_file is a
packaging error -> SystemExit at load time.

Prompts are resolved once per process (then cached); placeholder checks run
on every call because the same env var can serve multiple code paths with
different placeholder contracts (e.g. EDIT_PROMPT: the pointwise path also
supplies {relation}). Restart the server after editing a prompt file.
"""

import os
import string
from importlib.resources import files as resource_files
from pathlib import Path
from typing import Dict, Optional, Tuple

# Built-in variants selectable by id: env var -> {id: example filename}.
# The packaged example files are the single source of truth for non-default
# variants (dedup_en = the v7_en prompt validated on 2026-07-24).
_BUILTIN_VARIANTS: Dict[str, Dict[str, str]] = {
    "DEDUP_PROMPT": {
        "zh": "dedup_zh.example.txt",
        "en": "dedup_en.example.txt",
    },
}

# Cache resolved prompt text per env var; each prompt is read at most once
# per process. Placeholder checks are NOT cached (they depend on the call
# path, not just the text).
_cache: Dict[str, str] = {}


def _template_fields(text: str) -> set:
    """Top-level field names in a format template (``{{`` escapes excluded)."""
    return {
        f.split(".")[0].split("[")[0].split(":")[0].split("!")[0]
        for _, f, _, _ in string.Formatter().parse(text)
        if f
    }


def _check_placeholders(
    env_var: str,
    source: str,
    text: str,
    required_placeholders: Tuple[str, ...],
    supported_placeholders: Optional[Tuple[str, ...]],
) -> None:
    missing = [ph for ph in required_placeholders if "{" + ph + "}" not in text]
    if missing:
        need = ", ".join("{" + ph + "}" for ph in required_placeholders)
        miss = ", ".join("{" + m + "}" for m in missing)
        raise SystemExit(
            f"ERROR: {env_var} prompt {source} is missing required placeholder(s): {miss}\n"
            f"  This prompt requires: {need}\n"
            f"  Compare with the matching template in neatmem/prompts/examples/."
        )

    if supported_placeholders is None:
        return
    try:
        fields = _template_fields(text)
    except ValueError as e:
        raise SystemExit(
            f"ERROR: {env_var} prompt {source} is not a valid format template: {e}\n"
            f"  Literal braces must be escaped as {{{{ and }}}}."
        )
    extra = sorted(fields - set(supported_placeholders))
    if extra:
        got = ", ".join("{" + f + "}" for f in extra)
        ok = ", ".join("{" + ph + "}" for ph in supported_placeholders) or "(none)"
        raise SystemExit(
            f"ERROR: {env_var} prompt {source} uses placeholder(s) this code path does not supply: {got}\n"
            f"  Supplied here: {ok}\n"
            f"  Note: {{relation}} is only available on the pointwise dedup path."
        )


def load_prompt(
    env_var: str,
    default: str = "",
    required_placeholders: Tuple[str, ...] = (),
    default_file: Optional[str] = None,
    supported_placeholders: Optional[Tuple[str, ...]] = None,
) -> str:
    """Return the prompt template selected by ``env_var``, or the default.

    Resolution order for the env var value:
    1. known built-in id -> packaged example file
    2. existing file path -> that file
    3. otherwise -> SystemExit listing valid ids and path requirements

    Without the env var: ``default_file`` (packaged txt under
    neatmem/prompts/examples/) wins; otherwise the ``default`` string.

    A missing required ``{placeholder}``, or any ``{placeholder}`` outside
    ``supported_placeholders`` (when given), -> SystemExit.
    """
    if env_var in _cache:
        text = _cache[env_var]
        _check_placeholders(
            env_var, "(cached)", text, required_placeholders, supported_placeholders
        )
        return text

    value = os.environ.get(env_var)
    if value:
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
        source = repr(value)
    elif default_file is not None:
        try:
            text = (resource_files("neatmem.prompts") / "examples" / default_file).read_text(
                encoding="utf-8"
            )
        except FileNotFoundError:
            raise SystemExit(
                f"ERROR: packaged default prompt file {default_file!r} is missing from "
                f"neatmem/prompts/examples/ — the installation is broken (txt files are "
                f"the single source of truth for this prompt). Reinstall neatmem."
            )
        source = f"(packaged {default_file})"
    else:
        text = default
        source = "(built-in default)"

    _check_placeholders(env_var, source, text, required_placeholders, supported_placeholders)
    _cache[env_var] = text
    return text


def absolutize_prompt_value(env_var: str, value: str) -> str:
    """Anchor a user-supplied prompt file path at the invocation cwd.

    Built-in variant ids are returned unchanged. Prompt env vars are consumed
    by child processes whose cwd may differ from the cwd where the flag was
    parsed (e.g. `neatmem evaluate` spawns serve/ingest under the output
    dir), so relative paths must be absolutized before entering the env.
    """
    if value in _BUILTIN_VARIANTS.get(env_var, {}):
        return value
    p = Path(value)
    if not p.is_absolute():
        p = Path.cwd() / p
    return str(p)


def clear_prompt_cache() -> None:
    """Drop all cached prompts (mainly for tests)."""
    _cache.clear()
