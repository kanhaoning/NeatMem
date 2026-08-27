"""Prompt override loader (file path only).

Every core prompt can be overridden via a ``*_PROMPT`` environment variable
(or the matching ``neatmem serve --*-prompt`` CLI flag). The value must be a
path to the user's own prompt file (absolute recommended); anything that is
not an existing file is a hard error (SystemExit).

When the env var is not set, the default prompt is used: either the
``default`` string passed by the caller, or the packaged txt file named by
``default_file`` (prompts migrated to white-box txt files; the txt is the
single source of truth, no in-code copy). A missing default_file is a
packaging error -> SystemExit at load time. For the dedup prompt the
default_file is auto-paired from (DEDUP_DETECTOR, DEDUP_RESOLVER) via
``dedup_prompt_default_file``.

Every resolution is logged once per process with source and sha256 prefix
(observability requirement: the resolved prompt file must be visible in the
startup log). Prompts are resolved once per process (then cached);
placeholder checks run on every call because the same env var can serve
multiple code paths with different placeholder contracts (e.g. EDIT_PROMPT:
the pointwise path also supplies {relation}). Restart the server after
editing a prompt file.
"""

import hashlib
import logging
import os
import string
from importlib.resources import files as resource_files
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Dedup prompt auto-pairing: (detector, resolver) -> packaged default file.
# The detector owns the judgment protocol; the resolver only picks between
# the lenient (skip) and strict (update-capable) listwise variants.
#   listwise + skip                 -> v7en (validated best for skip)
#   listwise + replace/rewrite/edit -> strict (validated best under update)
#   listwise_multitarget (any)      -> per-candidate judgments protocol
#   pointwise (any)                 -> MemOS 3-class pair detector
# A miss means a new detector/resolver was added without registering its
# prompt -> fail fast, never silently fall back to a default.
_DEDUP_PROMPT_TABLE: Dict[Tuple[str, Optional[str]], str] = {
    ("listwise", "skip"): "dedup_listwise_en.txt",
    ("listwise", "replace"): "dedup_listwise_strict_en.txt",
    ("listwise", "rewrite"): "dedup_listwise_strict_en.txt",
    ("listwise", "edit"): "dedup_listwise_strict_en.txt",
    ("listwise_multitarget", None): "dedup_listwise_multitarget_en.txt",
    ("pointwise", None): "dedup_pointwise_en.txt",
}


def dedup_prompt_default_file(detector: str, resolver: str) -> str:
    """Packaged dedup prompt file for a (detector, resolver) combo.

    resolver=None rows match any resolver of that detector. A table miss is
    a hard error: silently picking a default would pair a judgment protocol
    with a prompt that does not describe it.
    """
    fname = _DEDUP_PROMPT_TABLE.get((detector, resolver))
    if fname is None:
        fname = _DEDUP_PROMPT_TABLE.get((detector, None))
    if fname is None:
        raise SystemExit(
            f"ERROR: no packaged dedup prompt registered for "
            f"DEDUP_DETECTOR={detector!r}, DEDUP_RESOLVER={resolver!r}.\n"
            f"  Register one in _DEDUP_PROMPT_TABLE (neatmem/prompts/loader.py), "
            f"or set DEDUP_PROMPT to a prompt file path."
        )
    return fname

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

    The env var value must be a path to an existing prompt file; anything
    else -> SystemExit.

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
        p = Path(value)
        if not p.is_file():
            raise SystemExit(
                f"ERROR: {env_var}={value!r} is not an existing file.\n"
                f"  Pass a path to your own prompt file (absolute path recommended)."
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

    logger.info(
        "prompt %s resolved: %s sha256=%s",
        env_var, source, hashlib.sha256(text.encode("utf-8")).hexdigest()[:12],
    )
    _check_placeholders(env_var, source, text, required_placeholders, supported_placeholders)
    _cache[env_var] = text
    return text


def absolutize_prompt_value(env_var: str, value: str) -> str:
    """Anchor a user-supplied prompt file path at the invocation cwd.

    Prompt env vars are consumed by child processes whose cwd may differ
    from the cwd where the flag was parsed (e.g. `neatmem evaluate` spawns
    serve/ingest under the output dir), so relative paths must be
    absolutized before entering the env.
    """
    p = Path(value)
    if not p.is_absolute():
        p = Path.cwd() / p
    return str(p)


def clear_prompt_cache() -> None:
    """Drop all cached prompts (mainly for tests)."""
    _cache.clear()
