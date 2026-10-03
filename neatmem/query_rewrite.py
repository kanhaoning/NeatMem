"""Query rewrite + semantic expansion (MemOS fine-style), flag-gated.

Plan: docs/internal-notes/20260929-query-rewrite-context-injection-plan.md
(§3 数据流 + 2026-10-01/10-02 修订 R1-R6).

Pipeline per search request (when QUERY_REWRITE_ENABLED):
  1. ``build_context``: recent turns (caller's source, e.g. messages.db) ->
     think-stripped, role-tagged, char-capped conversation string.
  2. ``rewrite_query``: one LLM call -> rephrased_query (self-judging empty)
     + expansions (post-filtered, capped).
  3. Caller runs multi-query retrieval over dedup([final_query, *expansions]).

Design rules honored here:
- Self-judging: no heuristics on whether to rewrite; the LLM returns an empty
  rephrased_query for self-contained/unresolvable queries (MemOS fine同款).
- Fail-open (R6-b): single attempt, SDK-level timeout, every failure mode
  degrades to RewriteResult(original, ()) = single-query retrieval, with
  ``fallback_reason`` set for the observation log.
- No retry on the hot path: fallback is the designed degradation; the
  observation period watches the fallback rate.
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from openai import APITimeoutError, APIConnectionError, RateLimitError

from neatmem.prompts.loader import load_prompt
from neatmem.utils.llm_client import complete_chat, extract_response_text
from neatmem.utils.think_tags import strip_think_tags

logger = logging.getLogger(__name__)

# Cap for the assembled conversation string (plan §3.1: 插件补传封顶 2000 字符,
# kept for the DB-first path so the rewrite prompt stays small).
CONTEXT_MAX_CHARS = 2000

_PROMPT_ENV_VAR = "QUERY_REWRITE_PROMPT"
_PROMPT_DEFAULT_FILE = "query_rewrite_en.txt"


@dataclass(frozen=True)
class RewriteResult:
    """Outcome of one rewrite attempt.

    ``fallback_reason`` is None on success; otherwise one of "timeout",
    "rate_limited", "parse", or "error:<ExceptionType>" — the caller logs it
    for the observation period (回退率指标).

    ``retries_used`` counts transient-error retries before the final outcome
    (0 on first-try success/failure). Production keeps retries=0 (R6-b);
    eval harnesses opt in via QUERY_REWRITE_RETRIES.
    """

    final_query: str
    expansions: Tuple[str, ...] = ()
    rephrased: bool = False
    fallback_reason: Optional[str] = None
    latency_ms: float = 0.0
    retries_used: int = 0


def _fallback(query: str, reason: str, latency_ms: float, retries_used: int = 0) -> RewriteResult:
    return RewriteResult(final_query=query, expansions=(), rephrased=False,
                         fallback_reason=reason, latency_ms=latency_ms,
                         retries_used=retries_used)


def _is_transient(exc: Exception) -> bool:
    """Retry-worthy upstream failure: timeout / rate limit / 529 overload /
    connection error. Anything else (parse, auth, deterministically bad
    request) fails immediately — retrying cannot change its outcome."""
    if isinstance(exc, (APITimeoutError, APIConnectionError, RateLimitError)):
        return True
    msg = str(exc).lower()
    return "429" in msg or "rate limit" in msg or "529" in msg or "overloaded" in msg


def _normalize(text: str) -> str:
    # Whitespace-insensitive: CJK rewrites differing only by incidental
    # spaces still count as verbatim repeats / duplicates.
    return re.sub(r"\s+", "", text).casefold()


def _parse_json_loose(text: str) -> Dict[str, Any]:
    """Loose JSON extraction (validated in the 9-29 prototype: 19 cases, 0
    parse failures; mirrors rerank._parse_json)."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    return {}


def build_context(
    turns: Iterable[Dict[str, Any]],
    max_chars: int = CONTEXT_MAX_CHARS,
) -> str:
    """Assemble the conversation string for the rewrite prompt.

    ``turns``: message dicts with ``role``/``content``, oldest first (any
    source — this module stays storage-agnostic). Only user/assistant turns
    are kept; assistant turns are think-stripped (R3: stored messages may
    carry raw ``<think>`` output); turns empty after stripping are dropped.
    Newest turns win when over ``max_chars``; an oversized newest turn is
    tail-truncated. Cleaning happens before capping so the budget is spent
    on real content.
    """
    lines: List[str] = []
    for turn in turns:
        role = str(turn.get("role", ""))
        if role not in ("user", "assistant"):
            continue
        content = str(turn.get("content", "") or "")
        if role == "assistant":
            content = strip_think_tags(content)
        else:
            content = content.strip()
        if content:
            lines.append(f"{role}: {content}")

    kept: List[str] = []
    total = 0
    for line in reversed(lines):
        cost = len(line) + (1 if kept else 0)  # account for the join newline
        if total + cost > max_chars:
            if not kept:
                # Newest turn alone exceeds the budget: keep its tail.
                kept.append(line[-max_chars:])
            break
        kept.append(line)
        total += cost
    kept.reverse()
    return "\n".join(kept)


def rewrite_query(
    query: str,
    context: str,
    *,
    client: Any,
    model: str,
    timeout_s: float,
    max_expansions: int,
    thinking: bool = False,
    retries: int = 0,
    chat_fn: Callable[..., Any] = complete_chat,
) -> RewriteResult:
    """One rewrite attempt: query + context -> rephrased + expansions.

    ``chat_fn`` is injectable for tests; production uses complete_chat
    (thinking shape + think-stripping built in). Any failure fails open to
    the original query with ``fallback_reason`` set.

    ``retries`` (default 0 = production R6-b single attempt): transient
    failures (timeout / 429 / 529 / connection) are retried with a
    5/10/15/20s backoff before falling back. A retried success is identical
    to a first-try success, so enabling it for eval does not distort the
    measured semantics — it only converts upstream bad-weather fallbacks
    into slower-but-valid measurements.
    """
    started = time.perf_counter()
    prompt = load_prompt(
        _PROMPT_ENV_VAR,
        default_file=_PROMPT_DEFAULT_FILE,
        required_placeholders=("conversation", "query"),
        supported_placeholders=("conversation", "query"),
    ).format(conversation=context, query=query)

    attempt = 0
    while True:
        try:
            resp = chat_fn(
                client,
                model,
                [{"role": "user", "content": prompt}],
                thinking,
                provider=None,
                temperature=0.0,
                max_tokens=32768,
                timeout=timeout_s,
            )
            raw = extract_response_text(resp) or ""
            break
        except Exception as exc:  # noqa: BLE001 - fail-open by design (R6-b)
            if isinstance(exc, APITimeoutError):
                reason = "timeout"
            elif isinstance(exc, RateLimitError):
                reason = "rate_limited"
            else:
                reason = f"error:{type(exc).__name__}"
            if attempt < retries and _is_transient(exc):
                attempt += 1
                wait = min(5 * attempt, 20)  # 5,10,15,20,20 (副本退避节奏)
                logger.info(
                    "[query-rewrite] transient %s, retry %d/%d in %ds",
                    reason, attempt, retries, wait,
                )
                time.sleep(wait)
                continue
            return _fallback(query, reason, _elapsed_ms(started), retries_used=attempt)

    parsed = _parse_json_loose(raw)
    rephrased = str(parsed.get("rephrased_query") or "").strip()
    # Verbatim-repeat shortcut (plan §5.1 shortcut ②): a rewrite that merely
    # restates the query is treated as self-judged empty.
    if rephrased and _normalize(rephrased) == _normalize(query):
        rephrased = ""

    if not rephrased and not parsed.get("expansions"):
        # Nothing usable came back (parse failure or empty JSON) — treat as
        # fallback rather than a genuine "self-contained" judgment.
        return _fallback(query, "parse", _elapsed_ms(started))

    final_query = rephrased or query
    expansions = _clean_expansions(
        parsed.get("expansions"), exclude={_normalize(final_query), _normalize(query)},
        max_expansions=max_expansions,
    )
    return RewriteResult(
        final_query=final_query,
        expansions=tuple(expansions),
        rephrased=bool(rephrased),
        fallback_reason=None,
        latency_ms=_elapsed_ms(started),
        retries_used=attempt,
    )


def _clean_expansions(
    raw: Any,
    *,
    exclude: set,
    max_expansions: int,
) -> List[str]:
    """Strip, dedup (against final/original query and each other, normalized),
    cap at ``max_expansions``. 0 is a valid outcome (single-query retrieval);
    no hard minimum — padding would inject noise."""
    if not isinstance(raw, list) or max_expansions <= 0:
        return []
    out: List[str] = []
    seen = set(exclude)
    for item in raw:
        text = str(item or "").strip()
        key = _normalize(text)
        if not text or key in seen:
            continue
        seen.add(key)
        out.append(text)
        if len(out) >= max_expansions:
            break
    return out


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000
