"""Feedback judge — decide whether each injected memory was used (used/unused).

Plan: docs/internal-notes/20261005-citation-feedback-memory-importance-plan.md
§5.3/§5.6. Ported from the validated experiment implementation
(tmp/experiments/20261005-feedback-replay/judge.py, prompt version v6) — the
prompt now lives in the white-box txt (neatmem/prompts/examples/
feedback_judge.txt) and resolves through the standard prompt loader
(MEMORY_FEEDBACK_JUDGE_PROMPT override), per the project loader convention.

Judgment semantics (fixed by the case-gate decisions):
- Judge "was the content used", not "did it cause the answer" — an answer
  consistent with the memory is used even if it might come from parametric
  knowledge (RMM convention).
- no_window is NOT produced here (callers detect the window rules; judge is
  only invoked on a judgeable window).

Rule 7/16 compliance: max_tokens explicitly 32768; finish_reason=length is an
error; parse/validation failures raise JudgeError — never silently unused.
"""

import json
import logging
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional

from openai import OpenAI

from neatmem import config
from neatmem.prompts.loader import load_prompt

logger = logging.getLogger(__name__)

VALID_VERDICTS = ("used", "unused")

# Balanced-brace extraction operates on think-stripped text. The judge prompt
# path is new (not part of the LOCOMO anchor measurement chain), so the shared
# stripper is used here — unlike legacy call sites frozen for anchor parity.
from neatmem.utils.think_tags import strip_think_tags

JUDGE_SYSTEM_PROMPT = (
    "You are evaluating whether injected memories were used by an assistant. "
    "Return JSON only with the format requested."
)

# Judge LLM call constants (rule 16: never below 32768; MiniMax-M3 thinks even
# with thinking disabled).
JUDGE_MAX_TOKENS = 32768
JUDGE_TEMPERATURE = 0.0
RATE_LIMIT_RETRIES = 15


class JudgeError(RuntimeError):
    """judge call/parse failure — rule 7: fail loudly, never silently unused."""


def load_judge_template() -> str:
    """Resolve the judge prompt through the standard loader."""
    return load_prompt(
        "MEMORY_FEEDBACK_JUDGE_PROMPT",
        default_file="feedback_judge.txt",
        required_placeholders=("question", "memories_block", "answer", "memory_ids"),
        supported_placeholders=("question", "memories_block", "answer", "memory_ids"),
    )


def make_judge_client(
    client: Optional[OpenAI] = None,
) -> "tuple[OpenAI, str]":
    """Resolve the judge LLM client + model.

    Dependency injection for tests; production resolves the
    MEMORY_FEEDBACK_JUDGE_* triple with fallback to the main LLM config
    (same convention as QUERY_REWRITE_MODEL_RESOLVED in main.py).
    """
    model = config.MEMORY_FEEDBACK_JUDGE_MODEL or os.environ.get("LLM_MODEL")
    if not model:
        raise JudgeError(
            "judge model unresolved: set MEMORY_FEEDBACK_JUDGE_MODEL or LLM_MODEL"
        )
    if client is not None:
        return client, model
    api_key = config.MEMORY_FEEDBACK_JUDGE_API_KEY or config.LLM_API_KEY
    base_url = config.MEMORY_FEEDBACK_JUDGE_BASE_URL or config.LLM_BASE_URL
    return OpenAI(api_key=api_key, base_url=base_url), model


def _extract_json(text: str) -> str:
    """Think-strip, then balanced-brace scan for the first complete object.

    The degenerate-repetition failure mode (2026-10-05 k200) makes a naive
    first-{-to-last-} slice unparseable across fragments; scanning for the
    first *complete* object recovers "valid JSON + trailing garbage". All-fragment
    outputs are retried/split by the caller.
    """
    text = strip_think_tags(text)
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        escape = False
        for pos in range(start, len(text)):
            ch = text[pos]
            if escape:
                escape = False
                continue
            if ch == "\\" and in_str:
                escape = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : pos + 1]
        start = text.find("{", start + 1)
    return text


def judge_memories(
    client: OpenAI,
    model: str,
    question: str,
    memories: List[Dict[str, Any]],
    answer: str,
    template: Optional[str] = None,
) -> Dict[str, Dict[str, str]]:
    """Judge each memory of one injection as used/unused.

    memories: [{"id": ..., "text": ...}, ...]
    template: optional prompt variant (prompt iteration); defaults to the
        loader-resolved template.
    Returns {"verdicts": {memory_id: verdict}, "evidence": {memory_id: str}}.

    Parse/validation failures retry 3 times (MiniMax unclosed-think truncation
    is transient), then the chunk splits in half; a singleton failure raises
    JudgeError (rule 7: never silently unused).
    """
    if not memories:
        raise JudgeError("judge_memories called with empty memories")
    if not answer.strip():
        raise JudgeError(
            "judge_memories called with empty answer — no complete assistant turn "
            "means no_window, which the caller's window rules must produce instead"
        )
    template = template or load_judge_template()
    last: Optional[JudgeError] = None
    for attempt in range(3):
        try:
            return _judge_once(client, model, question, memories, answer, template)
        except JudgeError as e:
            last = e
            logger.warning(
                "[feedback-judge] attempt %d/3 failed (%d memories): %s",
                attempt + 1, len(memories), e,
            )
    if len(memories) > 1:
        mid = len(memories) // 2
        logger.info("[feedback-judge] splitting chunk of %d into %d+%d",
                    len(memories), mid, len(memories) - mid)
        left = judge_memories(client, model, question, memories[:mid], answer, template)
        right = judge_memories(client, model, question, memories[mid:], answer, template)
        return {
            "verdicts": {**left["verdicts"], **right["verdicts"]},
            "evidence": {**left["evidence"], **right["evidence"]},
        }
    raise last


def _judge_once(
    client: OpenAI,
    model: str,
    question: str,
    memories: List[Dict[str, Any]],
    answer: str,
    template: str,
) -> Dict[str, Dict[str, str]]:
    memory_ids = [str(m["id"]) for m in memories]
    # Real ids are UUIDs — models miscopy long random strings (hallucinated one
    # char on 2026-10-05). Prompt uses short labels M1..Mk, mapped back after.
    labels = [f"M{i + 1}" for i in range(len(memories))]
    label_to_id = dict(zip(labels, memory_ids))
    memories_block = "\n".join(f"- [{label}] {m['text']}" for label, m in zip(labels, memories))

    resp = None
    last_err: Optional[Exception] = None
    for attempt in range(RATE_LIMIT_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": template.format(
                            question=question,
                            memories_block=memories_block,
                            answer=answer,
                            memory_ids=", ".join(labels),
                        ),
                    },
                ],
                response_format={"type": "json_object"},
                temperature=JUDGE_TEMPERATURE,
                max_tokens=JUDGE_MAX_TOKENS,
                # Provider fork aligned with llm_judge.py: MiniMax dual-shape,
                # others get only the chat-template switch.
                extra_body=(
                    {
                        "chat_template_kwargs": {"enable_thinking": False},
                        "thinking": {"type": "adaptive"},
                    }
                    if "minimax" in (model or "").lower()
                    else {"chat_template_kwargs": {"enable_thinking": False}}
                ),
            )
            break
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            if "429" in msg or "rate limit" in msg or "529" in msg or "overloaded" in msg:
                wait = min(2**attempt * 5, 90)
                logger.warning("[feedback-judge] 429/529 retry %d/%d, wait %ds",
                               attempt + 1, RATE_LIMIT_RETRIES, wait)
                time.sleep(wait)
            else:
                raise JudgeError(f"LLM call failed: {e}") from e
    if resp is None:
        raise JudgeError(f"LLM call failed after retries: {last_err}") from last_err

    choice = resp.choices[0]
    if choice.finish_reason == "length":
        raise JudgeError("response truncated at max_tokens (finish_reason=length)")
    content = choice.message.content or ""
    try:
        parsed = json.loads(_extract_json(content))
    except json.JSONDecodeError as e:
        raise JudgeError(f"judge output not parseable as JSON: {content[:500]}") from e

    raw_verdicts = parsed.get("verdicts")
    if not isinstance(raw_verdicts, list):
        raise JudgeError(f"judge output missing 'verdicts' list: {str(parsed)[:500]}")

    verdicts: Dict[str, str] = {}
    evidence: Dict[str, str] = {}
    for item in raw_verdicts:
        if not isinstance(item, dict):
            raise JudgeError(f"malformed verdict item: {str(item)[:200]}")
        label = str(item.get("memory_id", ""))
        verdict = str(item.get("verdict", ""))
        if label not in label_to_id:
            raise JudgeError(f"judge returned unknown memory label '{label}'")
        if verdict not in VALID_VERDICTS:
            raise JudgeError(f"invalid verdict '{verdict}' for memory '{label}'")
        verdicts[label_to_id[label]] = verdict
        evidence[label_to_id[label]] = str(item.get("evidence", ""))

    missing = [mid for mid in memory_ids if mid not in verdicts]
    if missing:
        raise JudgeError(f"judge output missing verdicts for: {missing}")
    return {"verdicts": verdicts, "evidence": evidence}
