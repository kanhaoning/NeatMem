"""Search-result rerank module: two engines behind one dispatcher.

RERANK_MODE selects the engine:
- llm:           LLM rerank (default). LLM_RERANK_MODE=listwise|pointwise;
                 model/key/base_url come from the main LLM_* config.
- cross_encoder: dedicated scoring model via CROSS_ENCODER_* params
                 (multi-provider presets + local sentence-transformers).
- off:           pure vector retrieval, no rerank.

Pipeline-level params (candidate count / text truncation) live inside each
engine group because their optimal defaults diverge (listwise is bounded by
lost-in-middle; cross-encoder scores each doc independently).
RERANK_MAX_CONCURRENT (in main.py) stays shared: it guards the server, and
the two engines never run concurrently.
"""

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import httpx

from neatmem.prompts.loader import load_prompt
from neatmem.utils.llm_client import extract_response_text, complete_chat

logger = logging.getLogger(__name__)

# Explicit provider for thinking-shape/constraint selection; None = legacy
# model-name matching (import-free env read keeps rerank importable standalone).
LLM_PROVIDER = os.environ.get("LLM_PROVIDER")

# ---------------------------------------------------------------------------
# Engine selection
# ---------------------------------------------------------------------------
RERANK_MODE = os.environ.get("RERANK_MODE", "llm")
if RERANK_MODE not in ("llm", "cross_encoder", "off"):
    raise ValueError(
        f"Invalid RERANK_MODE={RERANK_MODE!r}, expected llm|cross_encoder|off"
    )

# ---------------------------------------------------------------------------
# LLM rerank params
# ---------------------------------------------------------------------------
LLM_RERANK_MODE = os.environ.get("LLM_RERANK_MODE", "listwise")
if LLM_RERANK_MODE not in ("listwise", "pointwise"):
    raise ValueError(
        f"Invalid LLM_RERANK_MODE={LLM_RERANK_MODE!r}, expected listwise|pointwise"
    )

# Head size: only the top-N candidates (by vector score) are rescored, the
# rest are appended unchanged (head/tail design). Default 20 is the validated
# listwise value; pointwise shares it as a conservative default until sweep
# data justifies a per-mode override.
LLM_RERANK_CANDS = int(os.environ.get("LLM_RERANK_CANDS", "20"))

# Candidate text truncation length before sending to the LLM; 0 = no truncation.
LLM_RERANK_CAND_TEXT_LEN = int(os.environ.get("LLM_RERANK_CAND_TEXT_LEN", "120"))

# ---------------------------------------------------------------------------
# Cross-encoder params
# ---------------------------------------------------------------------------
# Only the siliconflow preset is verified against the live API. Other presets
# follow the same /rerank contract ({model,query,documents} ->
# {results:[{index,relevance_score}]}) but are untested; cohere/dashscope may
# need adapter tweaks. "local" loads the model in-process via
# sentence-transformers (pip install neatmem[local-reranker]).
CROSS_ENCODER_PROVIDER_PRESETS = {
    "siliconflow": {
        "base_url": "https://api.siliconflow.cn/v1",
        "model": "Qwen/Qwen3-Reranker-8B",
        "key_env": "SILICONFLOW_API_KEY",
    },
    "jina": {  # unverified
        "base_url": "https://api.jina.ai/v1",
        "model": "jina-reranker-v3",
        "key_env": "JINA_API_KEY",
    },
    "cohere": {  # unverified
        "base_url": "https://api.cohere.com/v2",
        "model": "rerank-v3.5",
        "key_env": "COHERE_API_KEY",
    },
    "dashscope": {  # unverified
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "gte-rerank-v2",
        "key_env": "DASHSCOPE_API_KEY",
    },
    "xinference": {  # unverified
        "base_url": "http://localhost:9997/v1",
        "model": "bge-reranker-v2-m3",
        "key_env": None,
    },
    "local": {
        "base_url": None,
        "model": "BAAI/bge-reranker-v2-m3",
        "key_env": None,
    },
}

CROSS_ENCODER_PROVIDER = os.environ.get("CROSS_ENCODER_PROVIDER", "siliconflow")
if CROSS_ENCODER_PROVIDER not in CROSS_ENCODER_PROVIDER_PRESETS:
    raise ValueError(
        f"Invalid CROSS_ENCODER_PROVIDER={CROSS_ENCODER_PROVIDER!r}, expected one of: "
        f"{', '.join(sorted(CROSS_ENCODER_PROVIDER_PRESETS))}"
    )
_CE_PRESET = CROSS_ENCODER_PROVIDER_PRESETS[CROSS_ENCODER_PROVIDER]

CROSS_ENCODER_MODEL = os.environ.get("CROSS_ENCODER_MODEL") or _CE_PRESET["model"]
CROSS_ENCODER_BASE_URL = os.environ.get("CROSS_ENCODER_BASE_URL") or _CE_PRESET["base_url"]
CROSS_ENCODER_API_KEY = os.environ.get("CROSS_ENCODER_API_KEY") or (
    os.environ.get(_CE_PRESET["key_env"], "") if _CE_PRESET["key_env"] else ""
)

# pointwise = per-doc relevance score (the /rerank contract). listwise is a
# reserved value: no target provider yet, so selecting it fails fast.
CROSS_ENCODER_MODE = os.environ.get("CROSS_ENCODER_MODE", "pointwise")
if CROSS_ENCODER_MODE not in ("pointwise", "listwise"):
    raise ValueError(
        f"Invalid CROSS_ENCODER_MODE={CROSS_ENCODER_MODE!r}, expected pointwise|listwise"
    )
if RERANK_MODE == "cross_encoder" and CROSS_ENCODER_MODE == "listwise":
    raise ValueError(
        "CROSS_ENCODER_MODE=listwise is reserved but not implemented: no "
        "cross-encoder provider with a listwise API is supported yet."
    )

# Head size for cross-encoder rerank. Default 100: per-doc independent scoring
# has no lost-in-middle degradation, so a wider head pays off.
CROSS_ENCODER_CANDS = int(os.environ.get("CROSS_ENCODER_CANDS", "100"))

# Candidate text truncation before scoring; 0 = no truncation (model max_seq_len
# is the backstop). No validated optimum on the cross-encoder side, so the
# default stays untruncated.
CROSS_ENCODER_CAND_TEXT_LEN = int(os.environ.get("CROSS_ENCODER_CAND_TEXT_LEN", "0"))

# Relative threshold filter: drop docs with score < ratio * top_score.
# 0 = sort + truncate only (mem0 style). Filter mode hurt LOCOMO scores in
# our benchmarks, so it is kept only as a controllable option.
CROSS_ENCODER_REL_THRESHOLD = float(os.environ.get("CROSS_ENCODER_REL_THRESHOLD", "0"))

# HTTP timeout seconds (API providers only). 60s default so large batches
# don't crash on a tight timeout.
CROSS_ENCODER_TIMEOUT = int(os.environ.get("CROSS_ENCODER_TIMEOUT", "60"))

# local provider only.
CROSS_ENCODER_DEVICE = os.environ.get("CROSS_ENCODER_DEVICE", "auto")
CROSS_ENCODER_BATCH_SIZE = int(os.environ.get("CROSS_ENCODER_BATCH_SIZE", "32"))


@dataclass
class LLMRerankResult:
    kept: List[Dict[str, Any]]           # memories kept (sorted)
    dropped: List[Dict[str, Any]]        # memories dropped
    raw_response: Optional[str] = None   # raw LLM output, for explainability


def validate_rerank_prompt_at_boot() -> None:
    """Fail fast on an invalid LLM_RERANK_PROMPT (also warms the loader cache).

    Placeholder requirements differ per LLM_RERANK_MODE, so validation must
    follow the active mode.
    """
    if LLM_RERANK_MODE == "pointwise":
        load_prompt("LLM_RERANK_PROMPT", _POINTWISE_PROMPT, ("query", "memory"))
    else:
        load_prompt("LLM_RERANK_PROMPT", _LISTWISE_PROMPT, ("query", "candidates_text"))


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def llm_rerank(openai_client, llm_model: str, query: str,
               documents: List[Dict[str, Any]], top_k: int = 5) -> LLMRerankResult:
    """Unified entry point; dispatches on RERANK_MODE.

    The returned kept = rescored head + tail appended in original order.
    The caller must truncate to [:top_k].
    """
    if RERANK_MODE == "off":
        return LLMRerankResult(kept=documents[:top_k], dropped=documents[top_k:])
    elif RERANK_MODE == "llm":
        if LLM_RERANK_MODE == "pointwise":
            kept, dropped = _llm_rerank_pointwise(openai_client, llm_model, query, documents, top_k)
        else:
            kept, dropped = _llm_rerank_listwise(openai_client, llm_model, query, documents, top_k)
        return LLMRerankResult(kept=kept, dropped=dropped)
    elif RERANK_MODE == "cross_encoder":
        # openai_client / llm_model are unused; cross-encoder has its own
        # provider config (CROSS_ENCODER_*).
        return _cross_encoder_rerank(query, documents, top_k)
    else:
        # Unreachable: RERANK_MODE is validated at import time.
        raise ValueError(f"Invalid RERANK_MODE={RERANK_MODE!r}")


def _head_tail_split(documents: List[Dict[str, Any]], cands: int
                     ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Sort by vector score desc and split into head (rescored) / tail (appended)."""
    sorted_docs = sorted(documents, key=lambda x: x.get("score", 0), reverse=True)
    return sorted_docs[:cands], sorted_docs[cands:]


def _truncate(text: str, limit: int) -> str:
    if limit > 0:
        text = text[:limit]
    return text.replace("\n", " ")


# ---------------------------------------------------------------------------
# Listwise rerank
# ---------------------------------------------------------------------------

_LISTWISE_PROMPT = """Judge whether the following candidate memories are relevant to the user's query, and return the indices of all relevant memories (sorted by relevance, most relevant first).

Query: "{query}"

Candidate memory list:
{candidates_text}

Task:
1. Evaluate each memory's relevance to the query. Relevance is not limited to literal matches; indirect inference from key facts counts. Pay special attention to specific entities the query asks about (book titles, activities, dates, locations, numbers, specific objects). Even if the overall topic does not fully match, containing these key entities is sufficient.
2. **Only return memories that are truly relevant.** Irrelevant memories must not appear in the list.
3. Sort relevant memories by relevance, descending. Earlier in the returned list = more relevant.

Output format (strict JSON):
{{"relevant": [index1, index2, ...]}}

Output JSON only, no other content."""


def _get_prompt(query: str, candidates_text: str) -> str:
    """Fill the rerank prompt."""
    # LLM_RERANK_PROMPT overrides the built-in template (loaded and cached on first call).
    template = load_prompt("LLM_RERANK_PROMPT", _LISTWISE_PROMPT, ("query", "candidates_text"))
    return template.format(query=query, candidates_text=candidates_text)


def _build_candidates_text(documents: List[Dict[str, Any]]) -> str:
    lines = []
    for i, doc in enumerate(documents, 1):
        text = _truncate(doc.get("memory", ""), LLM_RERANK_CAND_TEXT_LEN)
        lines.append(f"[{i}] {text}")
    return "\n".join(lines)


def _parse_json(text: str) -> Dict[str, Any]:
    """Extract JSON from LLM output."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1:
        try:
            return json.loads(text[start:end+1])
        except json.JSONDecodeError:
            pass
    return {}


def _llm_rerank_listwise(openai_client, llm_model: str, query: str,
                          documents: List[Dict[str, Any]], top_k: int = 5) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """LLM listwise rerank: the LLM picks all relevant memories and sorts
    them; top_k is not enforced here.

    Args:
        openai_client: OpenAI client instance
        llm_model: LLM model id
        query: search query
        documents: vector-search candidates, each with at least "memory" and "score"
        top_k: count expected by the answer stage (used for fallback key matching)

    Returns:
        (kept, dropped)
        - kept = relevant memories picked by the LLM from the head + tail appended
        - dropped = head memories not picked by the LLM
        The caller must truncate kept to [:top_k].
    """
    if not documents:
        return documents, []

    # head/tail design (after signetai): head is rescored by the LLM, the tail
    # is appended unchanged so no data is lost.
    head, tail = _head_tail_split(documents, LLM_RERANK_CANDS)

    candidates_text = _build_candidates_text(head)
    prompt = _get_prompt(query, candidates_text)

    try:
        resp = complete_chat(
            openai_client,
            llm_model,
            [{"role": "user", "content": prompt}],
            enable=False,
            provider=LLM_PROVIDER,
            temperature=0.0,
            max_tokens=2000,
        )
        response = extract_response_text(resp) or ""
    except Exception as e:
        print(f"  [WARN] Listwise rerank failed: {e}, fallback to score-based")
        return head + tail, []

    parsed = _parse_json(response)

    # Prefer the "relevant" array
    relevant_indices = None
    for key in ["relevant", "selected", "related", "indices"]:
        if key in parsed and isinstance(parsed[key], list):
            relevant_indices = parsed[key]
            break

    # Fallback 1: try top_k-style key names
    if not relevant_indices or not isinstance(relevant_indices, list):
        for key in [f"top{top_k}", "top5", "top10", "top_k", "top"]:
            if key in parsed and isinstance(parsed[key], list):
                relevant_indices = parsed[key]
                break

    if relevant_indices is None or not isinstance(relevant_indices, list):
        print(f"  [WARN] Listwise parse failed, response: {response[:200]}")
        return head + tail, []

    # Convert to 0-based indices and map to documents
    selected = []
    for idx in relevant_indices:
        try:
            i = int(idx) - 1  # numbering is 1-based
            if 0 <= i < len(head):
                selected.append(head[i])
        except (ValueError, TypeError):
            continue

    # Deduplicate
    seen = set()
    kept_head = []
    for d in selected:
        mem = d.get("memory", "")
        if mem not in seen:
            seen.add(mem)
            kept_head.append(d)

    # If the LLM returned nothing, emergency fallback: all head + tail as-is
    if not kept_head:
        return head + tail, []

    # No cap*2. Rescored head + tail appended; the caller truncates to [:top_k].
    kept_mems = set(d.get("memory", "") for d in kept_head)
    dropped_head = [d for d in head if d.get("memory", "") not in kept_mems]

    # The tail must go into kept (not dropped), otherwise callers that only
    # read kept would lose it.
    kept = kept_head + tail
    dropped = dropped_head

    return kept, dropped


# ---------------------------------------------------------------------------
# Pointwise LLM rerank
# ---------------------------------------------------------------------------

_POINTWISE_PROMPT = """Rate how relevant the following memory is for answering the user's query, on a scale of 0 to 10 (0 = completely irrelevant, 10 = essential). Relevance is not limited to literal matches; indirect inference from key facts counts.

Query: "{query}"

Memory: "{memory}"

Output format (strict JSON):
{{"score": <integer 0-10>}}

Output JSON only, no other content."""


def _llm_rerank_pointwise(openai_client, llm_model: str, query: str,
                           documents: List[Dict[str, Any]], top_k: int = 5) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """LLM pointwise rerank: score each head doc independently, sort by score.

    No filtering (mem0 style): kept = scored head sorted desc + tail, dropped
    is empty. A doc whose score fails to parse sinks to the bottom with a
    warning rather than being silently dropped.
    """
    if not documents:
        return documents, []

    head, tail = _head_tail_split(documents, LLM_RERANK_CANDS)
    template = load_prompt("LLM_RERANK_PROMPT", _POINTWISE_PROMPT, ("query", "memory"))

    scored = []
    for doc in head:
        text = _truncate(doc.get("memory", ""), LLM_RERANK_CAND_TEXT_LEN)
        prompt = template.format(query=query, memory=text)
        score: Optional[float] = None
        response = ""
        try:
            resp = complete_chat(
                openai_client,
                llm_model,
                [{"role": "user", "content": prompt}],
                enable=False,
                provider=LLM_PROVIDER,
                temperature=0.0,
                max_tokens=2000,
            )
            response = extract_response_text(resp) or ""
            parsed = _parse_json(response)
            raw = parsed.get("score")
            if raw is not None:
                score = float(raw)
        except Exception as e:
            logger.warning("[pointwise] scoring call failed: %s", e)
        if score is None:
            # Unscored docs sink to the bottom of kept (no silent drop).
            logger.warning("[pointwise] unparseable score, doc sinks to bottom: %s",
                           response[:120])
        doc_copy = dict(doc)
        doc_copy["rerank_score"] = score if score is not None else float("-inf")
        scored.append(doc_copy)

    scored.sort(key=lambda x: x["rerank_score"], reverse=True)
    return scored + tail, []


# ---------------------------------------------------------------------------
# Cross-encoder rerank
# ---------------------------------------------------------------------------

# Error code returned by SiliconFlow when quota is exhausted. The quota window
# is multi-hour and rolling, so retrying is wasted; abort immediately.
_SILICONFLOW_QUOTA_EXHAUSTED_CODE = 2056


def _call_rerank_api(query: str, doc_texts: List[str]) -> List[float]:
    """Call a /rerank HTTP API. Returns scores aligned with doc_texts order.

    Contract (shared by siliconflow/jina/cohere/dashscope/xinference presets):
    request {model, query, documents} -> response {results:[{index,
    relevance_score}]} sorted by score desc.

    Raises:
        ValueError: API key not set (providers that require one), or empty doc_texts.
        RuntimeError: HTTP failure after 3 retries (no silent fallback per
            CLAUDE.md rule 7 - silent fallback would propagate error state
            downstream and waste hours of run time); 2056 quota exhaustion
            (siliconflow) raised immediately without retry.

    Retry policy:
        - 429 / 5xx: retry with exponential backoff 1s/2s/4s (3 attempts total)
        - 2056 (quota exhausted, siliconflow): raise immediately, no retry
        - Other 4xx: raise immediately, no retry
    """
    if not doc_texts:
        raise ValueError("doc_texts must be non-empty")
    if _CE_PRESET["key_env"] and not CROSS_ENCODER_API_KEY:
        raise ValueError(
            f"CROSS_ENCODER_API_KEY not set (fallback env {_CE_PRESET['key_env']} also unset); "
            f"required for cross_encoder provider {CROSS_ENCODER_PROVIDER!r}"
        )

    url = f"{CROSS_ENCODER_BASE_URL}/rerank"
    headers = {"Content-Type": "application/json"}
    if CROSS_ENCODER_API_KEY:
        headers["Authorization"] = f"Bearer {CROSS_ENCODER_API_KEY}"
    payload = {
        "model": CROSS_ENCODER_MODEL,
        "query": query,
        "documents": doc_texts,
    }

    last_err: Optional[Exception] = None
    for attempt in range(3):
        try:
            resp = httpx.post(
                url,
                headers=headers,
                json=payload,
                timeout=CROSS_ENCODER_TIMEOUT,
            )
            if resp.status_code == 429 or 500 <= resp.status_code < 600:
                last_err = RuntimeError(
                    f"rerank API HTTP {resp.status_code}: {resp.text[:200]}"
                )
                wait = 2 ** attempt  # 1s, 2s, 4s
                logger.warning(
                    "[cross_encoder] HTTP %s, retry %d/3 after %ds",
                    resp.status_code, attempt + 1, wait,
                )
                time.sleep(wait)
                continue

            try:
                data = resp.json()
            except ValueError as je:
                last_err = RuntimeError(
                    f"rerank API JSON decode failed: {je}; body={resp.text[:200]}"
                )
                # JSON decode failure is non-retryable (server returned garbage)
                break

            # SiliconFlow error envelope: {"code": <int>, "message": "...", "data": null}
            if (CROSS_ENCODER_PROVIDER == "siliconflow"
                    and isinstance(data, dict) and "code" in data and data.get("code") != 0):
                err_code = data.get("code")
                if err_code == _SILICONFLOW_QUOTA_EXHAUSTED_CODE:
                    raise RuntimeError(
                        f"SiliconFlow quota exhausted (code 2056): {data.get('message', '')}. "
                        "Quota window is multi-hour and rolling; aborting without retry."
                    )
                last_err = RuntimeError(
                    f"SiliconFlow API error code={err_code}: {data.get('message', '')}"
                )
                # Non-retryable business error (4xx-like).
                break

            resp.raise_for_status()

            results = data.get("results") or []
            # CRITICAL: the API returns results sorted by relevance_score desc.
            # We must realign scores back to input doc_texts order using `index`.
            if len(results) != len(doc_texts):
                raise RuntimeError(
                    f"rerank API silent drop: got {len(results)} results for "
                    f"{len(doc_texts)} docs (model={CROSS_ENCODER_MODEL})"
                )

            # Build aligned score list: scores[i] = score for doc_texts[i].
            scores = [0.0] * len(doc_texts)
            for r in results:
                idx = r.get("index")
                score = r.get("relevance_score", 0.0)
                if idx is None:
                    raise RuntimeError(
                        f"rerank API result missing index: {r}"
                    )
                scores[idx] = float(score)
            return scores

        except httpx.TimeoutException as te:
            last_err = RuntimeError(f"rerank API timeout: {te}")
            wait = 2 ** attempt
            logger.warning(
                "[cross_encoder] timeout, retry %d/3 after %ds", attempt + 1, wait,
            )
            time.sleep(wait)
            continue

    # Exhausted 3 retries.
    raise RuntimeError(
        f"rerank API failed after 3 retries: {last_err}"
    )


_LOCAL_MODEL = None


def _call_local_rerank(query: str, doc_texts: List[str]) -> List[float]:
    """Score docs in-process with a sentence-transformers CrossEncoder."""
    global _LOCAL_MODEL
    if not doc_texts:
        raise ValueError("doc_texts must be non-empty")
    if _LOCAL_MODEL is None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "CROSS_ENCODER_PROVIDER=local requires sentence-transformers: "
                "pip install neatmem[local-reranker]"
            )
        device = None if CROSS_ENCODER_DEVICE == "auto" else CROSS_ENCODER_DEVICE
        _LOCAL_MODEL = CrossEncoder(CROSS_ENCODER_MODEL, device=device)
    pairs = [(query, t) for t in doc_texts]
    scores = _LOCAL_MODEL.predict(pairs, batch_size=CROSS_ENCODER_BATCH_SIZE)
    return [float(s) for s in scores]


def _cross_encoder_rerank(query: str, documents: List[Dict[str, Any]],
                         top_k: int) -> LLMRerankResult:
    """Cross-encoder rerank: score head docs independently, sort by score.

    Two modes controlled by CROSS_ENCODER_REL_THRESHOLD:
    - threshold > 0: drop docs whose score < ratio * top_score (filter style,
        mimics listwise filter behavior; hurt LOCOMO scores in our benchmarks)
    - threshold = 0: pure sort + top_k truncation (mem0 style, default)

    Reuses the head/tail split: head (top CROSS_ENCODER_CANDS by vector score)
    is sent to the scorer, tail is appended after kept_head; the caller
    truncates to top_k.

    Returns LLMRerankResult with kept = (kept_head + tail), dropped = dropped_head.
    """
    if not documents:
        return LLMRerankResult(kept=[], dropped=[])

    head, tail = _head_tail_split(documents, CROSS_ENCODER_CANDS)

    doc_texts = [
        _truncate(d.get("memory", "") or "", CROSS_ENCODER_CAND_TEXT_LEN)
        for d in head
    ]

    if CROSS_ENCODER_PROVIDER == "local":
        scores = _call_local_rerank(query, doc_texts)
    else:
        scores = _call_rerank_api(query, doc_texts)

    # Attach rerank score to each head doc and sort by it desc.
    scored_head = []
    for doc, score in zip(head, scores):
        doc_copy = dict(doc)
        doc_copy["rerank_score"] = score
        scored_head.append(doc_copy)
    scored_head.sort(key=lambda x: x["rerank_score"], reverse=True)

    if CROSS_ENCODER_REL_THRESHOLD > 0 and scored_head:
        # Filter style: drop docs below ratio * top_score.
        top_score = scored_head[0]["rerank_score"]
        threshold = top_score * CROSS_ENCODER_REL_THRESHOLD
        kept_head = [d for d in scored_head if d["rerank_score"] >= threshold]
        dropped_head = [d for d in scored_head if d["rerank_score"] < threshold]
        logger.info(
            "[cross_encoder] filter mode: top=%.4f thresh=%.4f kept=%d/%d",
            top_score, threshold, len(kept_head), len(scored_head),
        )
    else:
        # mem0 style: pure sort, no filter. dropped_head is empty.
        kept_head = scored_head
        dropped_head = []
        if scored_head:
            logger.info(
                "[cross_encoder] sort-only (threshold=0): top=%.4f kept=%d/%d",
                scored_head[0]["rerank_score"], len(kept_head), len(scored_head),
            )

    # Same as listwise: tail goes into kept (not dropped) so caller [:top_k]
    # can pick them up. Caller truncates to top_k after.
    kept = kept_head + tail
    return LLMRerankResult(kept=kept, dropped=dropped_head)
