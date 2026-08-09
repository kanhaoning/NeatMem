"""搜索结果 rerank 模块 - listwise LLM rerank + 统一入口

RERANK_MODE 环境变量：
- off: 纯向量检索，不调用 LLM
- llm_listwise: LLM listwise rerank（默认）
- llm_listwise_v2: 与 llm_listwise 同逻辑
- cross_encoder: 预留，暂不实现
"""

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from neatmem.prompts.loader import load_prompt
from neatmem.utils.llm_client import extract_response_text, complete_chat

# Explicit provider for thinking-shape/constraint selection; None = legacy
# model-name matching (import-free env read keeps rerank importable standalone).
LLM_PROVIDER = os.environ.get("LLM_PROVIDER")

RERANK_MODE = os.environ.get("RERANK_MODE", "llm_listwise")

# rerank head 大小：LLM listwise 只重排前 N 条，其余原样附加（signetai head/tail 设计）。
# 默认 20（延续原 MAX_CANDS 行为），env 可覆盖。
RERANK_CANDS = int(os.environ.get("RERANK_CANDS", "20"))

# 候选记忆截断长度，默认 120；0=不截断
RERANK_CAND_TEXT_LEN = int(os.environ.get("RERANK_CAND_TEXT_LEN", "120"))


@dataclass
class LLMRerankResult:
    kept: List[Dict[str, Any]]           # 保留的记忆（已排序）
    dropped: List[Dict[str, Any]]        # 丢弃的记忆
    raw_response: Optional[str] = None   # LLM 原始输出，供后续可解释性


# ---------------------------------------------------------------------------
# 统一入口
# ---------------------------------------------------------------------------

def llm_rerank(openai_client, llm_model: str, query: str,
               documents: List[Dict[str, Any]], top_k: int = 5) -> LLMRerankResult:
    """统一入口，根据 RERANK_MODE 分发

    返回的 kept 包含 head 重排结果 + tail 原序附加。调用方需自行 [:top_k] 截断。
    """
    if RERANK_MODE == "off":
        return LLMRerankResult(kept=documents[:top_k], dropped=documents[top_k:])
    elif RERANK_MODE in ("llm_listwise", "llm_listwise_v2"):
        kept, dropped = _llm_rerank_listwise(openai_client, llm_model, query, documents, top_k)
        return LLMRerankResult(kept=kept, dropped=dropped)
    else:
        # 未知模式 fallback 到纯截断
        return LLMRerankResult(kept=documents[:top_k], dropped=documents[top_k:])


# ---------------------------------------------------------------------------
# Listwise rerank v2
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
    """填充 rerank prompt。"""
    # RERANK_PROMPT overrides the built-in template (loaded and cached on first call).
    template = load_prompt("RERANK_PROMPT", _LISTWISE_PROMPT, ("query", "candidates_text"))
    return template.format(query=query, candidates_text=candidates_text)


def _build_candidates_text(documents: List[Dict[str, Any]]) -> str:
    lines = []
    for i, doc in enumerate(documents, 1):
        text = doc.get("memory", "")
        if RERANK_CAND_TEXT_LEN > 0:
            text = text[:RERANK_CAND_TEXT_LEN]
        text = text.replace("\n", " ")
        lines.append(f"[{i}] {text}")
    return "\n".join(lines)


def _parse_json(text: str) -> Dict[str, Any]:
    """从 LLM 输出中提取 JSON"""
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
    """LLM listwise rerank：让 LLM 筛选所有相关记忆并排序，不强制 top_k

    Args:
        openai_client: OpenAI 客户端实例
        llm_model: LLM 模型 ID
        query: 搜索查询
        documents: 向量搜索返回的候选，每条至少含 "memory" 和 "score"
        top_k: 下游 answer 阶段期望的条数（用于 fallback key 名匹配）

    Returns:
        (kept, dropped)
        - kept = LLM 从 head 中选出的相关记忆 + tail 原序附加
        - dropped = head 中未被 LLM 选中的记忆
        调用方需对 kept 做 [:top_k] 截断。
    """
    if not documents:
        return documents, []

    # head/tail 设计（借鉴 signetai）：head 送 LLM 重排，tail 原样附加不丢数据
    sorted_docs = sorted(documents, key=lambda x: x.get("score", 0), reverse=True)
    head = sorted_docs[:RERANK_CANDS]
    tail = sorted_docs[RERANK_CANDS:]

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

    # 首选 "relevant" 数组
    relevant_indices = None
    for key in ["relevant", "selected", "related", "indices"]:
        if key in parsed and isinstance(parsed[key], list):
            relevant_indices = parsed[key]
            break

    # Fallback 1：尝试 top_k 风格的键名
    if not relevant_indices or not isinstance(relevant_indices, list):
        for key in [f"top{top_k}", "top5", "top10", "top_k", "top"]:
            if key in parsed and isinstance(parsed[key], list):
                relevant_indices = parsed[key]
                break

    if relevant_indices is None or not isinstance(relevant_indices, list):
        print(f"  [WARN] Listwise parse failed, response: {response[:200]}")
        return head + tail, []

    # 转为 0-based 索引并映射到文档
    selected = []
    for idx in relevant_indices:
        try:
            i = int(idx) - 1  # 编号是 1-based
            if 0 <= i < len(head):
                selected.append(head[i])
        except (ValueError, TypeError):
            continue

    # 去重
    seen = set()
    kept_head = []
    for d in selected:
        mem = d.get("memory", "")
        if mem not in seen:
            seen.add(mem)
            kept_head.append(d)

    # 如果 LLM 返回为空，紧急 fallback：全部 head + tail 原序返回
    if not kept_head:
        return head + tail, []

    # 不再 cap*2。head 重排结果 + tail 原序附加，由调用方 [:top_k] 截断
    kept_mems = set(d.get("memory", "") for d in kept_head)
    dropped_head = [d for d in head if d.get("memory", "") not in kept_mems]

    # tail 必须放进 kept（不是 dropped），否则调用方只取 kept 时 tail 丢失
    kept = kept_head + tail
    dropped = dropped_head

    return kept, dropped


# 保持与实验命名一致
_llm_rerank_listwise_v2 = _llm_rerank_listwise
