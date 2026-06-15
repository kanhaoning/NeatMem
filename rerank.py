"""搜索结果 rerank 模块 — listwise LLM rerank + 统一入口

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

from utils.llm_client import build_thinking_extra, extract_response_text

RERANK_MODE = os.environ.get("RERANK_MODE", "llm_listwise")


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
    """统一入口，根据 RERANK_MODE 分发"""
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

_LISTWISE_PROMPT = """判断以下候选记忆是否与用户的查询相关，并返回所有相关记忆的编号（按相关度从高到低排序）。

查询："{query}"

候选记忆列表：
{candidates_text}

任务：
1. 逐条评估每条记忆与查询的相关性。注意：相关性不限于字面匹配，如果记忆包含能间接推断出答案的关键事实，也应视为相关。
2. 特别关注查询所询问的具体信息（如具体书名、活动名称、日期、地点、数字、特定事物等）。即使记忆的整体话题与查询不完全一致，只要包含这些关键实体，也应视为相关。
3. 如果多条记忆共同支持同一个推断结论，请尽可能保留这些证据，不要视为重复而只选一条。
4. **只返回确实相关的记忆**。不相关的记忆不要出现在列表中。
5. 将相关记忆按相关度从高到低排序，编号小的是最相关，编号大的是次相关。

输出格式（严格 JSON）：
{{"analysis": "简要说明筛选依据", "relevant": [编号1, 编号2, ...]}}

只输出 JSON，不要其他内容。"""


def _build_candidates_text(documents: List[Dict[str, Any]]) -> str:
    lines = []
    for i, doc in enumerate(documents, 1):
        text = doc.get("memory", "")[:120].replace("\n", " ")
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
        top_k: 下游 answer 阶段期望的条数（用于计算 cap = top_k * 2）

    Returns:
        (保留的文档列表, 丢弃的文档列表)
    """
    if not documents:
        return documents, []

    # 截断到前 20 条（避免注意力稀释）
    MAX_CANDS = 20
    sorted_docs = sorted(documents, key=lambda x: x.get("score", 0), reverse=True)
    truncated = sorted_docs[:MAX_CANDS]
    rest = sorted_docs[MAX_CANDS:]

    candidates_text = _build_candidates_text(truncated)
    prompt = _LISTWISE_PROMPT.format(query=query, candidates_text=candidates_text)

    try:
        resp = openai_client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=2000,
            extra_body=build_thinking_extra(llm_model, enable=False),
        )
        response = extract_response_text(resp) or ""
    except Exception as e:
        print(f"  [WARN] Listwise rerank failed: {e}, fallback to score-based")
        return truncated[:top_k], truncated[top_k:] + rest

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
        return truncated[:top_k], truncated[top_k:] + rest

    # 转为 0-based 索引并映射到文档
    selected = []
    for idx in relevant_indices:
        try:
            i = int(idx) - 1  # 编号是 1-based
            if 0 <= i < len(truncated):
                selected.append(truncated[i])
        except (ValueError, TypeError):
            continue

    # 去重
    seen = set()
    kept = []
    for d in selected:
        mem = d.get("memory", "")
        if mem not in seen:
            seen.add(mem)
            kept.append(d)

    # 如果 LLM 返回为空，紧急 fallback
    if not kept:
        return truncated[:top_k], truncated[top_k:] + rest

    # 上限 = top_k * 2，避免 flooding answer 模型
    cap = top_k * 2
    kept = kept[:cap]

    kept_mems = set(d.get("memory", "") for d in kept)
    dropped_truncated = [d for d in truncated if d.get("memory", "") not in kept_mems]
    dropped = dropped_truncated + rest

    return kept, dropped


# 保持与实验命名一致
_llm_rerank_listwise_v2 = _llm_rerank_listwise
