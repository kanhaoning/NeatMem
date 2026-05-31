"""搜索结果 rerank 模块"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

RERANK_PROMPT = """判断这条记忆是否与用户的查询相关。

查询："{query}"
记忆："{document}"

判断标准：
- 相关：记忆包含能帮助回答查询的信息，或与查询的话题直接相关
- 不相关：记忆与查询无关，或只是碰巧有少量词汇重叠但话题不同

只输出一个数字：1（相关）或 0（不相关）。不要输出其他内容。"""

_MAX_WORKERS = 4
_CALL_TIMEOUT = 6.0


def llm_rerank(llm, query: str, documents: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    """LLM 二分类过滤：保留相关记忆，踢掉无关记忆

    Args:
        llm: mem0 LLM 实例
        query: 搜索查询
        documents: 向量搜索返回的候选，每条至少含 "memory" 和 "score"
        top_k: 最多返回条数

    Returns:
        过滤+排序后的文档列表，每条新增 "rerank_score" 字段（0.0 或 1.0）
    """
    if not documents:
        return documents

    verdicts = _judge_batch(llm, query, [doc.get("memory", "") for doc in documents])

    for doc, verdict in zip(documents, verdicts):
        doc["rerank_score"] = float(verdict)

    relevant = [d for d in documents if d["rerank_score"] >= 0.5]
    relevant.sort(key=lambda x: x.get("score", 0), reverse=True)
    return relevant[:top_k]


def _judge_single(llm, query: str, document: str) -> int:
    """单条判断，异常时返回 0（不相关）"""
    prompt = RERANK_PROMPT.format(query=query, document=document)
    try:
        response = llm.generate_response(
            messages=[{"role": "user", "content": prompt}],
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return 1 if response.strip() == "1" else 0
    except Exception:
        return 0


def _judge_batch(llm, query: str, documents: List[str]) -> List[int]:
    if not documents:
        return []

    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
        futures = {
            executor.submit(_judge_single, llm, query, doc): i
            for i, doc in enumerate(documents)
        }
        results = [0] * len(documents)
        for future in as_completed(futures, timeout=_CALL_TIMEOUT * len(documents)):
            idx = futures[future]
            try:
                results[idx] = future.result(timeout=_CALL_TIMEOUT)
            except Exception:
                results[idx] = 0
        return results
