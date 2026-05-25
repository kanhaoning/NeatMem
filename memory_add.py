"""
自研 add 接口核心模块：LLM 提取 + 语义去重 + 写入
存储层复用 mem0 的 add(infer=False) / search / update 公开 API
"""

import asyncio
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from prompts.extraction import (
    ADDITIVE_EXTRACTION_PROMPT,
    generate_additive_extraction_prompt,
)
from mem0.memory.utils import extract_json, remove_code_blocks

logger = logging.getLogger(__name__)

# 同批次上下文补全规则：替换模糊指代（方向 F，仿真验证 2/3 Pass）
_BATCH_CONTEXT_RULE = (
    "CRITICAL OVERRIDE — Vague Reference Replacement: "
    "When a user's message contains a vague or generic reference that clearly "
    "refers to a specific entity mentioned in an earlier message IN THE SAME BATCH "
    "of New Messages, you MUST REPLACE the vague reference with the specific entity "
    "in your extracted memory. Do NOT preserve the user's vague wording. "
    "A 'vague reference' is any phrase like 'during development', 'in the process', "
    "'the project', 'that thing', 'this work', '开发过程中', '开发时', '开发中', "
    "'在这个过程中', '这个项目' — these are placeholder expressions that the user "
    "used for brevity, but they LOSE critical information when stored as memory. "
    "You MUST substitute them with the actual specific entity from the earlier message. "
    "Example: if the first new message says 'I am developing a mem0 memory module' "
    "and a later new message says 'ran into a duplicate memory issue during development', "
    "the second extraction MUST be 'User ran into a duplicate memory issue while "
    "developing the mem0 memory module' — the vague phrase 'during development' "
    "MUST BE REPLACED with 'while developing the mem0 memory module'. "
    "Preserving 'during development' in the extracted memory is WRONG because it "
    "discards the specific information (mem0, memory module) that makes the memory "
    "independently understandable. The user used 'during development' as a shorthand, "
    "not as a deliberate choice to be vague in permanent memory."
)

# 向量召回阈值（宽松，宁可多召回不漏）
DEDUP_RECALL_THRESHOLD = 0.40
# LLM 并发线程数（IO 密集型，4 线程足够）
DEDUP_MAX_WORKERS = 4

# 关系判断 prompt（三分类：redundant / relevant / independent）
RELATION_PROMPT = """判断两条记忆之间的关系。

记忆A：{memory_a}
记忆B：{memory_b}

判断标准：
- redundant：两条记忆表达同一事实，信息重叠，无独特信息差异
- relevant：关于同一话题，但至少一条有另一条没有的独特信息（补充、更新、纠正都算）
- independent：不同话题，或同一话题的不同独立方面，互不矛盾也不重叠

示例：
✓ redundant: "喜欢猫" vs "喜欢猫咪" — 同一事实不同说法，无信息差异
✓ redundant: "GPU坏了" vs "GPU坏了，改用硅基流动" — 后者完全包含前者信息
✗ relevant: "开发记忆模块" vs "用Python和FastAPI开发mem0记忆模块，遇到重复问题" — 同一话题，后者有独特细节
✗ relevant: "住在北京XX路" vs "搬到了上海" — 同一话题，后者是更新
✗ relevant: "养了三只猫" vs "养了两只猫" — 同一话题，数量矛盾
✗ independent: "喜欢猫" vs "养了三只猫" — 不同事实
✗ independent: "会弹钢琴" vs "会弹吉他" — 不同话题

只输出 JSON：{{"result": "redundant"}} 或 {{"result": "relevant"}} 或 {{"result": "independent"}}"""

# 记忆合并 prompt
MERGE_PROMPT = """将两条关于同一话题的记忆合并为一条完整记忆。

规则：
- 保留两条记忆的所有独特信息，不要丢弃任何一条的独有内容
- 如果新信息补充了旧信息，将补充内容融入
- 如果新信息与旧信息矛盾，以新信息为准，保留旧信息但标注已过时
- 输出自然流畅的陈述，不要有拼接痕迹
- 不要编造任何两条记忆中都没有的信息
- 如果无法合并（如信息完全无关或无法调和），result 设为 cannot_merge

旧记忆：{old_text}
新记忆：{new_text}

只输出 JSON：{{"result": "merged", "text": "合并后的记忆文本"}} 或 {{"result": "cannot_merge"}}"""


def strip_thinking(text: str) -> str:
    """剥离 LLM 思考标签，兼容各厂商"""
    return re.sub(r"<think\b[^>]*>.*?</think\s*>", "", text, flags=re.DOTALL).strip()


def check_relation(llm, text_a: str, text_b: str) -> str:
    """用 LLM 判断两条记忆的关系，返回 'redundant' / 'relevant' / 'independent'"""
    prompt = RELATION_PROMPT.format(memory_a=text_a, memory_b=text_b)
    response = llm.generate_response(
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    response = strip_thinking(response)
    try:
        result = json.loads(response, strict=False)
        verdict = result.get("result", "independent")
        # 兼容旧二分类输出（过渡期）
        if verdict == "duplicate":
            verdict = "redundant"
        elif verdict == "not_duplicate":
            verdict = "independent"
        # 只接受合法分类
        if verdict not in ("redundant", "relevant", "independent"):
            verdict = "independent"
        return verdict
    except (json.JSONDecodeError, KeyError):
        return "independent"


def check_relation_batch(llm, pairs: List[tuple]) -> List[str]:
    """并发判断多对记忆的关系，返回 verdict 列表（'redundant'/'relevant'/'independent'）"""
    if not pairs:
        return []

    with ThreadPoolExecutor(max_workers=DEDUP_MAX_WORKERS) as executor:
        futures = {
            executor.submit(check_relation, llm, a, b): i
            for i, (a, b) in enumerate(pairs)
        }
        results = ["independent"] * len(pairs)
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception:
                results[idx] = "independent"
        return results


def merge_memories(llm, old_text: str, new_text: str) -> str | None:
    """用 LLM 合并两条记忆，返回合并后文本；失败返回 None"""
    prompt = MERGE_PROMPT.format(old_text=old_text, new_text=new_text)
    try:
        response = llm.generate_response(
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        response = strip_thinking(response)
        result = json.loads(response, strict=False)
        if result.get("result") == "cannot_merge":
            return None
        merged_text = result.get("text", "")
        if not merged_text or len(merged_text) < 5:
            return None
        return merged_text
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"[合并记忆] JSON 解析失败: {e}")
        return None
    except Exception as e:
        logger.warning(f"[合并记忆] LLM 合并失败: {e}")
        return None


@dataclass
class DedupResult:
    """去重结果数据结构"""
    to_add: List[Dict[str, Any]] = field(default_factory=list)
    duplicates: List[Dict[str, Any]] = field(default_factory=list)
    merged: List[Dict[str, Any]] = field(default_factory=list)


def extract_memories(
    llm,
    messages: List[Dict[str, Any]],
    existing_memories: List[Dict[str, Any]],
    search_filters: Dict[str, Any],
    custom_instructions: Optional[str] = None,
    req_id: str = "",
) -> List[Dict[str, Any]]:
    """Step 2: LLM 提取记忆，复用 mem0 的 ADDITIVE_EXTRACTION_PROMPT"""

    prefix = f"[{req_id} 提取]" if req_id else "[提取]"

    # 构建已有记忆列表（id + text）
    existing_mem_list = [
        {"id": m["id"], "text": m["memory"]}
        for m in existing_memories
        if "id" in m and "memory" in m
    ]

    # 将同批次上下文补全规则追加到 custom_instructions
    effective_instructions = custom_instructions or ""
    if effective_instructions:
        effective_instructions = f"{effective_instructions}\n\n{_BATCH_CONTEXT_RULE}"
    else:
        effective_instructions = _BATCH_CONTEXT_RULE

    # 生成 prompt
    system_prompt = ADDITIVE_EXTRACTION_PROMPT
    user_prompt = generate_additive_extraction_prompt(
        existing_memories=existing_mem_list,
        new_messages=messages,
        current_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        custom_instructions=effective_instructions,
        use_input_language=True,
    )

    # 调用 LLM
    t0 = time.monotonic()
    response = llm.generate_response(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )
    llm_ms = (time.monotonic() - t0) * 1000

    # 解析响应（与 mem0 内部解析逻辑一致）
    response = remove_code_blocks(response)
    if not response or not response.strip():
        logger.info(f"{prefix} LLM 返回空，耗时 {llm_ms:.0f}ms")
        return []

    try:
        extracted = json.loads(response, strict=False).get("memory", [])
    except json.JSONDecodeError:
        extracted_json = extract_json(response)
        extracted = json.loads(extracted_json, strict=False).get("memory", [])

    logger.info(f"{prefix} LLM 提取出 {len(extracted)} 条记忆, 耗时 {llm_ms:.0f}ms")
    for i, mem in enumerate(extracted):
        logger.info(f"{prefix}   #{i+1}: attr={mem.get('attributed_to', '?')}, text='{mem.get('text', '')[:120]}'")

    return extracted


def dedup_memories(
    memory,
    llm,
    extracted_memories: List[Dict[str, Any]],
    search_filters: Dict[str, Any],
    req_id: str = "",
) -> DedupResult:
    """Step 3: 语义去重 — 逐条串行处理：搜索 → LLM判断 → merge/写入 → 下一条

    串行保证每条新记忆搜索时看到的都是最新状态，
    避免多条新记忆命中同一旧记忆时互相覆盖。
    """

    result = DedupResult()
    total = len(extracted_memories)
    prefix = f"[{req_id} 去重]" if req_id else "[去重]"

    for idx, new_mem in enumerate(extracted_memories, 1):
        new_text = new_mem.get("text", "")
        if not new_text:
            continue

        new_attr = new_mem.get("attributed_to", "user")
        tag = f"{prefix} #{idx}/{total}"

        # --- 搜索候选（每次都搜最新状态） ---
        dedup_filters = {**search_filters, "attr_source": new_attr}
        t0 = time.monotonic()
        search_result = memory.search(
            query=new_text,
            limit=5,
            filters=dedup_filters,
            rerank=False,
        )
        hits = search_result.get("results", [])
        search_ms = (time.monotonic() - t0) * 1000

        candidates = [h for h in hits if h.get("score", 0) >= DEDUP_RECALL_THRESHOLD]
        logger.info(
            f"{tag} 搜索 | 召回 {len(hits)} 条(阈值以上 {len(candidates)} 条), "
            f"耗时 {search_ms:.0f}ms | 新记忆: '{new_text[:120]}'"
        )
        for h in candidates:
            logger.info(
                f"{tag}   候选: score={h.get('score', 0):.4f} | '{h.get('memory', '')[:120]}'"
            )

        if not candidates:
            result.to_add.append(new_mem)
            logger.info(f"{tag} → 新增(无候选)")
            continue

        # --- 归因过滤 + 构建 LLM 判断对 ---
        pairs = []
        pair_candidates = []
        for cand in candidates:
            cand_attr = cand.get("metadata", {}).get("attr_source")
            if cand_attr and cand_attr != new_attr:
                logger.info(f"{tag}   归因隔离: [{new_attr}] vs [{cand_attr}] → independent")
                continue
            pairs.append((new_text, cand.get("memory", "")))
            pair_candidates.append(cand)

        if not pairs:
            result.to_add.append(new_mem)
            logger.info(f"{tag} → 新增(候选全被归因隔离)")
            continue

        # --- LLM 关系判断 ---
        t0 = time.monotonic()
        verdicts = check_relation_batch(llm, pairs)
        judge_ms = (time.monotonic() - t0) * 1000

        for cand, verdict in zip(pair_candidates, verdicts):
            logger.info(
                f"{tag}   判断({judge_ms:.0f}ms): vs '{cand.get('memory', '')[:80]}' → {verdict}"
            )

        # --- 按优先级取最高关系：relevant > redundant > independent ---
        best_verdict = "independent"
        best_cand = None
        for cand, verdict in zip(pair_candidates, verdicts):
            if verdict == "relevant":
                best_verdict = verdict
                best_cand = cand
                break  # relevant 最高，直接取
            if verdict == "redundant" and best_verdict == "independent":
                best_verdict = verdict
                best_cand = cand

        # --- 执行判定 ---
        if best_verdict == "independent":
            result.to_add.append(new_mem)
            logger.info(f"{tag} → 新增(有候选但独立)")
            continue

        cand = best_cand
        old_text = cand.get("memory", "")

        if best_verdict == "redundant":
            cand_attr = cand.get("metadata", {}).get("attr_source", "user")
            memory.update(memory_id=cand["id"], data=new_text, metadata={"attr_source": cand_attr})
            result.duplicates.append({
                "new_text": new_text,
                "old_id": cand["id"],
                "old_text": old_text,
                "score": cand.get("score"),
                "relation": "redundant",
            })
            logger.info(f"{tag} → 冗余替换: '{old_text[:80]}' → '{new_text[:80]}'")

        elif best_verdict == "relevant":
            t0 = time.monotonic()
            merged = merge_memories(llm, old_text, new_text)
            merge_ms = (time.monotonic() - t0) * 1000

            if merged:
                memory.update(memory_id=cand["id"], data=merged, metadata={"attr_source": new_attr})
                result.merged.append({
                    "new_text": new_text,
                    "old_id": cand["id"],
                    "old_text": old_text,
                    "merged_text": merged,
                    "score": cand.get("score"),
                    "relation": "relevant",
                })
                logger.info(
                    f"{tag} → 合并({merge_ms:.0f}ms): '{old_text[:80]}' + '{new_text[:80]}' → '{merged[:120]}'"
                )
            else:
                memory.update(memory_id=cand["id"], data=new_text, metadata={"attr_source": new_attr})
                result.duplicates.append({
                    "new_text": new_text,
                    "old_id": cand["id"],
                    "old_text": old_text,
                    "score": cand.get("score"),
                    "relation": "relevant_fallback",
                })
                logger.warning(
                    f"{tag} → 合并失败({merge_ms:.0f}ms), 兜底替换: '{old_text[:80]}' → '{new_text[:80]}'"
                )

    logger.info(
        f"{prefix} 完成 | 新增 {len(result.to_add)} 条, "
        f"冗余替换 {len(result.duplicates)} 条, "
        f"合并 {len(result.merged)} 条"
    )
    return result


def add_memories(
    memory,
    llm,
    messages: List[Dict[str, Any]],
    user_id: str,
    agent_id: Optional[str] = None,
    run_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    custom_instructions: Optional[str] = None,
    req_id: str = "",
) -> Dict[str, Any]:
    """串联 Step 1-4：搜索 → 提取 → 去重 → 写入"""

    prefix = f"[{req_id}]" if req_id else ""

    # 构建 search filters
    search_filters = {}
    if user_id:
        search_filters["user_id"] = user_id
    if agent_id:
        search_filters["agent_id"] = agent_id

    # Step 1: 搜索已有记忆
    t0 = time.monotonic()
    messages_text = json.dumps(messages, ensure_ascii=False)
    search_result = memory.search(
        query=messages_text,
        limit=10,
        filters=search_filters,
        rerank=False,
    )
    existing_memories = search_result.get("results", [])
    step1_ms = (time.monotonic() - t0) * 1000
    logger.info(f"{prefix}[Step 1] 搜索已有记忆 | 找到 {len(existing_memories)} 条, 耗时 {step1_ms:.0f}ms")

    # Step 2: LLM 提取记忆
    t0 = time.monotonic()
    extracted = extract_memories(
        llm=llm,
        messages=messages,
        existing_memories=existing_memories,
        search_filters=search_filters,
        custom_instructions=custom_instructions,
        req_id=req_id,
    )
    step2_ms = (time.monotonic() - t0) * 1000
    logger.info(f"{prefix}[Step 2] LLM 提取完成 | {len(extracted)} 条, 耗时 {step2_ms:.0f}ms")

    if not extracted:
        logger.info(f"{prefix} 未提取到任何记忆，结束")
        return {"results": [], "duplicates": []}

    # Step 3: 语义去重
    t0 = time.monotonic()
    dedup_result = dedup_memories(
        memory=memory,
        llm=llm,
        extracted_memories=extracted,
        search_filters=search_filters,
        req_id=req_id,
    )
    step3_ms = (time.monotonic() - t0) * 1000
    logger.info(f"{prefix}[Step 3] 语义去重完成, 耗时 {step3_ms:.0f}ms")

    # Step 4: 写入向量库
    if dedup_result.to_add:
        t0 = time.monotonic()
        logger.info(f"{prefix}[Step 4] 写入 {len(dedup_result.to_add)} 条新记忆...")
        added_memories = []
        for mem in dedup_result.to_add:
            mem_metadata = {
                "dedup_count": 0,
                "category": "",
                "source": "custom_add",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "attr_source": mem.get("attributed_to", "user"),
            }
            if run_id:
                mem_metadata["run_id"] = run_id
            if metadata:
                mem_metadata.update(metadata)

            add_params = {
                "messages": [{"role": "user", "content": mem["text"]}],
                "user_id": user_id,
                "infer": False,
                "metadata": mem_metadata,
            }
            if agent_id:
                add_params["agent_id"] = agent_id

            add_result = memory.add(**add_params)
            added_memories.extend(add_result.get("results", []))

        step4_ms = (time.monotonic() - t0) * 1000
        logger.info(f"{prefix}[Step 4] 写入完成 | 实际写入 {len(added_memories)} 条, 耗时 {step4_ms:.0f}ms")
    else:
        added_memories = []
        logger.info(f"{prefix}[Step 4] 无新记忆需要写入")

    return {
        "results": added_memories,
        "duplicates": dedup_result.duplicates,
        "merged": dedup_result.merged,
    }
