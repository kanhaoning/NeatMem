"""
自研 add 接口核心模块：LLM 提取 + 语义去重 + 写入
存储层复用 mem0 的 add(infer=False) / search / update 公开 API
"""

import asyncio
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from neatmem.utils.llm_client import build_thinking_extra
from neatmem.signals.entity.base import AbstractEntityExtractor
from neatmem.storage.entity.base import AbstractEntityStore
from neatmem.prompts.extraction import (
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
# 合并策略：rewrite | patch_diff | off
#   rewrite    — 全文重写（MERGE_PROMPT）
#   patch_diff — C3 PatchDiffStrict，失败 fallback 保留两条
#   off        — relevant_merge 视为 relevant_link，保留两条
MERGE_STRATEGY = os.environ.get("MERGE_STRATEGY", "off")

# 去重阶段关系分类器模式
#   pointwise_4class — 现有 pairwise 4-class（默认）
#   listwise_4class  — listwise 4-class，prompt 与 pointwise 规则对齐
DEDUP_MODE = os.environ.get("DEDUP_MODE", "pointwise_4class")

# 关系判断 prompt（四分类：redundant / relevant_merge / relevant_link / independent）
# 修改记录:
#   - 2026-05-30 [P0-c]: 三分类→四分类，拆 relevant 为 merge/link。
#     问题：relevant 一律 merge 导致记忆越合越长，语义锚点稀释，
#     检索召回后 answer LLM 无法精准提取。
#     根因："同一话题"不等于"应该合并"，同话题的不同事件/经历各有独立检索价值。
#     方案：用"删除测试"区分——删掉A后B是否失去独立可检索的信息？是则link，否则merge。
#     详见 docs/internal-notes/20260530-relation-four-class-plan.md
RELATION_PROMPT = """判断两条记忆之间的关系。

记忆A：{memory_a}
记忆B：{memory_b}

判断标准：
- redundant：同一事实，信息重叠，无独特差异
- relevant_merge：同一具体事实的更新或补充（换地址、加细节、纠正）
- relevant_link：同一大话题但不同事实/事件，各自有独立语义锚点
- independent：不同话题

区分 relevant_merge 和 relevant_link 的关键——删除测试：
如果删除记忆A，记忆B是否失去了一条独立可检索的信息？
- 是 → relevant_link（A有独立锚点，不应被合并进B）
- 否 → relevant_merge（B已包含A的核心信息，或A只是B的细节补充）

示例：
✓ redundant: "喜欢猫" vs "喜欢猫咪" — 同一事实
✓ redundant: "GPU坏了" vs "GPU坏了，改用硅基流动" — 后者完全包含前者
✓ relevant_merge: "住在北京" vs "搬到了上海" — 同一事实（住址）的更新，删掉"住北京"不影响"住上海"的独立性
✓ relevant_merge: "在开发mem0记忆模块" vs "用Python和FastAPI开发mem0记忆模块，遇到重复问题" — 后者是前者的补充细节
✗ relevant_link: "工作时感到overwhelmed" vs "意识到self-care很重要" — 同话题但不同事件，删掉前者后者仍是一条独立记忆
✗ relevant_link: "参加了慈善跑步" vs "开始每天跑步和读书" — 同话题(自我关爱)但不同事件
✗ relevant_link: "喜欢画画" vs "开始学小提琴" — 同话题(爱好)但不同事实
✗ independent: "喜欢猫" vs "养了三只猫" — 不同事实
✗ independent: "会弹钢琴" vs "会弹吉他" — 不同话题

只输出 JSON：{{"result": "redundant"}} 或 {{"result": "relevant_merge"}} 或 {{"result": "relevant_link"}} 或 {{"result": "independent"}}"""

# Fair listwise 4-class prompt：与 pointwise 使用相同的定义、删除测试和示例，
# 仅在输入格式上改为 listwise，用于公平对比两种格式本身的能力差异。
FAIR_FOURCLASS_LISTWISE_PROMPT = """判断新记忆与候选记忆集合之间的关系。

新记忆：{new_memory}

候选记忆列表：
{candidates_text}

判断标准：
- redundant：同一事实，信息重叠，无独特差异
- relevant_merge：同一具体事实的更新或补充（换地址、加细节、纠正）
- relevant_link：同一大话题但不同事实/事件，各自有独立语义锚点
- independent：不同话题

区分 relevant_merge 和 relevant_link 的关键——删除测试：
如果删除候选记忆，新记忆是否失去了一条独立可检索的信息？
- 是 → relevant_link（该候选有独立锚点，不应被合并进新记忆）
- 否 → relevant_merge（新记忆已包含该候选的核心信息，或该候选只是细节补充）

示例 1：
新记忆："User 搬到了上海"
候选：
[1] "User 住在北京" → relevant_merge（删除[1]不影响"搬到了上海"的独立性）
[2] "User 最近搬到上海浦东" → redundant
[3] "User 会弹钢琴" → independent

示例 2：
新记忆："User 开始每天跑步和读书"
候选：
[1] "User 参加了慈善跑步" → relevant_link（删除[1]会丢失"慈善跑步"这条独立信息）
[2] "User 喜欢跑步" → redundant
[3] "User 开始学小提琴" → independent

示例 3：
新记忆："User 用 Python 和 FastAPI 开发 mem0 记忆模块，遇到重复问题"
候选：
[1] "User 在开发 mem0 记忆模块" → relevant_merge
[2] "User 用 Python 开发 mem0 记忆模块" → redundant
[3] "User GPU 坏了，改用硅基流动" → independent

输出严格 JSON：
{{"relations": [{{"idx": 1, "relation": "relevant_merge"}}, {{"idx": 2, "relation": "redundant"}}, ...]}}"""


def _parse_listwise_response(response: str, n: int) -> List[str]:
    """解析 listwise 4-class 返回的 JSON，返回 n 条候选的 verdict 列表"""
    try:
        parsed = json.loads(response, strict=False)
    except json.JSONDecodeError:
        try:
            start = response.find("{")
            end = response.rfind("}")
            if start != -1 and end != -1:
                parsed = json.loads(response[start:end + 1], strict=False)
            else:
                return ["independent"] * n
        except json.JSONDecodeError:
            return ["independent"] * n

    relations = parsed.get("relations", [])
    verdicts = ["independent"] * n
    for item in relations:
        if not isinstance(item, dict):
            continue
        idx = item.get("idx")
        try:
            i = int(idx) - 1
        except (TypeError, ValueError):
            continue
        if 0 <= i < n:
            v = item.get("relation", "independent")
            if isinstance(v, str) and v.strip().lower() in (
                "redundant", "relevant_merge", "relevant_link", "independent"
            ):
                verdicts[i] = v.strip().lower()
    return verdicts


def check_relation_listwise_batch(
    openai_client, llm_model: str, new_text: str, candidates: List[Dict[str, Any]]
) -> List[str]:
    """用公平版 listwise 4-class prompt 判断新记忆与候选集合的关系"""
    if not candidates:
        return []

    candidates_text = "\n".join(
        f"[{i + 1}] {c.get('memory', '')}" for i, c in enumerate(candidates)
    )
    prompt = FAIR_FOURCLASS_LISTWISE_PROMPT.format(
        new_memory=new_text, candidates_text=candidates_text
    )

    try:
        resp = openai_client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=2000,
            extra_body=build_thinking_extra(llm_model, enable=True),
        )
        response = strip_thinking(resp.choices[0].message.content or "")
    except Exception as e:
        logger.warning(f"[listwise dedup] LLM 调用失败: {e}")
        return ["independent"] * len(candidates)

    return _parse_listwise_response(response, len(candidates))

# 记忆合并 prompt
# 修改记录:
#   - 2026-05-30 [P0-b]: 在规则末尾追加"动词保真"规则。
#     问题：merge LLM 合并时简化动词（"realized" → "found"，"decided" → "explored"），
#     导致语义改变。
#     详见 docs/internal-notes/20260530-extraction-merge-prompt-fix-plan.md
MERGE_PROMPT = """将两条关于同一话题的记忆合并为一条完整记忆。

规则：
- 保留两条记忆的所有独特信息，不要丢弃任何一条的独有内容
- 如果新信息补充了旧信息，将补充内容融入
- 如果新信息与旧信息矛盾，以新信息为准，保留旧信息但标注已过时
- 输出自然流畅的陈述，不要有拼接痕迹
- 不要编造任何两条记忆中都没有的信息
- 如果无法合并（如信息完全无关或无法调和），result 设为 cannot_merge
- **动词保真**：保留原始动词的语义重量，不可简化。"realized" 不可变成 "found"，"decided" 不可变成 "explored"，"quit" 不可变成 "reduced"。简化会改变原意。

旧记忆：{old_text}
新记忆：{new_text}

只输出 JSON：{{"result": "merged", "text": "合并后的记忆文本"}} 或 {{"result": "cannot_merge"}}"""

# Patch Diff 合并 prompt（C3 PatchDiffStrict）
# 来源：experiments/merge-methods/merge_test_framework.py
# 修改记录:
#   - 2026-06-09: 引入 MERGE_STRATEGY=patch_diff，从 C3 离线实验搬入生产管线
PATCH_DIFF_PROMPT = """You are a memory patch generator.

OLD MEMORY:
{old_text}

NEW INFORMATION:
{new_text}

Task: Generate a minimal patch to update the old memory with the new information.
DO NOT rewrite the entire memory. Only specify what needs to change.

Rules:
1. If new_info corrects old → "replace" with exact quote from OLD
2. If new_info adds detail → "append" with "after" referencing OLD
3. If new_info contradicts old but both may be true → "conflict", no changes
4. If completely different topic/entity/event → "unrelated", no changes. Sharing the same core entity and scene but describing a different facet is NOT unrelated — use "append".
5. NEVER rewrite unchanged text. Your "quote" and "after" must be copied from OLD.
6. PRESERVE ALL DETAILS FROM NEW INFORMATION. If NEW INFORMATION contains any specific details not in OLD MEMORY (verbs, proper nouns, emotions, time precision, activities), you MUST use "append" to include them. Do NOT use "replace" to simplify or summarize.
7. "replace" is ONLY for explicit corrections or outdated information. It must NOT remove unique details from either memory.
8. Prefer multiple small appends over one large replace.

Output JSON only:
{{"relationship": "update|append|conflict|unrelated", "changes": [{{"type": "replace", "quote": "...", "context": "...", "with": "..."}}, {{"type": "append", "after": "...", "text": "..."}}]}}"""


def strip_thinking(text: str) -> str:
    """剥离 LLM 思考标签，兼容各厂商"""
    return re.sub(r"<think\b[^>]*>.*?</think\s*>", "", text, flags=re.DOTALL).strip()


def check_relation(openai_client, llm_model: str, text_a: str, text_b: str) -> str:
    """用 LLM 判断两条记忆的关系，返回 'redundant' / 'relevant_merge' / 'relevant_link' / 'independent'"""
    prompt = RELATION_PROMPT.format(memory_a=text_a, memory_b=text_b)
    resp = openai_client.chat.completions.create(
        model=llm_model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        extra_body=build_thinking_extra(llm_model, enable=True),
    )
    response = strip_thinking(resp.choices[0].message.content or "")
    try:
        result = json.loads(response, strict=False)
        verdict = result.get("result", "independent")
        # 只接受合法分类
        if verdict not in ("redundant", "relevant_merge", "relevant_link", "independent"):
            verdict = "independent"
        return verdict
    except (json.JSONDecodeError, KeyError):
        return "independent"


def check_relation_batch(openai_client, llm_model: str, pairs: List[tuple]) -> List[str]:
    """并发判断多对记忆的关系，返回 verdict 列表（'redundant'/'relevant_merge'/'relevant_link'/'independent'）"""
    if not pairs:
        return []

    with ThreadPoolExecutor(max_workers=DEDUP_MAX_WORKERS) as executor:
        futures = {
            executor.submit(check_relation, openai_client, llm_model, a, b): i
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


def merge_memories(openai_client, llm_model: str, old_text: str, new_text: str) -> str | None:
    """用 LLM 合并两条记忆，返回合并后文本；失败返回 None"""
    prompt = MERGE_PROMPT.format(old_text=old_text, new_text=new_text)
    try:
        resp = openai_client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            extra_body=build_thinking_extra(llm_model, enable=True),
        )
        response = strip_thinking(resp.choices[0].message.content or "")
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


def fuzzy_find(needle: str, haystack: str, threshold: float = 0.8) -> Optional[str]:
    """模糊匹配：在 haystack 中找到与 needle 最相似的子串"""
    import difflib
    if not needle or not haystack:
        return None
    n = len(needle)
    best_ratio = 0.0
    best_match = None
    for i in range(len(haystack) - n + 1):
        window = haystack[i:i + n]
        ratio = difflib.SequenceMatcher(None, needle, window).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = window
    if best_ratio >= threshold:
        return best_match
    if needle in haystack:
        return needle
    return None


def apply_patch(old_memory: str, patch_json: str) -> tuple[str, str]:
    """应用 patch diff 到旧记忆，返回 (新文本, 状态)

    状态值：
    - "success" — patch 成功应用
    - "parse_error" — JSON 解析失败
    - "fallback_conflict" — LLM 判断为冲突
    - "fallback_unrelated" — LLM 判断为无关
    - "fallback_quote_not_found: ..." — replace 的 quote 在旧记忆中找不到
    - "fallback_after_not_found: ..." — append 的 after 在旧记忆中找不到
    """
    import difflib
    try:
        patch = json.loads(patch_json, strict=False)
    except json.JSONDecodeError:
        return old_memory, "parse_error"

    rel = patch.get("relationship", "unrelated")
    if rel in ("conflict", "unrelated"):
        return old_memory, f"fallback_{rel}"

    new_memory = old_memory
    changes = patch.get("changes", [])

    for change in changes:
        ctype = change.get("type")
        if ctype == "replace":
            quote = change.get("quote", "")
            with_text = change.get("with", "")
            best = fuzzy_find(quote, new_memory, threshold=0.8)
            if best is None:
                return old_memory, f"fallback_quote_not_found: {quote[:30]}"
            new_memory = new_memory.replace(best, with_text, 1)
        elif ctype == "append":
            after = change.get("after", "")
            text = change.get("text", "")
            best = fuzzy_find(after, new_memory, threshold=0.8)
            if best is None:
                return old_memory, f"fallback_after_not_found: {after[:30]}"
            new_memory = new_memory.replace(best, best + text, 1)

    return new_memory, "success"


def patch_merge_memories(openai_client, llm_model: str, old_text: str, new_text: str) -> tuple[Optional[str], Dict[str, Any]]:
    """用 Patch Diff (C3) 合并两条记忆

    Returns:
        (merged_text, metadata)
        merged_text: 成功时为合并后的文本，失败时为 None（由调用方决定 fallback）
        metadata: 含 patch_status, patch_raw 等
    """
    prompt = PATCH_DIFF_PROMPT.format(old_text=old_text, new_text=new_text)
    metadata = {}
    try:
        resp = openai_client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            extra_body=build_thinking_extra(llm_model, enable=True),
        )
        response = strip_thinking(resp.choices[0].message.content or "")
        response = remove_code_blocks(response)  # 剥离 ```json ... ``` 包裹（MiniMax-M3 兼容）
        metadata["patch_raw"] = response

        merged, status = apply_patch(old_text, response)
        metadata["patch_status"] = status

        if status == "success":
            return merged, metadata
        else:
            # fallback: 不替换，不合并，保留两条
            logger.info(f"[patch_diff] fallback: {status}")
            return None, metadata

    except Exception as e:
        logger.warning(f"[patch_diff] LLM 调用失败: {e}")
        metadata["patch_status"] = "llm_error"
        metadata["patch_raw"] = str(e)
        return None, metadata


def _add_related(memory, source_id: str, target_id: str, rel_type: str):
    """在 source 记忆的 metadata 中追加 related 条目（双向关联）

    直接操作 Qdrant payload，避免 memory.update() 触发重新嵌入和实体重链接。
    """
    try:
        existing = memory.vector_store.get(vector_id=source_id)
        if existing is None:
            return
        meta = existing.payload or {}
        related = meta.get("related", [])
        # 去重
        if any(r.get("id") == target_id for r in related):
            return
        # 限制最多 5 个 related
        if len(related) >= 5:
            related = related[-4:]
        related.append({"id": target_id, "type": rel_type})
        meta["related"] = related
        # 直接更新 payload，不触发嵌入和实体重链接
        memory.vector_store.update(vector_id=source_id, vector=None, payload=meta)
    except Exception as e:
        logger.warning(f"[related] 写入失败 source={source_id} target={target_id}: {e}")


@dataclass
class DedupResult:
    """去重结果数据结构"""
    to_add: List[Dict[str, Any]] = field(default_factory=list)
    duplicates: List[Dict[str, Any]] = field(default_factory=list)
    merged: List[Dict[str, Any]] = field(default_factory=list)
    link_pairs: List[Dict[str, Any]] = field(default_factory=list)  # 新增：记录关联对


def extract_memories(
    openai_client,
    llm_model: str,
    messages: List[Dict[str, Any]],
    existing_memories: List[Dict[str, Any]],
    search_filters: Dict[str, Any],
    custom_instructions: Optional[str] = None,
    req_id: str = "",
    metadata: Optional[Dict[str, Any]] = None,
    last_k_messages: Optional[List[Dict[str, Any]]] = None,
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
    # metadata["timestamp"]: 事件发生时间（event time），区别于 NeatMem 自动盖的 created_at（处理时间）
    # 缺省时 mem0 的 _resolve_dates 兜底 observation_date = current_date（行为=修复前）
    observation_ts = (metadata or {}).get("timestamp")

    system_prompt = ADDITIVE_EXTRACTION_PROMPT
    user_prompt = generate_additive_extraction_prompt(
        existing_memories=existing_mem_list,
        new_messages=messages,
        last_k_messages=last_k_messages,
        current_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        timestamp=observation_ts,
        custom_instructions=effective_instructions,
        use_input_language=True,
    )

    # 调用 LLM
    t0 = time.monotonic()
    resp = openai_client.chat.completions.create(
        model=llm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        extra_body=build_thinking_extra(llm_model, enable=True),
    )
    llm_ms = (time.monotonic() - t0) * 1000

    # 解析响应（与 mem0 内部解析逻辑一致）
    response = remove_code_blocks(resp.choices[0].message.content or "")
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
    openai_client,
    llm_model: str,
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
            top_k=5,
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
        if DEDUP_MODE == "listwise_4class":
            verdicts = check_relation_listwise_batch(
                openai_client, llm_model, new_text, pair_candidates
            )
        else:
            verdicts = check_relation_batch(openai_client, llm_model, pairs)
        judge_ms = (time.monotonic() - t0) * 1000

        for cand, verdict in zip(pair_candidates, verdicts):
            logger.info(
                f"{tag}   判断({judge_ms:.0f}ms): vs '{cand.get('memory', '')[:80]}' → {verdict}"
            )

        # --- 按优先级取最高关系：relevant_merge > redundant > relevant_link > independent ---
        priority = {"relevant_merge": 4, "redundant": 3, "relevant_link": 2, "independent": 1}
        best_verdict = "independent"
        best_cand = None
        for cand, verdict in zip(pair_candidates, verdicts):
            if priority.get(verdict, 0) > priority.get(best_verdict, 0):
                best_verdict = verdict
                best_cand = cand

        # --- 执行判定 ---
        if best_verdict == "relevant_link":
            result.to_add.append(new_mem)
            result.link_pairs.append({
                "new_text": new_text,
                "old_id": cand["id"],
                "old_text": cand.get("memory", ""),
                "relation_type": "same_topic",
                "score": cand.get("score"),
            })
            logger.info(f"{tag} → 新增+关联(same_topic): vs '{cand.get('memory', '')[:80]}'")
            continue
        elif best_verdict == "independent":
            result.to_add.append(new_mem)
            logger.info(f"{tag} → 新增(独立)")
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

        elif best_verdict == "relevant_merge":
            if MERGE_STRATEGY == "off":
                # 保留两条，不合并，标记 same_fact 关联
                result.to_add.append(new_mem)
                result.link_pairs.append({
                    "new_text": new_text,
                    "old_id": cand["id"],
                    "old_text": old_text,
                    "relation_type": "same_fact",
                    "score": cand.get("score"),
                })
                logger.info(f"{tag} → 新增+关联(same_fact): vs '{old_text[:80]}'")

            elif MERGE_STRATEGY == "patch_diff":
                t0 = time.monotonic()
                merged, pd_meta = patch_merge_memories(openai_client, llm_model, old_text, new_text)
                merge_ms = (time.monotonic() - t0) * 1000

                if merged:
                    memory.update(memory_id=cand["id"], data=merged, metadata={"attr_source": new_attr})
                    result.merged.append({
                        "new_text": new_text,
                        "old_id": cand["id"],
                        "old_text": old_text,
                        "merged_text": merged,
                        "score": cand.get("score"),
                        "relation": "relevant_merge_patch_diff",
                        "patch_status": pd_meta.get("patch_status"),
                    })
                    logger.info(
                        f"{tag} → patch合并({merge_ms:.0f}ms, {pd_meta.get('patch_status')}): "
                        f"'{old_text[:80]}' + '{new_text[:80]}' → '{merged[:120]}'"
                    )
                else:
                    # fallback: 保留两条
                    result.to_add.append(new_mem)
                    logger.info(
                        f"{tag} → patch失败({merge_ms:.0f}ms, {pd_meta.get('patch_status')}), 保留两条"
                    )

            else:  # rewrite（默认）
                t0 = time.monotonic()
                merged = merge_memories(openai_client, llm_model, old_text, new_text)
                merge_ms = (time.monotonic() - t0) * 1000

                if merged:
                    memory.update(memory_id=cand["id"], data=merged, metadata={"attr_source": new_attr})
                    result.merged.append({
                        "new_text": new_text,
                        "old_id": cand["id"],
                        "old_text": old_text,
                        "merged_text": merged,
                        "score": cand.get("score"),
                        "relation": "relevant_merge",
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
                        "relation": "relevant_merge_fallback",
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
    openai_client,
    llm_model: str,
    messages: List[Dict[str, Any]],
    user_id: str,
    agent_id: Optional[str] = None,
    run_id: Optional[str] = None,
    app_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    custom_instructions: Optional[str] = None,
    req_id: str = "",
    message_store: Optional[Any] = None,
    extract_last_k: Optional[int] = None,
    last_k_messages_input: Optional[List[Dict[str, Any]]] = None,
    entity_extractor: Optional[AbstractEntityExtractor] = None,
    entity_store: Optional[AbstractEntityStore] = None,
    bm25_index=None,
) -> Dict[str, Any]:
    """串联 Step 0-4：消息存储 → 搜索 → 提取 → 去重 → 写入

    Step 0 (lastk_before_save): 先取 last_k_messages,后 save 当前 batch,
    确保 last_k 不包含当前 batch（避免 New Messages 与 Last k Messages 重合）,
    同时保留真正的历史上下文用于指代消解。

    外部注入模式: 当 last_k_messages_input 不为 None 时,用用户传的 last_k,
    不调 save_messages/get_last_messages（用户有自己的 messages db）。
    """

    prefix = f"[{req_id}]" if req_id else ""

    # 构建 search filters
    search_filters = {}
    if user_id:
        search_filters["user_id"] = user_id
    if agent_id:
        search_filters["agent_id"] = agent_id
    if app_id:
        search_filters["app_id"] = app_id

    # Step 0: 获取最近上下文 (lastk_before_save + 外部注入优先 + 截断策略分模式)
    last_k_messages = None
    k = extract_last_k if extract_last_k is not None else getattr(message_store, "extract_last_k", 10)

    if last_k_messages_input is not None:
        # 外部注入(用户从自己 db 取的):默认不截断,用户传多少用多少
        # 只有用户显式传了 extract_last_k 才截断
        last_k_messages = last_k_messages_input
        if extract_last_k is not None and len(last_k_messages) > extract_last_k:
            last_k_messages = last_k_messages[-extract_last_k:]
        # 不调 save_messages(用户有自己的 db,NeatMem 不存)
        # 不调 get_last_messages(用户已经传了,不需要从 store 取)
    else:
        # 走 store:NeatMem 自己取,用 k 控制 limit
        if message_store is not None:
            last_k_messages = message_store.get_last_messages(search_filters, limit=k)
            message_store.save_messages(messages, search_filters)

    # Step 1: 搜索已有记忆
    t0 = time.monotonic()
    messages_text = json.dumps(messages, ensure_ascii=False)
    search_result = memory.search(
        query=messages_text,
        top_k=10,
        filters=search_filters,
        rerank=False,
    )
    existing_memories = search_result.get("results", [])
    step1_ms = (time.monotonic() - t0) * 1000
    logger.info(f"{prefix}[Step 1] 搜索已有记忆 | 找到 {len(existing_memories)} 条, 耗时 {step1_ms:.0f}ms")

    # Step 2: LLM 提取记忆
    t0 = time.monotonic()
    extracted = extract_memories(
        openai_client=openai_client,
        llm_model=llm_model,
        messages=messages,
        existing_memories=existing_memories,
        search_filters=search_filters,
        custom_instructions=custom_instructions,
        req_id=req_id,
        metadata=metadata,
        last_k_messages=last_k_messages,
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
        openai_client=openai_client,
        llm_model=llm_model,
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
        _text_to_id_map = {}  # text → memory_id 映射，用于 Step 4.5 回填 related
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
            if app_id:
                add_params["app_id"] = app_id

            add_result = memory.add(**add_params)
            added_memories.extend(add_result.get("results", []))

            # 写入后补建实体索引 + 记录 text→id 映射
            for added in add_result.get("results", []):
                mid = added.get("id") or added.get("memory_id")
                if mid:
                    _text_to_id_map[mem["text"]] = mid
                    if entity_extractor and entity_store:
                        try:
                            scope_parts = []
                            if app_id:
                                scope_parts.append(f"app_id={app_id}")
                            if user_id:
                                scope_parts.append(f"user_id={user_id}")
                            if agent_id:
                                scope_parts.append(f"agent_id={agent_id}")
                            if run_id:
                                scope_parts.append(f"run_id={run_id}")
                            scope = "&".join(scope_parts)
                            entities = entity_extractor.extract(mem["text"])
                            entity_store.link_entities(
                                entities,
                                memory_id=mid,
                                scope=scope,
                                embed_fn=memory.embedding_model.embed,
                            )
                        except Exception as e:
                            logger.warning(f"{prefix}[Step 4] Entity link failed for {mid}: {e}")
                    # BM25 sparse vector 写入（add 路径）
                    if bm25_index is not None:
                        try:
                            bm25_index.index_memory(mid, mem["text"], search_filters)
                        except Exception as e:
                            logger.warning(f"{prefix}[Step 4] BM25 index failed for {mid}: {e}")

        step4_ms = (time.monotonic() - t0) * 1000
        logger.info(f"{prefix}[Step 4] 写入完成 | 实际写入 {len(added_memories)} 条, 耗时 {step4_ms:.0f}ms")

        # Step 4b: 对更新/合并的旧记忆覆盖 BM25 sparse vector
        updated_ids = set()
        for dup in dedup_result.duplicates:
            mid = dup.get("old_id")
            text = dup.get("new_text", "")
            if mid and mid not in updated_ids and bm25_index is not None:
                try:
                    bm25_index.index_memory(mid, text, search_filters)
                except Exception as e:
                    logger.warning(f"[Step 4b] BM25 index failed for {mid}: {e}")
                updated_ids.add(mid)
        for m in dedup_result.merged:
            mid = m.get("old_id")
            text = m.get("merged_text") or m.get("new_text", "")
            if mid and mid not in updated_ids and bm25_index is not None:
                try:
                    bm25_index.index_memory(mid, text, search_filters)
                except Exception as e:
                    logger.warning(f"[Step 4b] BM25 index failed for {mid}: {e}")
                updated_ids.add(mid)

        # Step 4.5: 回填 related metadata（双向关联）
        if dedup_result.link_pairs:
            t0_link = time.monotonic()
            link_count = 0
            for link in dedup_result.link_pairs:
                new_id = _text_to_id_map.get(link["new_text"])
                old_id = link["old_id"]
                rel_type = link["relation_type"]
                if new_id:
                    _add_related(memory, new_id, old_id, rel_type)
                    _add_related(memory, old_id, new_id, rel_type)
                    link_count += 1
            link_ms = (time.monotonic() - t0_link) * 1000
            logger.info(
                f"{prefix}[Step 4.5] 回填 related 完成 | {link_count} 对关联, "
                f"same_fact={sum(1 for l in dedup_result.link_pairs if l['relation_type']=='same_fact')}, "
                f"same_topic={sum(1 for l in dedup_result.link_pairs if l['relation_type']=='same_topic')}, "
                f"耗时 {link_ms:.0f}ms"
            )
    else:
        added_memories = []
        logger.info(f"{prefix}[Step 4] 无新记忆需要写入")

    return {
        "results": added_memories,
        "duplicates": dedup_result.duplicates,
        "merged": dedup_result.merged,
        "link_pairs": dedup_result.link_pairs,
    }
