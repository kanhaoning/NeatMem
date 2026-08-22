"""
自研 add 接口核心模块：LLM 提取 + 语义去重 + 写入
存储层复用 mem0 的 add(infer=False) / search / update 公开 API
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from neatmem.utils.llm_client import complete_chat
from neatmem.signals.entity.base import AbstractEntityExtractor
from neatmem.storage.entity.base import AbstractEntityStore
from neatmem.batching import VECTOR_STORE_TRACK
from neatmem.prompts.extraction import (
    ADDITIVE_EXTRACTION_PROMPT,
    generate_additive_extraction_prompt,
)
from neatmem.prompts.loader import load_prompt
from neatmem.memory_search import search_memories
from neatmem.config import (
    DEDUP_ENABLED,
    DEDUP_RESOLVER,
    DEDUP_DETECTOR,
    DEDUP_RECALL_THRESHOLD,
    DEDUP_THINKING,
    EDIT_THINKING,
    ENABLE_GRAPH,
    LLM_PROVIDER,
)
from neatmem.utils.text_parsing import extract_json, remove_code_blocks

logger = logging.getLogger(__name__)


def _convert_search_results(results):
    """将 search_memories() 的返回格式转换为 memory.search() 的格式。

    search_memories: {"id", "score", "payload"}
    memory.search:   {"id", "memory", "score", "metadata"}
    """
    return [
        {
            "id": r.get("id"),
            "memory": r.get("payload", {}).get("data", ""),
            "score": r.get("score"),
            "metadata": r.get("payload", {}),
        }
        for r in results
    ]


def _enrich_payloads(results, memory):
    """search_memories() 可能不返回 payload，用 qdrant client 补查。"""
    if not results:
        return results
    ids_to_fetch = [r.get("id") for r in results if not r.get("payload")]
    if ids_to_fetch:
        client = memory.vector_store.client
        collection = memory.collection_name
        points = client.retrieve(
            collection_name=collection,
            ids=ids_to_fetch,
            with_payload=True,
            with_vectors=False,
        )
        payload_map = {str(p.id): p.payload for p in points}
        for r in results:
            if not r.get("payload") and r.get("id") in payload_map:
                r["payload"] = payload_map[r["id"]]
    return results


# _BATCH_CONTEXT_RULE removed 2026-08-01 (redundant with extraction.py
# Same-Batch Context Resolution + Language Requirement; see docs/internal-notes/
# 20260801-batch-context-rule-redundancy-and-language-anchor-analysis.md)

# Shadow mode：只记录分类结果到日志，不执行任何操作（所有记忆都 add）
DEDUP_DRY_RUN = os.environ.get("DEDUP_DRY_RUN", "false").lower() == "true"

# 操作导向 listwise prompt（v7-think-off，Step 2 测试选定，准确率 70.6%）
# 与 4-class prompt 的核心区别：
#   1. 操作导向（add/none/update）而非关系导向（redundant/merge/link/independent）
#   2. listwise 1 次 LLM 调用（1 个 action + 1 个 targetId），非 k 次 pairwise
#   3. 信息点检查：强制 LLM 逐个检查候选信息点是否被新事实明确提到
#   4. thinking OFF（Step 2 测试：thinking ON 准确率 -11.8pp，token 8.8x）
ACTION_DEDUP_PROMPT = """你是记忆管理系统。给定一条新事实和已有候选记忆，决定最佳操作。

新事实："{new_text}"

已有候选记忆：
{candidate_block}

操作定义：
- "add"：新事实和候选是不同事件或不同事实，各自独立存储
- "none"：新事实和候选是同一事实的重新表述，信息内容实质相同，跳过不写
- "update"：新事实是候选的更新版本，且新事实明确提到了候选记忆中的所有具体信息点，用新覆盖旧

判断步骤：
1. 列出候选记忆中的所有具体信息点（日期、地点、人物、动作、情感等）
2. 逐一检查新事实是否**明确提到**了每个信息点（"大致相关"不算"提到"）
3. 如果候选有任何新事实未明确提到的信息点 -> add（两条都保留，不丢信息）
4. 如果新事实提到了所有信息点 + 有新信息 -> update
5. 如果新事实提到了所有信息点 + 无新信息 -> none

注意："同话题"不等于"同事实"。两条记忆可以关于同一话题但是不同事件，此时应 add。

返回 JSON 对象：
{{"action": "add|none|update", "targetId": "候选编号-if-update", "reason": "简要说明"}}

只返回 JSON。"""


def _parse_action_response(response: str, candidates_count: int) -> Dict[str, Any]:
    """解析操作导向 listwise 返回的 JSON，返回 {action, target_idx, reason}"""
    try:
        parsed = json.loads(response, strict=False)
    except json.JSONDecodeError:
        try:
            start = response.find("{")
            end = response.rfind("}")
            if start != -1 and end != -1:
                parsed = json.loads(response[start:end + 1], strict=False)
            else:
                return {"action": "add", "target_idx": -1, "reason": "parse_error"}
        except json.JSONDecodeError:
            return {"action": "add", "target_idx": -1, "reason": "parse_error"}

    action = str(parsed.get("action", "add")).strip().lower()
    if action not in ("add", "none", "update"):
        action = "add"

    target_id = parsed.get("targetId", "")
    target_idx = -1
    if action == "update":
        # targetId 可能是数字、"[1]"、"1" 等格式
        try:
            target_idx = int(str(target_id).strip("[] ")) - 1  # 转为 0-based
        except (ValueError, TypeError):
            target_idx = 0  # 默认选第一个候选

    reason = str(parsed.get("reason", ""))
    return {"action": action, "target_idx": target_idx, "reason": reason}


def _build_candidate_block(candidates: List[Dict[str, Any]]) -> str:
    """构建候选记忆块文本，含 id、content、created_at"""
    lines = []
    for i, c in enumerate(candidates):
        created = c.get("metadata", {}).get("created_at", "")
        if not created:
            created = c.get("created_at", "unknown")
        lines.append(f"[{i + 1}] ID: {i + 1}, Content: {c.get('memory', '')}, Created: {created}")
    return "\n".join(lines)


def dedup_memories_action(
    memory,
    openai_client,
    llm_model: str,
    extracted_memories: List[Dict[str, Any]],
    search_filters: Dict[str, Any],
    req_id: str = "",
    bm25_index=None,
    entity_extractor=None,
    entity_store=None,
    use_bm25: bool = False,
    use_entity: bool = False,
) -> DedupResult:
    """操作导向 listwise 去重（DEDUP_DETECTOR=listwise）

    每条新记忆：
    1. 搜索 top-5 候选（统一走 search_memories，use_bm25/use_entity 控制信号开关）
    2. 1 次 LLM 调用，返回 action（add/none/update）+ targetId
    3. 执行操作：
       - add -> 写入新记忆
       - none -> 跳过（不写）
       - update -> 按 DEDUP_RESOLVER 处理：skip 时降级 add，
         replace/rewrite/edit 时更新旧记忆
    4. DEDUP_DRY_RUN=true 时只记录日志，所有记忆都 add
    """
    result = DedupResult()
    total = len(extracted_memories)
    prefix = f"[{req_id} 去重]" if req_id else "[去重]"
    strategy = "skip" if DEDUP_RESOLVER == "skip" else "update"

    for idx, new_mem in enumerate(extracted_memories, 1):
        new_text = new_mem.get("text", "")
        if not new_text:
            continue

        new_attr = new_mem.get("attributed_to", "user")
        tag = f"{prefix} #{idx}/{total}"

        # --- 搜索候选（每次都搜最新状态） ---
        # 统一走 search_memories()：use_bm25/use_entity 全关时退化为纯 dense
        # over-fetch + threshold 过滤，与此前 mem0 memory.search(rerank=False)
        # （BM25/entity 已被 monkey-patch 关闭）语义等价。
        dedup_filters = {**search_filters, "attr_source": new_attr}
        t0 = time.monotonic()
        _sr = search_memories(
            memory=memory,
            query=new_text,
            filters=dedup_filters,
            top_k=5,
            entity_extractor=entity_extractor if use_entity else None,
            entity_store=entity_store if use_entity else None,
            use_entity=use_entity,
            use_bm25=use_bm25,
            bm25_index=bm25_index,
        )
        _hits = _enrich_payloads(_sr.get("results", []), memory)
        hits = _convert_search_results(_hits)
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
            logger.info(f"{tag} -> 新增(无候选)")
            continue

        # --- 归因过滤 ---
        filtered_candidates = []
        for cand in candidates:
            cand_attr = cand.get("metadata", {}).get("attr_source")
            if cand_attr and cand_attr != new_attr:
                logger.info(f"{tag}   归因隔离: [{new_attr}] vs [{cand_attr}] -> skip")
                continue
            filtered_candidates.append(cand)

        if not filtered_candidates:
            result.to_add.append(new_mem)
            logger.info(f"{tag} -> 新增(候选全被归因隔离)")
            continue

        # --- 1 次 LLM 调用（listwise，thinking OFF） ---
        candidate_block = _build_candidate_block(filtered_candidates)
        prompt = load_prompt(
            "DEDUP_PROMPT", ACTION_DEDUP_PROMPT,
            ("new_text", "candidate_block"),
        ).format(
            new_text=new_text,
            candidate_block=candidate_block,
        )

        t0 = time.monotonic()
        try:
            resp = complete_chat(
                openai_client,
                llm_model,
                [{"role": "user", "content": prompt}],
                enable=DEDUP_THINKING,
                provider=LLM_PROVIDER,
                temperature=0.0,
                max_tokens=6000 if DEDUP_THINKING else 2000,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content or ""
            # 处理未闭合的 <think> 标签
            if "<think" in raw and "</think" not in raw:
                raw = re.sub(r"<think\b[^>]*>.*", "", raw, flags=re.DOTALL).strip()
            raw = re.sub(r"<think\b[^>]*>.*?</think\s*>", "", raw, flags=re.DOTALL).strip()
            parsed = _parse_action_response(raw, len(filtered_candidates))
        except Exception as e:
            logger.warning(f"{tag} LLM 调用失败: {e}")
            parsed = {"action": "add", "target_idx": -1, "reason": f"llm_error: {e}"}

        judge_ms = (time.monotonic() - t0) * 1000
        action = parsed["action"]
        target_idx = parsed["target_idx"]
        reason = parsed["reason"]

        logger.info(
            f"{tag}   判断({judge_ms:.0f}ms): action={action}, "
            f"target={target_idx + 1 if target_idx >= 0 else 'N/A'}, "
            f"reason={reason[:100]}"
        )

        # --- DEDUP_DRY_RUN：只记录，不执行 ---
        if DEDUP_DRY_RUN:
            result.to_add.append(new_mem)
            logger.info(f"{tag} -> [DRY_RUN] 记录 action={action}, 实际新增")
            continue

        # --- 执行操作 ---
        if action == "add":
            result.to_add.append(new_mem)
            logger.info(f"{tag} -> 新增(action=add)")

        elif action == "none":
            # Duplicate memory: skip writing
            cand = filtered_candidates[0] if filtered_candidates else None
            logger.info(f"{tag} -> 跳过(action=none): 重复记忆不写入")
            result.duplicates.append({
                "new_text": new_text,
                "old_id": cand["id"] if cand else None,
                "old_text": cand.get("memory", "") if cand else "",
                "score": cand.get("score") if cand else 0,
                "relation": "none_skip",
            })

        elif action == "update":
            if strategy == "skip":
                # skip 模式不支持 update，降级为 add（新旧共存）
                result.to_add.append(new_mem)
                logger.info(f"{tag} -> 新增(action=update降级为add, skip模式不支持替换)")

            elif strategy == "update":
                # update 模式：根据 DEDUP_RESOLVER 决定如何处理
                if target_idx < 0 or target_idx >= len(filtered_candidates):
                    target_idx = 0  # 默认选第一个
                cand = filtered_candidates[target_idx]
                old_text = cand.get("memory", "")
                cand_attr = cand.get("metadata", {}).get("attr_source", "user")
                pd_meta = {}

                if DEDUP_RESOLVER == "edit":
                    merged, pd_meta = patch_merge_memories(openai_client, llm_model, old_text, new_text)
                    write_text = merged if merged else new_text  # fallback: 新覆盖旧
                    logger.info(f"{tag} -> edit(patch_diff): {pd_meta.get('patch_status')}, merged={len(write_text)} chars")
                elif DEDUP_RESOLVER == "rewrite":
                    merged, resolver_status = _memos_resolve(
                        openai_client, llm_model, "contradictory",
                        new_text, old_text,
                        cand.get("metadata", {}).get("created_at"),
                        tag=tag,
                    )
                    # Unresolvable/error all fall back to new covering old
                    # (= legacy cannot_merge semantics).
                    write_text = merged if merged else new_text
                    logger.info(f"{tag} -> rewrite(memos): {resolver_status}, merged={len(write_text)} chars")
                else:  # replace
                    write_text = new_text
                    logger.info(f"{tag} -> 更新替换(action=update): '{old_text[:80]}' -> '{new_text[:80]}'")

                memory.update(memory_id=cand["id"], data=write_text, metadata={"attr_source": cand_attr})
                result.duplicates.append({
                    "new_text": new_text,
                    "old_id": cand["id"],
                    "old_text": old_text,
                    "write_text": write_text,
                    "score": cand.get("score"),
                    "relation": f"update_{DEDUP_RESOLVER}",
                    "patch_status": pd_meta.get("patch_status") if DEDUP_RESOLVER == "edit" else None,
                })

    logger.info(
        f"{prefix} 完成 | 新增 {len(result.to_add)} 条, "
        f"跳过/替换 {len(result.duplicates)} 条"
    )
    return result


# ============================================================================
# Pointwise dedup (DEDUP_DETECTOR=pointwise)
# MemOS 3-class detector port; plan:
# docs/internal-notes/20260812-dedup-mode-to-enabled-resolver-detector.md
# ============================================================================

# MemOS MEMORY_RELATION_DETECTOR_PROMPT, verbatim
# (memos/templates/tree_reorganize_prompts.py:197)
PAIRWISE_DETECTOR_PROMPT = """You are a memory relationship analyzer.
You are given two plaintext statements. Determine the relationship between them. Classify the relationship into one of the following categories:

contradictory: The two statements describe the same event or related aspects of it but contain factually conflicting details.
redundant: The two statements describe essentially the same event or information with significant overlap in content and details, conveying the same core information (even if worded differently).
independent: The two statements are either about different events/topics (unrelated) OR describe different, non-overlapping aspects or perspectives of the same event without conflict (complementary). In both sub-cases, they provide distinct information without contradiction.
Respond only with one of the three labels: contradictory, redundant, or independent.
Do not provide any explanation or additional text.

Statement 1: {statement_1}
Statement 2: {statement_2}
"""

_PAIRWISE_RELATIONS = ("contradictory", "redundant", "independent")


def _detect_relation(openai_client, llm_model: str, new_text: str, cand_text: str, tag: str = "") -> str:
    """One DETECTOR call for a (new, candidate) pair.

    Fail-open: LLM error or off-whitelist output -> "independent"
    (consistent with the existing parse_error -> add philosophy).
    """
    prompt = PAIRWISE_DETECTOR_PROMPT.format(statement_1=new_text, statement_2=cand_text)
    try:
        resp = complete_chat(
            openai_client,
            llm_model,
            [{"role": "user", "content": prompt}],
            enable=False,
            provider=LLM_PROVIDER,
            temperature=0.0,
            max_tokens=200,
        )
        raw = strip_thinking(resp.choices[0].message.content or "").strip().lower()
    except Exception as e:
        logger.warning(f"{tag} DETECTOR 调用失败: {e} -> independent(fail-open)")
        return "independent"
    if raw in _PAIRWISE_RELATIONS:
        return raw
    logger.warning(f"{tag} DETECTOR 非法输出 {raw[:80]!r} -> independent(fail-open)")
    return "independent"


# Rewrite resolver prompt: MemOS MEMORY_RELATION_RESOLVER_PROMPT, adapted (variant B).
# Adaptations vs MemOS original (tree_reorganize_prompts.py:211):
# - verb fidelity rule appended (ported from MERGE_PROMPT 2026-05-30 fix)
# - metadata feeds created_at only (prompt tolerates "if available")
# Single source of truth: neatmem/prompts/examples/rewrite_en.txt (loaded via
# load_prompt with default_file; REWRITE_PROMPT env var overrides).
def _memos_resolve(openai_client, llm_model: str, relation: str,
                   new_text: str, old_text: str, old_created_at, tag: str = "") -> tuple[Optional[str], str]:
    """Variant B: MemOS RESOLVER adapted. Returns (merged_text, status).

    status:
    - "resolved"              -> caller updates old with merged_text
    - "unresolvable_keep_new" -> caller deletes old, adds new
    - "unresolvable_keep_old" -> caller skips new
    - "error"                 -> caller fallback: new covers old (aligns rewrite fallback)
    """
    metadata_1 = json.dumps({"created_at": datetime.now(timezone.utc).isoformat()}, ensure_ascii=False)
    metadata_2 = json.dumps({"created_at": str(old_created_at or "unknown")}, ensure_ascii=False)
    prompt = load_prompt(
        "REWRITE_PROMPT",
        default_file="rewrite_en.txt",
        required_placeholders=("statement_1", "statement_2"),
        supported_placeholders=("statement_1", "statement_2", "relation", "metadata_1", "metadata_2"),
    ).format(
        relation=relation,
        statement_1=new_text,
        metadata_1=metadata_1,
        statement_2=old_text,
        metadata_2=metadata_2,
    )
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
        raw = strip_thinking(resp.choices[0].message.content or "")
    except Exception as e:
        logger.warning(f"{tag} RESOLVER 调用失败: {e}")
        return None, "error"
    m = re.search(r"<answer>(.*?)</answer>", raw, re.DOTALL)
    if not m:
        logger.warning(f"{tag} RESOLVER 缺少 <answer> 标签: {raw[:100]!r}")
        return None, "error"
    answer = m.group(1).strip()
    if answer == "No":
        # Unresolvable: keep the newer by created_at (new memory created at
        # write time = now, so it wins unless old carries a future timestamp)
        try:
            old_time = datetime.fromisoformat(str(old_created_at).replace("Z", "+00:00"))
        except (ValueError, TypeError):
            old_time = None
        if old_time is not None and old_time > datetime.now(timezone.utc):
            return None, "unresolvable_keep_old"
        return None, "unresolvable_keep_new"
    if len(answer) < 5:
        logger.warning(f"{tag} RESOLVER 输出过短: {answer!r}")
        return None, "error"
    return answer, "resolved"


def patch_merge_memories_pointwise(openai_client, llm_model: str, old_text: str, new_text: str,
                                   relation: str) -> tuple[Optional[str], Dict[str, Any]]:
    """Pointwise edit path: same edit prompt as listwise, DETECTOR label
    supplied as ``relation`` (the default V3/F2 prompts ignore it; legacy
    pointwise prompts that consume {relation} work via EDIT_PROMPT override).

    Same machinery as patch_merge_memories, but the prompt carries no
    relationship field, so apply_patch skips the relationship early-exit.
    """
    prompt = load_prompt(
        "EDIT_PROMPT",
        default_file="edit_en.txt",
        required_placeholders=("old_text", "new_text"),
        supported_placeholders=("old_text", "new_text", "relation"),
    ).format(old_text=old_text, new_text=new_text, relation=relation)
    metadata: Dict[str, Any] = {}
    try:
        resp = complete_chat(
            openai_client,
            llm_model,
            [{"role": "user", "content": prompt}],
            enable=EDIT_THINKING,
            provider=LLM_PROVIDER,
            response_format={"type": "json_object"},
        )
        response = strip_thinking(resp.choices[0].message.content or "")
        response = remove_code_blocks(response)
        metadata["patch_raw"] = response

        merged, status = apply_patch(old_text, response)
        metadata["patch_status"] = status

        if status == "success":
            return merged, metadata
        logger.info(f"[patch_diff_pointwise] fallback: {status}")
        return None, metadata

    except Exception as e:
        logger.warning(f"[patch_diff_pointwise] LLM 调用失败: {e}")
        metadata["patch_status"] = "llm_error"
        metadata["patch_raw"] = str(e)
        return None, metadata


def dedup_memories_pairwise(
    memory,
    openai_client,
    llm_model: str,
    extracted_memories: List[Dict[str, Any]],
    search_filters: Dict[str, Any],
    req_id: str = "",
    bm25_index=None,
    entity_extractor=None,
    entity_store=None,
    use_bm25: bool = False,
    use_entity: bool = False,
) -> DedupResult:
    """Pointwise 去重（DEDUP_DETECTOR=pointwise, MemOS 三分类移植）

    每条新记忆：
    1. 搜索 top-5 候选（召回块与 dedup_memories_action 逐字一致，控制变量）
    2. 每条候选 1 次 DETECTOR 调用（contradictory/redundant/independent）
    3. 多目标（MemOS 式）：遍历全部候选，返回所有命中，不 break
    4. 命中处理按 DEDUP_RESOLVER：
       - skip:     two-phase - 全 redundant -> 不写; 有 contradictory -> 降级 add
       - replace:  单目标（多目标会创建 N 的多个副本，无意义）
       - rewrite:  多目标循环 - MemOS RESOLVER 适配版（_memos_resolve）;
                   状态机保两条 MemOS 不变量（2026-08-13 修复，
                   替代 already_resolved 守卫）:
                   (1) N 最多存一份（末尾单点入队）;
                   (2) N 被 merge 消费后不单独入库;
                   merge 路径允许分裂繁殖（M1'/M2'/...）
       - edit:     多目标循环 - patch_merge_memories_pointwise
                   （不写 N 不删旧, 天然满足两条不变量）
    """
    result = DedupResult()
    total = len(extracted_memories)
    prefix = f"[{req_id} 去重pw]" if req_id else "[去重pw]"

    for idx, new_mem in enumerate(extracted_memories, 1):
        new_text = new_mem.get("text", "")
        if not new_text:
            continue

        new_attr = new_mem.get("attributed_to", "user")
        tag = f"{prefix} #{idx}/{total}"

        # --- 搜索候选（与 dedup_memories_action 逐字一致） ---
        dedup_filters = {**search_filters, "attr_source": new_attr}
        t0 = time.monotonic()
        _sr = search_memories(
            memory=memory,
            query=new_text,
            filters=dedup_filters,
            top_k=5,
            entity_extractor=entity_extractor if use_entity else None,
            entity_store=entity_store if use_entity else None,
            use_entity=use_entity,
            use_bm25=use_bm25,
            bm25_index=bm25_index,
        )
        _hits = _enrich_payloads(_sr.get("results", []), memory)
        hits = _convert_search_results(_hits)
        search_ms = (time.monotonic() - t0) * 1000

        candidates = [h for h in hits if h.get("score", 0) >= DEDUP_RECALL_THRESHOLD]
        logger.info(
            f"{tag} 搜索 | 召回 {len(hits)} 条(阈值以上 {len(candidates)} 条), "
            f"耗时 {search_ms:.0f}ms | 新记忆: '{new_text[:120]}'"
        )

        if not candidates:
            result.to_add.append(new_mem)
            logger.info(f"{tag} -> 新增(无候选)")
            continue

        # --- 归因过滤（与 dedup_memories_action 逐字一致） ---
        filtered_candidates = []
        for cand in candidates:
            cand_attr = cand.get("metadata", {}).get("attr_source")
            if cand_attr and cand_attr != new_attr:
                logger.info(f"{tag}   归因隔离: [{new_attr}] vs [{cand_attr}] -> skip")
                continue
            filtered_candidates.append(cand)

        if not filtered_candidates:
            result.to_add.append(new_mem)
            logger.info(f"{tag} -> 新增(候选全被归因隔离)")
            continue

        # --- 逐候选 DETECTOR 调用（多目标：遍历全部候选，不 break） ---
        # MemOS-style: detect all candidates, return all hits.
        # DRY_RUN also traverses all (unchanged from single-target dry-run).
        labeled = []  # [(cand, relation, detect_ms)]
        for cand in filtered_candidates:
            t0 = time.monotonic()
            relation = _detect_relation(
                openai_client, llm_model, new_text, cand.get("memory", ""), tag=tag
            )
            detect_ms = (time.monotonic() - t0) * 1000
            labeled.append((cand, relation, detect_ms))
            logger.info(
                f"{tag}   判断({detect_ms:.0f}ms): {relation} | '{cand.get('memory', '')[:100]}'"
            )

        hits_labeled = [(c, r) for c, r, _ in labeled if r != "independent"]

        # --- DEDUP_DRY_RUN：只记录，不执行 ---
        if DEDUP_DRY_RUN:
            result.to_add.append(new_mem)
            labels = [r for _, r in hits_labeled] or ["all_independent"]
            logger.info(f"{tag} -> [DRY_RUN] 命中标签={labels}, 实际新增")
            continue

        if not hits_labeled:
            result.to_add.append(new_mem)
            logger.info(f"{tag} -> 新增(全部 independent)")
            continue

        # --- 多目标：遍历全部命中（MemOS 式） ---
        # skip: two-phase — N is write-or-not as a whole, can't per-candidate.
        #   Phase 1: collect all hit labels (already in hits_labeled).
        #   Phase 2: unified decision — all redundant -> N not written;
        #            any contradictory -> N degrade to add.
        if DEDUP_RESOLVER == "skip":
            relations = [r for _, r in hits_labeled]
            if all(r == "redundant" for r in relations):
                for cand, relation in hits_labeled:
                    old_text = cand.get("memory", "")
                    result.duplicates.append({
                        "new_text": new_text,
                        "old_id": cand["id"],
                        "old_text": old_text,
                        "score": cand.get("score"),
                        "relation": "pointwise_redundant_none",
                    })
                logger.info(f"{tag} -> 跳过(全部 redundant 不写, {len(hits_labeled)} 命中)")
            else:
                result.to_add.append(new_mem)
                n_contra = sum(1 for r in relations if r == "contradictory")
                logger.info(
                    f"{tag} -> 新增({n_contra} contradictory 降级为add, "
                    f"skip模式不合并, {len(hits_labeled)} 命中)"
                )
            continue

        # replace: single-target only — multi-target would overwrite multiple
        # old memories with identical N, creating duplicates. Not a MemOS pattern.
        if DEDUP_RESOLVER == "replace":
            cand, relation = hits_labeled[0]
            if len(hits_labeled) > 1:
                logger.info(
                    f"{tag}   replace 多目标无意义, 只处理第一个, "
                    f"其余 {len(hits_labeled) - 1} 个跳过"
                )
            old_text = cand.get("memory", "")
            cand_attr = cand.get("metadata", {}).get("attr_source", "user")
            write_text = new_text
            logger.info(f"{tag} -> 新盖旧({relation}): '{old_text[:80]}' -> '{new_text[:80]}'")
            memory.update(memory_id=cand["id"], data=write_text, metadata={"attr_source": cand_attr})
            result.duplicates.append({
                "new_text": new_text,
                "old_id": cand["id"],
                "old_text": old_text,
                "write_text": write_text,
                "score": cand.get("score"),
                "relation": f"pointwise_{relation}_{DEDUP_RESOLVER}",
            })
            continue

        # rewrite / edit: multi-target loop.
        # rewrite uses a two-flag state machine (2026-08-13 fix, replaces the
        # already_resolved guard) emulating MemOS's two invariants:
        #   (1) N stored at most once  -> single to_add decision after the loop
        #   (2) once N is consumed by a merge, it is never stored standalone
        # The loop never skips hits: keep_new still deletes old memories and
        # merges still update them. Split breeding (M1'/M2'/...) is allowed,
        # that's the MemOS behavior this experiment observes.
        # edit never queues N and never deletes old memories, so it satisfies
        # both invariants trivially; its branch is unchanged.
        n_consumed = False
        keep_new_seen = False
        for cand, relation in hits_labeled:
            old_text = cand.get("memory", "")
            cand_attr = cand.get("metadata", {}).get("attr_source", "user")
            pd_meta: Dict[str, Any] = {}
            resolver_status = ""

            if DEDUP_RESOLVER == "rewrite":
                merged, resolver_status = _memos_resolve(
                    openai_client, llm_model, relation,
                    new_text, old_text,
                    cand.get("metadata", {}).get("created_at"),
                    tag=tag,
                )
                if resolver_status == "unresolvable_keep_new":
                    memory.delete(memory_id=cand["id"])
                    keep_new_seen = True
                    result.duplicates.append({
                        "new_text": new_text,
                        "old_id": cand["id"],
                        "old_text": old_text,
                        "score": cand.get("score"),
                        "relation": f"pointwise_{relation}_memos_keep_new",
                    })
                    logger.info(f"{tag} -> 不可调和删旧({relation}), N 去向循环末决策")
                    continue
                if resolver_status == "unresolvable_keep_old":
                    result.duplicates.append({
                        "new_text": new_text,
                        "old_id": cand["id"],
                        "old_text": old_text,
                        "score": cand.get("score"),
                        "relation": f"pointwise_{relation}_memos_keep_old",
                    })
                    logger.info(f"{tag} -> 不可调和留旧删新({relation})")
                    continue
                write_text = merged if merged else new_text  # error fallback
                n_consumed = True
                logger.info(
                    f"{tag} -> memos_resolve({relation}): {resolver_status}, "
                    f"merged={len(write_text)} chars, n_consumed=True"
                )

            else:  # edit
                merged, pd_meta = patch_merge_memories_pointwise(
                    openai_client, llm_model, old_text, new_text, relation
                )
                write_text = merged if merged else new_text  # fallback
                logger.info(
                    f"{tag} -> patch_diff_pointwise({relation}): "
                    f"{pd_meta.get('patch_status')}, merged={len(write_text)} chars"
                )

            memory.update(memory_id=cand["id"], data=write_text, metadata={"attr_source": cand_attr})
            result.duplicates.append({
                "new_text": new_text,
                "old_id": cand["id"],
                "old_text": old_text,
                "write_text": write_text,
                "score": cand.get("score"),
                "relation": f"pointwise_{relation}_{DEDUP_RESOLVER}",
                "patch_status": pd_meta.get("patch_status"),
                "resolver_status": resolver_status or None,
            })

        # rewrite-only: single-point decision on N's fate after the loop.
        # n_consumed takes precedence (N lives on inside a merged old memory);
        # otherwise queue N iff a keep_new occurred and N was never consumed.
        if DEDUP_RESOLVER == "rewrite":
            if n_consumed:
                logger.info(f"{tag} -> N 已被 merge 消费, 不单独入库")
            elif keep_new_seen:
                result.to_add.append(new_mem)
                logger.info(f"{tag} -> N 单点入队(keep_new)")

    logger.info(
        f"{prefix} 完成 | 新增 {len(result.to_add)} 条, "
        f"跳过/替换 {len(result.duplicates)} 条"
    )
    return result


# 记忆合并 prompt
# 修改记录:
#   - 2026-05-30 [P0-b]: 在规则末尾追加"动词保真"规则。
#     问题：merge LLM 合并时简化动词（"realized" → "found"，"decided" → "explored"），
#     导致语义改变。
#     详见 docs/internal-notes/20260530-extraction-merge-prompt-fix-plan.md
#   - 2026-08-23: MERGE_PROMPT 退役。rewrite resolver 改走 MemOS 改编版
#     (neatmem/prompts/examples/rewrite_en.txt, 含本条动词保真规则的英文移植),
#     中文原文归档 tmp/20260822-prompt-whitebox-archive/。
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


# ---- 正向 patch_diff 调优 prompts（Phase 1 选出最优）----
# 2026-08-23: F2 (=Phase 1 最优, PATCH_DIFF_PROMPT_FORWARD_BEST) 迁移为
# neatmem/prompts/examples/edit_en.txt —— txt 是唯一出处, 代码不再持副本。
# F3 (forward append-only): no replace option
PATCH_DIFF_PROMPT_FORWARD_F3 = """You are a memory patch generator.

OLD MEMORY:
{old_text}

NEW INFORMATION:
{new_text}

Task: Append any details from NEW INFORMATION that are missing from OLD MEMORY.
DO NOT rewrite the old memory. Only add what's missing.

Rules:
1. If new_info adds information not in old -> "append" with "after" referencing OLD
2. If new_info contradicts old but both may be true -> "conflict", no changes
3. If completely different topic/entity/event -> "unrelated", no changes
4. NEVER use "replace". Do not modify existing text in OLD MEMORY.
5. PRESERVE ALL DETAILS FROM NEW INFORMATION. If NEW INFORMATION contains any specific details not in OLD MEMORY, you MUST use "append" to include them.
6. Your "after" must be copied from OLD.

Output JSON only:
{{"relationship": "append|conflict|unrelated", "changes": [{{"type": "append", "after": "...", "text": "..."}}]}}"""

# Phase 1 选出最优正向 prompt = F2 (append bias, thinking OFF)
# F2-OFF: 20/20 success, 100% old retention, 95.9% new retention, 4 bad replaces, 107 tok, 1.95s
# (F2 本体见 neatmem/prompts/examples/edit_en.txt)

# ---- 反向 patch_diff prompts（new 为基底，old 为补充）----
# Phase 1 选出最优后，将 PATCH_DIFF_PROMPT_REVERSED_BEST 设为对应的 prompt

# R1 (baseline reversed): F1 prompt, slots swapped ({new_text}=base, {old_text}=supplement)
PATCH_DIFF_PROMPT_REVERSED_R1 = """You are a memory patch generator.

OLD MEMORY:
{new_text}

NEW INFORMATION:
{old_text}

Task: Generate a minimal patch to update the old memory with the new information.
DO NOT rewrite the entire memory. Only specify what needs to change.

Rules:
1. If new_info corrects old -> "replace" with exact quote from OLD
2. If new_info adds detail -> "append" with "after" referencing OLD
3. If new_info contradicts old but both may be true -> "conflict", no changes
4. If completely different topic/entity/event -> "unrelated", no changes. Sharing the same core entity and scene but describing a different facet is NOT unrelated - use "append".
5. NEVER rewrite unchanged text. Your "quote" and "after" must be copied from OLD.
6. PRESERVE ALL DETAILS FROM NEW INFORMATION. If NEW INFORMATION contains any specific details not in OLD MEMORY (verbs, proper nouns, emotions, time precision, activities), you MUST use "append" to include them. Do NOT use "replace" to simplify or summarize.
7. "replace" is ONLY for explicit corrections or outdated information. It must NOT remove unique details from either memory.
8. Prefer multiple small appends over one large replace.

Output JSON only:
{{"relationship": "update|append|conflict|unrelated", "changes": [{{"type": "replace", "quote": "...", "context": "...", "with": "..."}}, {{"type": "append", "after": "...", "text": "..."}}]}}"""

# R2 (reversed append-only): CURRENT/SUPPLEMENTARY labels, no replace
PATCH_DIFF_PROMPT_REVERSED_R2 = """You are a memory patch generator.

CURRENT MEMORY:
{new_text}

SUPPLEMENTARY DETAILS (older, may contain details missing from current):
{old_text}

Task: Append any details from SUPPLEMENTARY that are missing from CURRENT.
DO NOT rewrite the current memory. Only add what's missing.

Rules:
1. If SUPPLEMENTARY adds information not in CURRENT -> "append" with "after" referencing CURRENT
2. If SUPPLEMENTARY contradicts CURRENT but both may be true -> "conflict", no changes
3. If completely different topic/entity/event -> "unrelated", no changes
4. NEVER use "replace". SUPPLEMENTARY may be outdated; do not let it overwrite CURRENT.
5. PRESERVE ALL DETAILS FROM SUPPLEMENTARY that are not already in CURRENT.
6. Your "after" must be copied from CURRENT.

Output JSON only:
{{"relationship": "append|conflict|unrelated", "changes": [{{"type": "append", "after": "...", "text": "..."}}]}}"""

# R3 (reversed + replace with warning): CURRENT/SUPPLEMENTARY, replace allowed but warned
PATCH_DIFF_PROMPT_REVERSED_R3 = """You are a memory patch generator.

CURRENT MEMORY:
{new_text}

SUPPLEMENTARY DETAILS (older, may contain details missing from current):
{old_text}

Task: Append any details from SUPPLEMENTARY that are missing from CURRENT.
DO NOT rewrite the current memory. Only add what's missing.

Rules:
1. If SUPPLEMENTARY adds information not in CURRENT -> "append" with "after" referencing CURRENT
2. If SUPPLEMENTARY corrects a clear factual error in CURRENT (wrong date, wrong name, wrong number) -> "replace" with exact quote from CURRENT
3. If SUPPLEMENTARY contradicts CURRENT but both may be true -> "conflict", no changes
4. If completely different topic/entity/event -> "unrelated", no changes
5. WARNING: SUPPLEMENTARY may be outdated. Do NOT replace CURRENT with SUPPLEMENTARY unless CURRENT has a clear factual error. When uncertain, use "append".
6. PRESERVE ALL DETAILS FROM SUPPLEMENTARY that are not already in CURRENT.
7. Your "quote" and "after" must be copied from CURRENT.
8. Prefer multiple small appends over one large replace.

Output JSON only:
{{"relationship": "append|conflict|unrelated", "changes": [{{"type": "replace", "quote": "...", "with": "..."}}, {{"type": "append", "after": "...", "text": "..."}}]}}"""

# Phase 1 选出最优反向 prompt = R3 (replace with warning, thinking OFF)
# R3-OFF: 20/20 success, 98% old retention, 100% new retention, 1 bad replace, 72 tok, 1.74s
PATCH_DIFF_PROMPT_REVERSED_BEST = PATCH_DIFF_PROMPT_REVERSED_R3


def strip_thinking(text: str) -> str:
    """剥离 LLM 思考标签，兼容各厂商"""
    return re.sub(r"<think\b[^>]*>.*?</think\s*>", "", text, flags=re.DOTALL).strip()


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

    # F2（有 relationship）的早退逻辑：保持向后兼容
    # F2-no-rel prompt 不输出 relationship 字段，跳过此早退
    if "relationship" in patch:
        rel = patch.get("relationship", "unrelated")
        if rel in ("conflict", "unrelated"):
            return old_memory, f"fallback_{rel}"

    new_memory = old_memory
    changes = patch.get("changes", [])

    # F2-no-rel 的早退：空 changes
    if not changes:
        return old_memory, "fallback_no_changes"

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

    Prompt source: neatmem/prompts/examples/edit_en.txt (F2, Phase 1 best
    forward prompt), loaded via load_prompt("EDIT_PROMPT", default_file=...).

    Returns:
        (merged_text, metadata)
        merged_text: 成功时为合并后的文本，失败时为 None（由调用方决定 fallback）
        metadata: 含 patch_status, patch_raw 等
    """
    prompt = load_prompt(
        "EDIT_PROMPT",
        default_file="edit_en.txt",
        required_placeholders=("old_text", "new_text"),
        supported_placeholders=("old_text", "new_text"),
    ).format(old_text=old_text, new_text=new_text)
    metadata = {}
    try:
        resp = complete_chat(
            openai_client,
            llm_model,
            [{"role": "user", "content": prompt}],
            enable=EDIT_THINKING,
            provider=LLM_PROVIDER,
            response_format={"type": "json_object"},
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


def patch_merge_memories_reversed(openai_client, llm_model: str, old_text: str, new_text: str) -> tuple[Optional[str], Dict[str, Any]]:
    """反向 Patch Diff：以 new_text 为基底，将 old_text 的独有信息 append 到 new 上。

    用于 action_dedup update 分支的 patch_diff_reversed 模式（Arm H）。
    prompt 使用 PATCH_DIFF_PROMPT_REVERSED_BEST（Phase 1 选出的反向最优 prompt）。

    Returns:
        (merged_text, metadata) - 同 patch_merge_memories
    """
    prompt = PATCH_DIFF_PROMPT_REVERSED_BEST.format(new_text=new_text, old_text=old_text)
    metadata = {}
    try:
        resp = complete_chat(
            openai_client,
            llm_model,
            [{"role": "user", "content": prompt}],
            enable=EDIT_THINKING,
            provider=LLM_PROVIDER,
            response_format={"type": "json_object"},
        )
        response = strip_thinking(resp.choices[0].message.content or "")
        response = remove_code_blocks(response)
        metadata["patch_raw"] = response

        # 以 new_text 为基底 apply_patch
        merged, status = apply_patch(new_text, response)
        metadata["patch_status"] = status

        if status == "success":
            return merged, metadata
        else:
            logger.info(f"[patch_diff_reversed] fallback: {status}")
            return None, metadata

    except Exception as e:
        logger.warning(f"[patch_diff_reversed] LLM 调用失败: {e}")
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

    effective_instructions = custom_instructions or ""

    # 生成 prompt
    # metadata["timestamp"]: 事件发生时间（event time），区别于 NeatMem 自动盖的 created_at（处理时间）
    # 缺省时 mem0 的 _resolve_dates 兜底 observation_date = current_date（行为=修复前）
    observation_ts = (metadata or {}).get("timestamp")

    system_prompt = load_prompt("EXTRACTION_PROMPT", ADDITIVE_EXTRACTION_PROMPT)
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
    resp = complete_chat(
        openai_client,
        llm_model,
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        enable=True,
        provider=LLM_PROVIDER,
        response_format={"type": "json_object"},
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


def _advance_cursor_after_save(
    message_store: Optional[Any],
    user_id: Optional[str],
    agent_id: Optional[str],
    saved_message_max_seq: Optional[int],
) -> None:
    """Advance the vector-track extraction cursor after a successful inline extraction.

    This must live in add_memories, not in the HTTP layer: eval ingest calls
    add_memories in-process and never touches the endpoint, so an
    endpoint-level advance leaves the cursor at 0 and the batch scheduler
    re-extracts every already-extracted message (observed 2026-08-16:
    scheduler polluted the eval library during the search phase).
    Only the save-owning branch advances: saved_message_max_seq is None in
    external-injection mode (NeatMem does not own the message store there).
    run_id is '' because save_messages here uses search_filters (no run_id).
    """
    if saved_message_max_seq is None or message_store is None:
        return
    message_store.advance_cursor(
        user_id or "", agent_id or "", "", VECTOR_STORE_TRACK, saved_message_max_seq
    )


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
    use_bm25: bool = False,
    use_entity: bool = False,
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
    saved_message_max_seq = None  # max seq of the batch saved below (queue-mode cursor advance)
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
            saved = message_store.save_messages(messages, search_filters)
            if saved:
                saved_message_max_seq = max(m["seq"] for m in saved)

    # Step 1: 搜索已有记忆（统一走 search_memories()，等价性见 dedup 候选搜索处注释）
    t0 = time.monotonic()
    messages_text = json.dumps(messages, ensure_ascii=False)
    _sr = search_memories(
        memory=memory,
        query=messages_text,
        filters=search_filters,
        top_k=10,
        entity_extractor=entity_extractor if use_entity else None,
        entity_store=entity_store if use_entity else None,
        use_entity=use_entity,
        use_bm25=use_bm25,
        bm25_index=bm25_index,
    )
    _hits = _enrich_payloads(_sr.get("results", []), memory)
    existing_memories = _convert_search_results(_hits)
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
        _advance_cursor_after_save(message_store, user_id, agent_id, saved_message_max_seq)
        return {"results": [], "duplicates": [], "saved_message_max_seq": saved_message_max_seq}

    # Step 3: 语义去重
    t0 = time.monotonic()
    if not DEDUP_ENABLED:
        # DEDUP_ENABLED=false：直接写入，不调 dedup
        dedup_result = DedupResult()
        dedup_result.to_add = list(extracted)
        dedup_label = "dedup_off"
    else:
        # 两层分发：DEDUP_DETECTOR 选判定机制，DEDUP_RESOLVER 选判中后处理
        #   listwise  -> action dedup（skip/replace/rewrite/edit）
        #   pointwise -> MemOS 三分类逐对判定（rewrite 固定走 memos resolver）
        if DEDUP_DETECTOR == "pointwise":
            dedup_result = dedup_memories_pairwise(
                memory=memory,
                openai_client=openai_client,
                llm_model=llm_model,
                extracted_memories=extracted,
                search_filters=search_filters,
                req_id=req_id,
                bm25_index=bm25_index,
                entity_extractor=entity_extractor,
                entity_store=entity_store,
                use_bm25=use_bm25,
                use_entity=use_entity,
            )
            dedup_label = f"pointwise_{DEDUP_RESOLVER}" + (
                "_memos" if DEDUP_RESOLVER == "rewrite" else ""
            )
        else:
            dedup_result = dedup_memories_action(
                memory=memory,
                openai_client=openai_client,
                llm_model=llm_model,
                extracted_memories=extracted,
                search_filters=search_filters,
                req_id=req_id,
                bm25_index=bm25_index,
                entity_extractor=entity_extractor,
                entity_store=entity_store,
                use_bm25=use_bm25,
                use_entity=use_entity,
            )
            dedup_label = f"listwise_{DEDUP_RESOLVER}"
    step3_ms = (time.monotonic() - t0) * 1000
    logger.info(f"{prefix}[Step 3] 语义去重 {dedup_label}, 耗时 {step3_ms:.0f}ms")

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
                    # --- 图记忆 hook（mem0 1.0.11 忠实复现）---
                    # 关图时 ENABLE_GRAPH=false，此块不执行，代码路径与 baseline 一致。
                    # 开图时 lazy import kuzu/factory，调 graph_store.add()，失败降级 warning。
                    if ENABLE_GRAPH:
                        try:
                            from neatmem.signals.graph.factory import get_graph_store
                            gs = get_graph_store()
                            gs.add(mem["text"], search_filters)
                        except Exception as e:
                            logger.warning(f"{prefix}[Step 4] Graph add failed for {mid}: {e}")

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

    _advance_cursor_after_save(message_store, user_id, agent_id, saved_message_max_seq)
    return {
        "results": added_memories,
        "duplicates": dedup_result.duplicates,
        "merged": dedup_result.merged,
        "link_pairs": dedup_result.link_pairs,
        "saved_message_max_seq": saved_message_max_seq,
    }
