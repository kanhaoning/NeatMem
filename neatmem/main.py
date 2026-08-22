import asyncio
import json
import os
import time
import uuid
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from openai import OpenAI
from neatmem.memory_add import add_memories
from neatmem.batching import (
    FlushConflictError,
    VECTOR_STORE_TRACK,
    compute_next_batch,
    flush_scope,
)

# Per-user 写入锁，防止同一用户并发写入导致覆盖
_user_locks: dict[str, asyncio.Lock] = {}


def _get_user_lock(user_id: str) -> asyncio.Lock:
    if user_id not in _user_locks:
        _user_locks[user_id] = asyncio.Lock()
    return _user_locks[user_id]


from neatmem.config import (
    build_memory_store,
    logger,
    ENABLE_BM25,
    ENABLE_ENTITY,
    HISTORY_DB_PATH,
    EXTRACT_LAST_K_MESSAGES,
    MESSAGE_STORE_BACKEND,
    ENTITY_EXTRACTOR_BACKEND,
    ENTITY_STORE_BACKEND,
    ENABLE_GRAPH,
    GRAPH_SEARCH_TOP_K,
    LLM_API_KEY,
    LLM_BASE_URL,
    MESSAGE_BATCHING_ENABLED,
    MESSAGE_BATCHING_CHECK_INTERVAL_SECS,
    MESSAGE_BATCH_SIZE,
    MESSAGE_BATCH_DEADLINE_SECS,
    DEDUP_DETECTOR,
)
from neatmem.rerank import (
    llm_rerank,
    validate_rerank_prompt_at_boot,
    RERANK_MODE,
    LLM_RERANK_CANDS,
    CROSS_ENCODER_CANDS,
)
from neatmem.storage.message.factory import create_message_store
from neatmem.memory_search import search_memories
from neatmem.signals.entity.factory import create_entity_extractor
from neatmem.storage.entity.factory import create_entity_store
from neatmem.signals.bm25.factory import create_bm25_index

# BM25 works best with spaCy lemmatization; degrade to raw tokens if missing
# (service stays up — keyword matching just loses word-form normalization).
if ENABLE_BM25:
    _spacy_ok = False
    try:
        import spacy
        _spacy_ok = spacy.util.is_package("en_core_web_sm")
    except Exception:
        _spacy_ok = False
    if not _spacy_ok:
        logger.warning(
            "spaCy not installed — BM25 keyword search will only match exact word forms "
            "(e.g. searching \"memory\" won't match stored \"memories\"). "
            "For better keyword matching: pip install \"neatmem[nlp]\" && "
            "python -m spacy download en_core_web_sm"
        )

# 初始化存储层（自研 MemoryStore，mem0-compatible API）
memory = build_memory_store()

# Validate prompt overrides at boot (fail fast; also warms the loader cache).
from neatmem.prompts.loader import load_prompt
from neatmem.prompts.extraction import ADDITIVE_EXTRACTION_PROMPT
from neatmem.memory_add import ACTION_DEDUP_PROMPT

load_prompt("EXTRACTION_PROMPT", ADDITIVE_EXTRACTION_PROMPT)
load_prompt(
    "DEDUP_PROMPT", ACTION_DEDUP_PROMPT, ("new_text", "candidate_block"),
    supported_placeholders=("new_text", "candidate_block"),
)
load_prompt(
    "REWRITE_PROMPT",
    default_file="rewrite_en.txt",
    required_placeholders=("statement_1", "statement_2"),
    supported_placeholders=("statement_1", "statement_2", "relation", "metadata_1", "metadata_2"),
)
# {relation} is only supplied on the pointwise path; which set an override may
# use depends on the configured detector.
_edit_supported = ("old_text", "new_text") + (
    ("relation",) if DEDUP_DETECTOR == "pointwise" else ()
)
load_prompt(
    "EDIT_PROMPT",
    default_file="edit_en.txt",
    required_placeholders=("old_text", "new_text"),
    supported_placeholders=_edit_supported,
)
validate_rerank_prompt_at_boot()

# 初始化消息历史存储
message_store = create_message_store(
    HISTORY_DB_PATH,
    extract_last_k=EXTRACT_LAST_K_MESSAGES,
    backend=MESSAGE_STORE_BACKEND,
)

# 初始化自研 entity 提取 / 存储
entity_extractor = create_entity_extractor(ENTITY_EXTRACTOR_BACKEND)
entity_store = create_entity_store(
    ENTITY_STORE_BACKEND,
    qdrant_client=memory.vector_store.client,
    collection_name=os.environ.get("ENTITY_COLLECTION_NAME", f"{memory.collection_name}_entities"),
    vector_size=memory.vector_store.embedding_model_dims,
)

# 初始化自研 BM25 索引
bm25_index = create_bm25_index(
    "qdrant_sparse" if ENABLE_BM25 else "none",
    vector_store=memory.vector_store,
    collection_name=memory.collection_name,
)

# --- 启动契约断言：显式开启的信号必须可用，失败直接 raise（不静默降级）---
if ENABLE_BM25:
    _info = memory.vector_store.client.get_collection(memory.collection_name)
    _sparse_cfg = _info.config.params.sparse_vectors
    assert _sparse_cfg and "bm25" in _sparse_cfg, \
        f"ENABLE_BM25=true but collection '{memory.collection_name}' has no 'bm25' sparse slot"
    logger.info("启动契约: BM25 sparse slot OK")

if ENABLE_ENTITY:
    memory.vector_store.client.get_collection(
        os.environ.get("ENTITY_COLLECTION_NAME", f"{memory.collection_name}_entities"))
    logger.info("启动契约: entity collection 可达")

if ENABLE_GRAPH:
    from neatmem.signals.graph.factory import get_graph_store
    get_graph_store()  # kuzu 打不开时此处直接 raise
    logger.info("启动契约: graph store (kuzu) 可打开")

# NeatMem 自建 LLM 客户端（与 mem0 解耦）
# Key/base_url resolution lives in config.py (LLM_API_KEY/LLM_BASE_URL):
# explicit env > provider preset > OpenAI default.
openai_client = OpenAI(
    api_key=LLM_API_KEY,
    base_url=LLM_BASE_URL,
)
LLM_MODEL = os.getenv("LLM_MODEL")
if not LLM_MODEL:
    raise SystemExit(
        "ERROR: LLM_MODEL is not set (no default — a wrong default is worse than none).\n"
        "  Pick a model for your provider, e.g.:\n"
        "    LLM_PROVIDER=minimax LLM_MODEL=MiniMax-M3\n"
        "  See docs/configuration.md for provider presets."
    )

# 限制同时进行的 rerank LLM 调用数，避免触发 MiniMax Token Plan 限速
RERANK_MAX_CONCURRENT = int(os.getenv("RERANK_MAX_CONCURRENT", "12"))
_rerank_semaphore = asyncio.Semaphore(RERANK_MAX_CONCURRENT)


# --- 批处理调度器（进程内 asyncio 任务） ---
# 队列模式 = messages/add 只存不抽 + 本调度器按游标攒批后调同步 add 提取。
# 提取语义只有一份（add_memories）；调度器只是"代替客户端调 add"的机械件。
async def _extract_batch_for_scope(scope: Dict[str, str], message_ids: List[str], req_id: str) -> None:
    """Run sync-add extraction for a scheduled batch (scheduler path).

    Uses the external-injection branch of add_memories: messages come from
    the store by ID (already saved by /v1/messages/add/), last_k is fetched
    with before_seq so it matches the inline lastk_before_save semantics.
    """
    rows = await asyncio.to_thread(message_store.get_messages_by_ids, message_ids)
    batch_messages = [
        {"role": r["role"], "content": r["content"], **({"name": r["name"]} if r["name"] else {})}
        for r in rows
    ]
    if not batch_messages:
        raise ValueError(f"scheduled batch vanished from store: {message_ids}")
    min_seq = rows[0]["seq"]  # rows are ordered by seq ASC
    lk_filters = {
        k: v
        for k, v in {
            "user_id": scope["user_id"],
            "agent_id": scope["agent_id"] or None,
        }.items()
        if v
    }
    k = getattr(message_store, "extract_last_k", 10)
    last_k = await asyncio.to_thread(
        message_store.get_last_messages, lk_filters, k, min_seq
    )
    lock = _get_user_lock(scope["user_id"])
    async with lock:
        await asyncio.to_thread(
            add_memories,
            memory=memory,
            openai_client=openai_client,
            llm_model=LLM_MODEL,
            messages=batch_messages,
            user_id=scope["user_id"],
            agent_id=scope["agent_id"] or None,
            run_id=scope["run_id"] or None,
            app_id=None,
            metadata=None,
            custom_instructions=None,
            req_id=req_id,
            message_store=message_store,
            extract_last_k=None,  # last_k already limited above; no re-truncation
            last_k_messages_input=last_k,
            entity_extractor=entity_extractor,
            entity_store=entity_store,
            bm25_index=bm25_index,
        )


async def _batch_scheduler_loop() -> None:
    """Poll pending messages per scope and extract full/deadline-flushed batches.

    Cursor commit happens only after extraction succeeds (at-least-once):
    a failed batch is retried on the next iteration. Scopes are processed
    serially — never parallelize per scope, or a slow failed batch could be
    overtaken by a fast one and its messages skipped (see plan §11.3).
    """
    logger.info(
        "批处理调度器启动 | interval=%ss batch_size=%s deadline=%ss",
        MESSAGE_BATCHING_CHECK_INTERVAL_SECS, MESSAGE_BATCH_SIZE, MESSAGE_BATCH_DEADLINE_SECS,
    )
    while True:
        try:
            scopes = await asyncio.to_thread(message_store.list_message_scopes)
            for scope in scopes:
                if not scope["user_id"]:
                    # Extraction requires a user scope; messages saved without
                    # user_id stay pending forever (visible via pending_count).
                    continue
                batch = await asyncio.to_thread(
                    compute_next_batch,
                    message_store,
                    scope,
                    store_track=VECTOR_STORE_TRACK,
                    batch_size=MESSAGE_BATCH_SIZE,
                    deadline_secs=MESSAGE_BATCH_DEADLINE_SECS,
                )
                if not batch["message_ids"]:
                    continue
                req_id = uuid.uuid4().hex[:8]
                try:
                    await _extract_batch_for_scope(scope, batch["message_ids"], req_id)
                except Exception:
                    logger.exception(
                        "[%s] 批提取失败 scope=%s seqs=%s..%s, 游标不推进, 下轮重试",
                        req_id, scope, batch["seqs"][0], batch["seqs"][-1],
                    )
                    continue
                await asyncio.to_thread(
                    message_store.advance_cursor,
                    scope["user_id"], scope["agent_id"], scope["run_id"],
                    VECTOR_STORE_TRACK, batch["seqs"][-1],
                )
                logger.info(
                    "[%s] 批提取完成 scope=%s 批=%s条 seqs=%s..%s pending=%s",
                    req_id, scope, len(batch["seqs"]), batch["seqs"][0],
                    batch["seqs"][-1], batch["pending_count"],
                )
        except Exception:
            logger.exception("批处理调度器本轮扫描失败, 下轮继续")
        await asyncio.sleep(MESSAGE_BATCHING_CHECK_INTERVAL_SECS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context for startup/shutdown hooks."""
    scheduler_task = None
    if MESSAGE_BATCHING_ENABLED and message_store is not None:
        scheduler_task = asyncio.create_task(_batch_scheduler_loop())
    yield
    if scheduler_task is not None:
        scheduler_task.cancel()
        with suppress(asyncio.CancelledError):
            await scheduler_task
    if message_store is not None:
        message_store.close()
        logger.info("MessageStore closed on shutdown")


app = FastAPI(
    title="NeatMem",
    description="A local mem0-compatible memory server for AI agents",
    lifespan=lifespan,
)

# 请求日志中间件
@app.middleware("http")
async def log_requests(request: Request, call_next):
    # 跳过健康检查，减少噪音
    if request.url.path in ("/v1/ping/", "/health"):
        return await call_next(request)

    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"{request.method} {request.url.path} → {response.status_code} ({process_time:.3f}s)")
    return response

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 请求模型
class AddMemoryRequest(BaseModel):
    messages: Optional[List[Dict[str, Any]]] = None
    # neatmem extension: extract from messages already in the store (queue mode).
    # Mutually exclusive with `messages`.
    message_ids: Optional[List[str]] = None
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    app_id: Optional[str] = None
    run_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    immutable: bool = False
    infer: bool = True
    expiration_date: Optional[str] = None
    categories: Optional[List[str]] = None
    custom_instructions: Optional[str] = None
    prompt: Optional[str] = None
    extract_last_k: Optional[int] = None
    last_k_messages: Optional[List[Dict[str, Any]]] = None  # 外部注入用

class SearchMemoryRequest(BaseModel):
    query: str
    top_k: int = 20  # 对齐 mem0 默认值（memory/main.py:1247）
    threshold: float = 0.1
    filters: Optional[Dict[str, Any]] = None
    rerank: Optional[bool] = None  # None=跟全局开关, True/False=强制
    keyword_search: bool = False
    fields: Optional[List[str]] = None

class ListMemoryRequest(BaseModel):
    filters: Optional[Dict[str, Any]] = None
    page: int = 1
    page_size: int = 100

class UpdateMemoryRequest(BaseModel):
    text: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class QueryMessagesRequest(BaseModel):
    app_id: Optional[str] = None
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    run_id: Optional[str] = None
    content_like: Optional[str] = None
    after: Optional[str] = None
    before: Optional[str] = None
    roles: Optional[List[str]] = None
    limit: int = 100
    offset: int = 0
    order: str = "desc"


class SessionsRequest(BaseModel):
    app_id: Optional[str] = None
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    limit: int = 100
    offset: int = 0


class DeleteMessagesRequest(BaseModel):
    app_id: Optional[str] = None
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    run_id: Optional[str] = None


class AddMessagesRequest(BaseModel):
    """Store-only message ingest (queue mode). Extraction is scheduled later."""
    messages: List[Dict[str, Any]]
    user_id: str  # required: cursor accounting needs a definite scope
    agent_id: Optional[str] = None
    app_id: Optional[str] = None
    run_id: Optional[str] = None


class NextBatchRequest(BaseModel):
    user_id: str
    agent_id: Optional[str] = None
    run_id: Optional[str] = None
    store: str = "vector"  # cursor track: vector (L1 facts) / graph (reserved)


class MarkProcessedRequest(BaseModel):
    user_id: str
    agent_id: Optional[str] = None
    run_id: Optional[str] = None
    store: str = "vector"
    last_processed_seq: int


class FlushMessagesRequest(BaseModel):
    user_id: str
    agent_id: Optional[str] = None
    run_id: Optional[str] = None
    store: str = "vector"

# 工具函数
def _build_filters(opts: Dict[str, Any]) -> Dict[str, Any]:
    """构建Mem0兼容的过滤器"""
    and_conditions = []
    if opts.get("user_id"):
        and_conditions.append({"user_id": opts["user_id"]})
    if opts.get("agent_id"):
        and_conditions.append({"agent_id": opts["agent_id"]})
    if opts.get("app_id"):
        and_conditions.append({"app_id": opts["app_id"]})
    if opts.get("run_id"):
        and_conditions.append({"run_id": opts["run_id"]})

    if opts.get("extraFilters"):
        for k, v in opts["extraFilters"].items():
            and_conditions.append({k: v})

    if len(and_conditions) == 1:
        return and_conditions[0]
    if len(and_conditions) > 1:
        return {"AND": and_conditions}
    return {}

def _convert_memory_format(mem: Dict[str, Any]) -> Dict[str, Any]:
    """转换Mem0返回格式到OpenClaw期望的格式"""
    result = {
        "id": mem.get("id", ""),
        "memory": mem.get("memory", ""),
        "hash": mem.get("hash", ""),
        "metadata": mem.get("metadata", {}),
        "score": mem.get("score", 0.0),
        "created_at": mem.get("created_at", datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")),
        "updated_at": mem.get("updated_at", None),
        "user_id": mem.get("user_id", None),
        "agent_id": mem.get("agent_id", None),
        "app_id": mem.get("app_id", None),
        "run_id": mem.get("run_id", None),
    }
    # Pass through event (mem0 add result items may carry ADD/UPDATE/DELETE/NONE markers; omit key when absent)
    if "event" in mem:
        result["event"] = mem["event"]
    # 透传 attributed_to（可能在 metadata 子字段中，key 为 attr_source）
    attr = mem.get("attributed_to") or (mem.get("metadata") or {}).get("attr_source")
    if attr:
        result["attributed_to"] = attr
    return result

# 健康检查接口
@app.get("/v1/ping/")
async def ping():
    return {"status": "ok", "version": "0.1.0-preview", "backend": "neatmem"}

@app.get("/health")
async def health_check():
    # 信号状态可见化：bm25 slot 有无、各 collection 点数、graph 初始化状态
    signals = {}
    try:
        info = memory.vector_store.client.get_collection(memory.collection_name)
        sparse_cfg = info.config.params.sparse_vectors
        signals["mem0_points"] = info.points_count
        signals["bm25_slot"] = bool(sparse_cfg and "bm25" in sparse_cfg)
    except Exception as e:
        signals["mem0_error"] = str(e)
    if ENABLE_ENTITY:
        try:
            ent = memory.vector_store.client.get_collection(
                os.environ.get("ENTITY_COLLECTION_NAME", f"{memory.collection_name}_entities"))
            signals["entity_points"] = ent.points_count
        except Exception as e:
            signals["entity_error"] = str(e)
    signals["graph_enabled"] = ENABLE_GRAPH
    if ENABLE_GRAPH:
        try:
            from neatmem.signals.graph.factory import get_graph_store
            get_graph_store()
            signals["graph"] = "initialized"
        except Exception as e:
            signals["graph_error"] = str(e)
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat(),
            "signals": signals}

# 记忆接口
@app.post("/v1/memories/")
async def add_memory(request: AddMemoryRequest):
    if request.messages and request.message_ids:
        raise HTTPException(status_code=400, detail="messages and message_ids are mutually exclusive")
    if not request.messages and not request.message_ids:
        raise HTTPException(status_code=400, detail="messages is required")
    if request.message_ids and not request.infer:
        raise HTTPException(status_code=400, detail="message_ids requires infer=true")

    req_id = uuid.uuid4().hex[:8]
    _log = lambda msg: logger.info(f"[{req_id}] {msg}")

    _log(f"添加记忆 | user={request.user_id or 'default_user'} agent={request.agent_id} msgs={len(request.messages or [])} ids={len(request.message_ids or [])}")
    if request.custom_instructions:
        _log(f"自定义规则(前200字): {request.custom_instructions[:200]}")

    if not request.user_id:
        request.user_id = "default_user"

    if request.infer:
        # message_ids 分支（队列模式）：消息已在 message_store（由 /v1/messages/add/ 写入），
        # 按 ID 取出并重建成 {role, content, name?}，保证提取输入与内联路径逐字节一致；
        # last_k 取本批第一条之前的 k 条（before_seq），等价于内联的 lastk_before_save。
        batch_messages = None
        batch_last_k = None
        if request.message_ids:
            rows = await asyncio.to_thread(message_store.get_messages_by_ids, request.message_ids)
            found = {r["message_id"] for r in rows}
            missing = [mid for mid in request.message_ids if mid not in found]
            if missing:
                raise HTTPException(
                    status_code=404,
                    detail=f"{len(missing)} message_ids not found in store, first: {missing[0]}",
                )
            batch_messages = [
                {"role": r["role"], "content": r["content"], **({"name": r["name"]} if r["name"] else {})}
                for r in rows
            ]
            min_seq = rows[0]["seq"]  # rows ordered by seq ASC
            lk_filters = {
                k: v
                for k, v in {
                    "user_id": request.user_id,
                    "agent_id": request.agent_id,
                    "app_id": request.app_id,
                }.items()
                if v
            }
            k = request.extract_last_k if request.extract_last_k is not None else getattr(message_store, "extract_last_k", 10)
            batch_last_k = await asyncio.to_thread(
                message_store.get_last_messages, lk_filters, k, min_seq
            )

        # per-user 锁，防止同一用户并发写入导致覆盖
        lock = _get_user_lock(request.user_id)
        async with lock:
            result = add_memories(
                memory=memory,
                openai_client=openai_client,
                llm_model=LLM_MODEL,
                messages=batch_messages if batch_messages is not None else request.messages,
                user_id=request.user_id,
                agent_id=request.agent_id,
                run_id=request.run_id,
                app_id=request.app_id,
                metadata=request.metadata,
                custom_instructions=request.custom_instructions,
                req_id=req_id,
                message_store=message_store,
                # message_ids 分支: last_k 已按 k 截好, 不再二次截断
                extract_last_k=None if batch_messages is not None else request.extract_last_k,
                last_k_messages_input=batch_last_k if batch_messages is not None else request.last_k_messages,
                entity_extractor=entity_extractor,
                entity_store=entity_store,
                bm25_index=bm25_index,
            )

        # 内联路径的 vector 轨游标由 add_memories 内部在推进成功后推进
        # （_advance_cursor_after_save）——eval ingest 进程内直调 add_memories
        # 不经过本端点，推进必须放在库里。message_ids 分支不推进：
        # 提交游标是调用方职责（mark-processed 协议）。

        memories = [_convert_memory_format(item) for item in result.get("results", [])]
        # add 返回带 user_id(不是 null),从 request 填
        for mem in memories:
            if mem.get("user_id") is None:
                mem["user_id"] = request.user_id
        duplicates = result.get("duplicates", [])
        merged = result.get("merged", [])

        _log(f"完成 | 新增 {len(memories)} 条, 冗余替换 {len(duplicates)} 条, 合并 {len(merged)} 条")
        for i, mem in enumerate(memories):
            _log(f"  新增{i+1}: {mem['memory'][:120]}")
        for i, dup in enumerate(duplicates):
            _log(f"  替换{i+1}: '{dup['old_text'][:80]}' → '{dup['new_text'][:80]}' (score={dup['score']:.4f})")
        for i, m in enumerate(merged):
            _log(f"  合并{i+1}: '{m['old_text'][:80]}' + '{m['new_text'][:80]}' → '{m['merged_text'][:120]}'")

        return {"results": memories, "duplicates": duplicates, "merged": merged}
    else:
        # 直接写入模式：仍须保存原始消息到 message_store，保证后续 infer=True 能读取历史上下文
        msg_filters = {
            "app_id": request.app_id,
            "user_id": request.user_id,
            "agent_id": request.agent_id,
            "run_id": request.run_id,
        }
        msg_filters = {k: v for k, v in msg_filters.items() if v}
        saved = []
        if message_store is not None and msg_filters:
            saved = await asyncio.to_thread(message_store.save_messages, request.messages, msg_filters)

        # mem0 原生 add(infer=False)：直接存储
        add_params = {
            "messages": request.messages,
            "user_id": request.user_id,
            "infer": False,
        }
        if request.agent_id:
            add_params["agent_id"] = request.agent_id
        if request.app_id:
            add_params["app_id"] = request.app_id
        if request.metadata:
            add_params["metadata"] = request.metadata

        result = memory.add(**add_params)

        # infer=False 语义是"内容已是记忆, 无需提取"；保存成功后推进游标,
        # 否则调度器会把这批原始消息当待提取再抽一遍（双重写入）。
        if saved:
            await asyncio.to_thread(
                message_store.advance_cursor,
                request.user_id, request.agent_id or "", request.run_id or "",
                VECTOR_STORE_TRACK, max(m["seq"] for m in saved),
            )

        memories = [_convert_memory_format(item) for item in result.get("results", [])]
        _log(f"直接写入 {len(memories)} 条")
        return {"results": memories}

@app.post("/v2/memories/search/")
async def search_memory(request: SearchMemoryRequest):
    logger.info(f"[搜索记忆] 查询: {request.query}, top_k: {request.top_k}, 阈值: {request.threshold}")
    if request.filters:
        logger.info(f"[搜索记忆] 过滤器: {request.filters}")

    # 处理过滤器：和notebook逻辑保持一致，没有传的话默认使用user_id=default_user
    search_filters = request.filters
    if not search_filters or not any(k in search_filters for k in ['user_id', 'agent_id', 'run_id']):
        search_filters = {"user_id": "default_user"}
        logger.info(f"[搜索记忆] 自动添加默认过滤器: {search_filters}")

    # --- 搜索路径：统一走 NeatMem memory_search（dense + entity boosting） ---
    # request.rerank is a per-request on/off switch; when absent it follows
    # the RERANK_MODE config (off = no rerank).
    use_rerank = request.rerank if request.rerank is not None else (RERANK_MODE != "off")

    # With rerank on, dense recall must cover the rerank head (cands);
    # otherwise the head is truncated by top_k and cands is a dead parameter.
    search_top_k = request.top_k
    if use_rerank:
        active_cands = LLM_RERANK_CANDS if RERANK_MODE == "llm" else CROSS_ENCODER_CANDS
        search_top_k = max(request.top_k, active_cands)

    result = await asyncio.to_thread(
        search_memories,
        memory=memory,
        query=request.query,
        filters=search_filters,
        top_k=search_top_k,
        threshold=request.threshold,
        entity_extractor=entity_extractor,
        entity_store=entity_store,
        use_entity=ENABLE_ENTITY,
        use_bm25=ENABLE_BM25,
        bm25_index=bm25_index,
    )
    candidates = result["results"]

    if use_rerank:
        t0 = time.monotonic()
        async with _rerank_semaphore:
            rank_result = await asyncio.to_thread(
                llm_rerank, openai_client, LLM_MODEL, request.query, candidates,
                top_k=request.top_k)
        reranked = rank_result.kept[:request.top_k]  # 最终截断到 top_k（删 cap*2，head/tail 由 rerank 返回）
        rerank_ms = (time.monotonic() - t0) * 1000
        logger.info(f"[rerank:{RERANK_MODE}] 耗时 {rerank_ms:.0f}ms, 保留 {len(reranked)} 条")

        memories = [_convert_memory_format(item) for item in reranked]
    else:
        memories = [_convert_memory_format(item) for item in candidates[:request.top_k]]

    logger.info(f"[搜索记忆成功] 找到 {len(memories)} 条相关记忆")
    for i, mem in enumerate(memories):
        logger.info(f"  结果{i+1} (得分{mem['score']:.3f}): {mem['memory'][:100]}...")

    # --- 图记忆 hook（mem0 1.0.11 忠实复现）---
    # 关图时 ENABLE_GRAPH=false，直接 return {"results": memories}，与 baseline 逐字段一致。
    # 开图时追加 graph_relations 字段；图检索失败则降级返回 baseline shape。
    if ENABLE_GRAPH:
        try:
            from neatmem.signals.graph.factory import get_graph_store
            gs = get_graph_store()
            graph_relations = await asyncio.to_thread(gs.search, request.query, search_filters, GRAPH_SEARCH_TOP_K)
            logger.info(f"[图记忆] 返回 {len(graph_relations)} 条关系")
            return {"results": memories, "graph_relations": graph_relations}
        except Exception as e:
            logger.warning(f"[图记忆] search failed: {e}")

    return {"results": memories}

# 兼容v1搜索接口
@app.post("/v1/search/")
async def search_memory_v1(request: SearchMemoryRequest):
    return await search_memory(request)

@app.get("/v1/memories/{memory_id}/")
async def get_memory(memory_id: str):
    result = memory.get(memory_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Memory with id {memory_id} not found")
    return _convert_memory_format(result)

@app.post("/v2/memories/")
async def list_memories(request: ListMemoryRequest, page: int = 1, page_size: int = 100):
    # 调用Mem0获取所有记忆
    result = memory.get_all(filters=request.filters)

    # 分页处理
    start = (page - 1) * page_size
    end = start + page_size
    paginated = result.get("results", [])[start:end]

    # 转换格式
    memories = []
    for item in paginated:
        memories.append(_convert_memory_format(item))

    return {"results": memories, "page": page, "page_size": page_size}

@app.put("/v1/memories/{memory_id}/")
async def update_memory(memory_id: str, request: UpdateMemoryRequest):
    logger.info(f"[更新记忆] 记忆ID: {memory_id}")
    if request.text:
        logger.info(f"  新内容: {request.text[:100]}...")
    if request.metadata:
        logger.info(f"  新元数据: {request.metadata}")

    result = memory.update(
        memory_id=memory_id,
        data=request.text,
        metadata=request.metadata,
    )

    formatted = _convert_memory_format(result)
    logger.info(f"[更新记忆成功] 记忆ID: {memory_id}, 更新后内容: {formatted['memory'][:100]}...")
    return formatted

@app.delete("/v1/memories/{memory_id}/")
async def delete_memory(memory_id: str):
    memory.delete(memory_id)
    return {"status": "ok", "message": "Memory deleted successfully"}

@app.delete("/v1/memories/")
async def delete_all_memories(user_id: Optional[str] = None, agent_id: Optional[str] = None, app_id: Optional[str] = None, run_id: Optional[str] = None):
    filters = {}
    if user_id:
        filters["user_id"] = user_id
    if agent_id:
        filters["agent_id"] = agent_id
    if app_id:
        filters["app_id"] = app_id
    if run_id:
        filters["run_id"] = run_id

    memory.delete_all(**filters)
    return {"status": "ok", "message": "All memories deleted successfully"}

@app.get("/v1/memories/{memory_id}/history/")
async def memory_history(memory_id: str):
    return {"results": await asyncio.to_thread(memory.history, memory_id)}

@app.post("/v1/reset/")
async def reset_memory():
    # Reset mem0 memories only; message history is controlled separately via /v1/messages/reset/
    await asyncio.to_thread(memory.reset)
    return {"status": "ok", "message": "All memories reset successfully"}

# 实体接口
@app.delete("/v2/entities/{entity_type}/{entity_id}/")
async def delete_entity(entity_type: str, entity_id: str):
    filters = {f"{entity_type}_id": entity_id}
    memory.delete_all(**filters)
    return {"status": "ok", "message": f"Entity {entity_type}:{entity_id} deleted successfully"}

@app.get("/v1/entities/")
async def list_entities():
    # 返回空列表兼容平台模式
    return {"results": []}

# 事件接口
@app.get("/v1/events/")
async def list_events():
    # 返回空列表兼容平台模式
    return {"results": []}

@app.get("/v1/event/{event_id}/")
async def get_event(event_id: str):
    # 返回空对象兼容平台模式
    return {}

# 消息历史接口
@app.post("/v1/messages/add/")
async def add_messages(request: AddMessagesRequest):
    """Store-only ingest: messages are persisted but NOT extracted.

    Extraction happens later via the in-process scheduler (queue mode) or an
    external worker driving next-batch/mark-processed. Returns the assigned
    message_id + seq per message so callers can correlate.
    """
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages must be non-empty")
    filters = {
        k: v
        for k, v in {
            "app_id": request.app_id,
            "user_id": request.user_id,
            "agent_id": request.agent_id,
            "run_id": request.run_id,
        }.items()
        if v
    }
    saved = await asyncio.to_thread(message_store.save_messages, request.messages, filters)
    return {"results": saved, "count": len(saved)}


@app.post("/v1/messages/next-batch/")
async def next_batch(request: NextBatchRequest):
    """Return IDs/seqs of the next batch due for extraction (no content).

    Full batch when >= MESSAGE_BATCH_SIZE pending; partial batch only when the
    oldest pending message exceeds MESSAGE_BATCH_DEADLINE_SECS; empty otherwise.
    """
    scope = {
        "user_id": request.user_id,
        "agent_id": request.agent_id or "",
        "run_id": request.run_id or "",
    }
    result = await asyncio.to_thread(
        compute_next_batch,
        message_store,
        scope,
        store_track=request.store,
        batch_size=MESSAGE_BATCH_SIZE,
        deadline_secs=MESSAGE_BATCH_DEADLINE_SECS,
    )
    return result


@app.post("/v1/messages/mark-processed/")
async def mark_processed(request: MarkProcessedRequest):
    """Advance the extraction cursor after a batch was successfully extracted.

    Rejects regression (last_processed_seq <= current cursor) with 409 —
    cursors never move backwards through this endpoint.
    """
    if request.last_processed_seq < 1:
        raise HTTPException(status_code=400, detail="last_processed_seq must be >= 1")
    ok = await asyncio.to_thread(
        message_store.advance_cursor,
        request.user_id, request.agent_id or "", request.run_id or "",
        request.store, request.last_processed_seq,
    )
    if not ok:
        current = await asyncio.to_thread(
            message_store.get_cursor,
            request.user_id, request.agent_id or "", request.run_id or "", request.store,
        )
        raise HTTPException(
            status_code=409,
            detail=f"cursor regression: requested {request.last_processed_seq} <= current {current}",
        )
    return {"marked": True, "last_processed_seq": request.last_processed_seq}


@app.post("/v1/messages/flush/")
async def flush_messages(request: FlushMessagesRequest):
    """Synchronously extract all pending messages for the scope, right now.

    Bypasses the full-batch/deadline policy: pending messages are extracted
    in chunks of MESSAGE_BATCH_SIZE (final chunk may be partial). Extraction
    itself is identical to the scheduler path (add by message_ids).
    """
    scope = {
        "user_id": request.user_id,
        "agent_id": request.agent_id or "",
        "run_id": request.run_id or "",
    }

    async def _extract(s: Dict[str, str], message_ids: List[str]) -> None:
        await _extract_batch_for_scope(s, message_ids, uuid.uuid4().hex[:8])

    try:
        return await flush_scope(
            message_store,
            scope,
            store_track=request.store,
            batch_size=MESSAGE_BATCH_SIZE,
            extract_batch=_extract,
        )
    except FlushConflictError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.post("/v1/messages/query/")
async def query_messages(request: QueryMessagesRequest):
    if not any([request.app_id, request.user_id, request.agent_id, request.run_id]):
        raise HTTPException(
            status_code=400,
            detail="At least one of app_id, user_id, agent_id, run_id is required",
        )
    if request.limit < 1 or request.limit > 1000:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 1000")
    if request.offset < 0:
        raise HTTPException(status_code=400, detail="offset must be >= 0")
    if request.order not in ("asc", "desc"):
        raise HTTPException(status_code=400, detail="order must be 'asc' or 'desc'")

    filters = {
        k: v
        for k, v in {
            "app_id": request.app_id,
            "user_id": request.user_id,
            "agent_id": request.agent_id,
            "run_id": request.run_id,
        }.items()
        if v
    }

    def _query():
        msgs = message_store.query_messages(
            filters,
            content_like=request.content_like,
            after=request.after,
            before=request.before,
            roles=request.roles,
            limit=request.limit,
            offset=request.offset,
            order=request.order,
        )
        total = message_store.count_messages(
            filters,
            content_like=request.content_like,
            after=request.after,
            before=request.before,
            roles=request.roles,
        )
        return msgs, total

    msgs, total = await asyncio.to_thread(_query)
    return {
        "messages": msgs,
        "total": total,
        "limit": request.limit,
        "offset": request.offset,
    }


@app.post("/v1/messages/sessions/")
async def list_sessions(request: SessionsRequest):
    if not any([request.app_id, request.user_id, request.agent_id]):
        raise HTTPException(
            status_code=400,
            detail="At least one of app_id, user_id, agent_id is required",
        )
    if request.limit < 1 or request.limit > 1000:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 1000")
    if request.offset < 0:
        raise HTTPException(status_code=400, detail="offset must be >= 0")

    filters = {
        k: v
        for k, v in {
            "app_id": request.app_id,
            "user_id": request.user_id,
            "agent_id": request.agent_id,
        }.items()
        if v
    }

    def _query():
        sessions = message_store.list_sessions(
            filters,
            limit=request.limit,
            offset=request.offset,
        )
        return sessions

    sessions = await asyncio.to_thread(_query)
    return {"sessions": sessions, "total": len(sessions)}


@app.post("/v1/messages/delete/")
async def delete_messages(request: DeleteMessagesRequest):
    if not any([request.app_id, request.user_id, request.agent_id, request.run_id]):
        raise HTTPException(
            status_code=400,
            detail="At least one of app_id, user_id, agent_id, run_id is required",
        )

    filters = {
        k: v
        for k, v in {
            "app_id": request.app_id,
            "user_id": request.user_id,
            "agent_id": request.agent_id,
            "run_id": request.run_id,
        }.items()
        if v
    }

    deleted = await asyncio.to_thread(message_store.delete_messages, filters)
    return {"deleted": deleted}


@app.post("/v1/messages/reset/")
async def reset_messages():
    await asyncio.to_thread(message_store.reset)
    return {"reset": True}


if __name__ == "__main__":
    host = os.environ.get("NEATMEM_HOST", "0.0.0.0")
    port = int(os.environ.get("NEATMEM_PORT", "8790"))
    uvicorn.run(app, host=host, port=port)
