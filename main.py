import asyncio
import json
import os
import time
import uuid
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import uvicorn

from mem0 import Memory
from memory_add import add_memories

# Per-user 写入锁，防止同一用户并发写入导致覆盖
_user_locks: dict[str, asyncio.Lock] = {}

def _get_user_lock(user_id: str) -> asyncio.Lock:
    if user_id not in _user_locks:
        _user_locks[user_id] = asyncio.Lock()
    return _user_locks[user_id]

from config import config, logger, reranker_model_path, LLM_RERANK
from rerank import llm_rerank

# 初始化Mem0
memory = Memory.from_config(config)

app = FastAPI(title="NeatMem", description="A local mem0-compatible memory server for AI agents")

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

class SearchMemoryRequest(BaseModel):
    query: str
    top_k: int = 10
    threshold: float = 0.1
    filters: Optional[Dict[str, Any]] = None
    rerank: Optional[bool] = False  # 保留 Optional，因为代码中有 is not None 判断
    keyword_search: bool = False
    fields: Optional[List[str]] = None

class ListMemoryRequest(BaseModel):
    filters: Optional[Dict[str, Any]] = None
    page: int = 1
    page_size: int = 100

class UpdateMemoryRequest(BaseModel):
    text: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

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
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}

# 记忆接口
@app.post("/v1/memories/")
async def add_memory(request: AddMemoryRequest):
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages is required")

    req_id = uuid.uuid4().hex[:8]
    _log = lambda msg: logger.info(f"[{req_id}] {msg}")

    _log(f"添加记忆 | user={request.user_id or 'default_user'} agent={request.agent_id} msgs={len(request.messages)}")
    if request.custom_instructions:
        _log(f"自定义规则(前200字): {request.custom_instructions[:200]}")

    if not request.user_id:
        request.user_id = "default_user"

    if request.infer:
        # per-user 锁，防止同一用户并发写入导致覆盖
        lock = _get_user_lock(request.user_id)
        async with lock:
            result = add_memories(
                memory=memory,
                llm=memory.llm,
                messages=request.messages,
                user_id=request.user_id,
                agent_id=request.agent_id,
                run_id=request.run_id,
                metadata=request.metadata,
                custom_instructions=request.custom_instructions,
                req_id=req_id,
            )

        memories = [_convert_memory_format(item) for item in result.get("results", [])]
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
        # mem0 原生 add(infer=False)：直接存储
        add_params = {
            "messages": request.messages,
            "user_id": request.user_id,
            "infer": False,
        }
        if request.agent_id:
            add_params["agent_id"] = request.agent_id
        if request.metadata:
            add_params["metadata"] = request.metadata

        result = memory.add(**add_params)

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

    # --- 搜索路径：LLM rerank（新） 与 原有逻辑 互斥 ---
    if LLM_RERANK:
        # 向量搜索不做 rerank，由自己处理
        result = memory.search(
            query=request.query,
            limit=request.top_k,
            filters=search_filters,
            rerank=False,
        )

        # 余弦阈值粗筛（减少 LLM 调用量）
        candidates = [
            item for item in result.get("results", [])
            if item.get("score", 0) >= request.threshold
        ]

        # LLM 二分类精筛
        t0 = time.monotonic()
        reranked = llm_rerank(memory.llm, request.query, candidates, top_k=request.top_k)
        rerank_ms = (time.monotonic() - t0) * 1000
        logger.info(f"[LLM rerank] 耗时 {rerank_ms:.0f}ms, 保留 {len(reranked)} 条")

        # rerank 通过的直接放行，score 保留原始余弦分
        memories = [_convert_memory_format(item) for item in reranked]
    else:
        # 原有路径：完全不动
        use_rerank = request.rerank if request.rerank is not None else bool(reranker_model_path)
        result = memory.search(
            query=request.query,
            limit=request.top_k,
            filters=search_filters,
            rerank=use_rerank
        )

        memories = []
        for item in result.get("results", []):
            if "rerank_score" in item:
                item["score"] = item["rerank_score"]

            mem = _convert_memory_format(item)
            if mem["score"] >= request.threshold:
                memories.append(mem)

    logger.info(f"[搜索记忆成功] 找到 {len(memories)} 条相关记忆")
    for i, mem in enumerate(memories):
        logger.info(f"  结果{i+1} (得分{mem['score']:.3f}): {mem['memory'][:100]}...")

    return {"results": memories}

# 兼容v1搜索接口
@app.post("/v1/search/")
async def search_memory_v1(request: SearchMemoryRequest):
    return await search_memory(request)

@app.get("/v1/memories/{memory_id}/")
async def get_memory(memory_id: str):
    result = memory.get(memory_id)
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

if __name__ == "__main__":
    host = os.environ.get("NEATMEM_HOST", "0.0.0.0")
    port = int(os.environ.get("NEATMEM_PORT", "8790"))
    uvicorn.run(app, host=host, port=port)
