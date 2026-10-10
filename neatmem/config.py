import os
import logging

import numpy as np
from dotenv import load_dotenv
from langchain_community.embeddings import XinferenceEmbeddings

# 加载 .env 环境变量
load_dotenv()

# Qdrant 使用余弦相似度，分数方向天然"越大越相似"，无需 ChromaDB L2 补丁

# 配置 HuggingFace 镜像（fastembed BM25 模型下载需要）
if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# --- Embedder 配置 ---
EMBEDDER_PROVIDER = os.environ.get("EMBEDDER_PROVIDER", "siliconflow")  # siliconflow / openai / dashscope / xinference
EMBEDDER_MODEL = os.environ.get("EMBEDDER_MODEL", "BAAI/bge-m3")
# Per-provider presets (batch limits verified in smoke: DashScope hard-errors
# above 10). Explicit EMBEDDER_BASE_URL always wins over the preset.
EMBEDDER_PROVIDER_PRESETS = {
    "siliconflow": {"base_url": "https://api.siliconflow.cn/v1", "batch_size": 100},
    "openai": {"base_url": "https://api.openai.com/v1", "batch_size": 100},
    "dashscope": {"base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "batch_size": 10},
}
_EMBEDDER_PRESET = EMBEDDER_PROVIDER_PRESETS.get(EMBEDDER_PROVIDER, {})
EMBEDDER_BASE_URL = os.environ.get(
    "EMBEDDER_BASE_URL",
    _EMBEDDER_PRESET.get("base_url", "https://api.siliconflow.cn/v1"),
)
EMBEDDER_BATCH_SIZE = _EMBEDDER_PRESET.get("batch_size", 100)
# Generic key name preferred; SILICONFLOW_API_KEY kept as fallback (legacy envs).
EMBEDDER_API_KEY = os.environ.get("EMBEDDER_API_KEY") or os.environ.get("SILICONFLOW_API_KEY", "")
# Explicit dimension override. When unset, the dimension is auto-detected
# from the startup probe embedding (see build_memory_store).
EMBEDDER_DIMS = int(os.environ["EMBEDDER_DIMS"]) if os.environ.get("EMBEDDER_DIMS") else None

# --- Dedup query 截断上限 ---
# EMBEDDING_MAX_TOKENS 是 embedding 模型的 token 上下文上限，直接用作 dedup
# query 的字符截断预算（query[:EMBEDDING_MAX_TOKENS]，见 memory_add.py Step 1）。
# 换算依据（2026-09-23 探针实测，docs/internal-notes/20260922-batch-dedup-query-oversize-fix-plan.md §2）：
# 中/英/代码正常内容 token/字符 ≤ ~0.6（中文 0.50、英文 0.27、"两英文字母≈一中文字"实测成立），
# 等值字符截断最坏 ≈ 0.6×上限 token，天然带 ~1.7 倍余量；ZWJ emoji（实测 1.20）/
# 高密度中文等 >0.6 的尾部输入仍可能 400，由调度器熔断（连续失败跳批，main.py）收容。
# 解析优先级：显式 EMBEDDING_MAX_TOKENS > 按 EMBEDDER_MODEL 查表 > 保守 512。
# 查表覆盖常见模型；512 上下文模型（bge-large 系等）至今是主流，乐观默认会
# 给这类用户制造确定性 400 毒批，故未知模型一律走 512 + warning。
_EMBEDDING_MODEL_MAX_TOKENS_TABLE = [
    # (小写子串模式, max_tokens)；按序首个命中生效，具体模式须排在泛化模式前
    ("qwen3-embedding", 32768),
    ("bge-m3", 8192),
    ("bge-large", 512),
    ("bge-base", 512),
    ("bge-small", 512),
    ("bce-embedding", 512),
    ("text-embedding-v1", 2048),
    ("text-embedding-v2", 2048),
    ("text-embedding-v3", 8192),
    ("text-embedding-v4", 8192),
    ("text-embedding-3", 8191),
    ("text-embedding-ada-002", 8191),
]
_EMBEDDING_MAX_TOKENS_FALLBACK = 512


def resolve_embedding_max_tokens(model_name, override=None):
    """Resolve the dedup-query char budget from an explicit override or model name."""
    if override is not None:
        return override
    normalized = (model_name or "").lower().split("/")[-1]
    for pattern, max_tokens in _EMBEDDING_MODEL_MAX_TOKENS_TABLE:
        if pattern in normalized:
            return max_tokens
    logger.warning(
        "Unknown embedding model %r: falling back to EMBEDDING_MAX_TOKENS=%d. "
        "Set EMBEDDING_MAX_TOKENS explicitly if the model's context limit differs.",
        model_name, _EMBEDDING_MAX_TOKENS_FALLBACK,
    )
    return _EMBEDDING_MAX_TOKENS_FALLBACK


EMBEDDING_MAX_TOKENS = resolve_embedding_max_tokens(
    EMBEDDER_MODEL,
    int(os.environ["EMBEDDING_MAX_TOKENS"]) if os.environ.get("EMBEDDING_MAX_TOKENS") else None,
)

# --- LLM provider (multi-provider support) ---
# Explicit provider selects the verified parameter shape from PROVIDER_TABLE;
# unset keeps the legacy model-name matching (byte-identical legacy behavior).
from neatmem.utils.llm_client import normalize_provider, provider_default_base_url

LLM_PROVIDER = normalize_provider(os.environ.get("LLM_PROVIDER"))
# LLM_API_KEY preferred (mem0 server convention), OPENAI_API_KEY fallback.
LLM_API_KEY = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
# base_url priority: explicit OPENAI_BASE_URL > provider preset > OpenAI default.
LLM_BASE_URL = (
    os.environ.get("OPENAI_BASE_URL")
    or provider_default_base_url(LLM_PROVIDER)
    or "https://api.openai.com/v1"
)

# --- Data root: NEATMEM_DIR is the single parent dir for all local data.
# Priority per path: dedicated env > $NEATMEM_DIR/<child> > ~/.neatmem/<child>.
# MEM0_DIR is honored as a legacy fallback for users on mem0 layouts.
NEATMEM_DIR = (
    os.environ.get("NEATMEM_DIR")
    or os.environ.get("MEM0_DIR")
    or os.path.join(os.path.expanduser("~"), ".neatmem")
)

# --- 多信号开关（默认全开，A/B 测试时用环境变量切换）---
QDRANT_HOST = os.environ.get("QDRANT_HOST", "")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
QDRANT_PATH = os.environ.get("QDRANT_PATH") or os.path.join(NEATMEM_DIR, "qdrant")
ENABLE_BM25 = os.environ.get("ENABLE_BM25", "true").lower() == "true"
ENABLE_ENTITY = os.environ.get("ENABLE_ENTITY", "false").lower() == "true"

# --- 存储层构建（自研，向量存储仅支持 qdrant；对外 mem0-compatible API）---
# SQLite path for memory-change history (ADD/UPDATE/DELETE events).
# Distinct from MESSAGES_DB_PATH (chat message store).
MEMORY_HISTORY_DB_PATH = os.environ.get(
    "MEMORY_HISTORY_DB_PATH",
    os.path.join(NEATMEM_DIR, "history.db"),
)


def build_memory_store():
    """Construct the mem0-compatible MemoryStore from env configuration.

    Wires the self-managed parts: OpenAI-compatible embedder + Qdrant vector
    store + SQLite history. Boot-time contract (fail loudly, no silent
    degradation): a probe embedding is always issued at startup -- when
    EMBEDDER_DIMS is set its dimension must match, otherwise the probed
    dimension is auto-detected and used for the collection.
    """
    from neatmem.embeddings import LangchainEmbedder, OpenAIEmbedder
    from neatmem.memory_store import MemoryStore
    from neatmem.storage.vector.factory import create_vector_store

    try:
        if EMBEDDER_PROVIDER in EMBEDDER_PROVIDER_PRESETS:
            embedding_model = OpenAIEmbedder(
                model=EMBEDDER_MODEL,
                api_key=EMBEDDER_API_KEY,
                base_url=EMBEDDER_BASE_URL,
                expected_dims=EMBEDDER_DIMS,
                batch_size=EMBEDDER_BATCH_SIZE,
            )
        elif EMBEDDER_PROVIDER == "xinference":
            # 本地 Xinference Embedding (备用)
            embedding_model = LangchainEmbedder(XinferenceEmbeddings(
                server_url=os.environ.get("XINFERENCE_SERVER_URL", "http://localhost:9997"),
                model_uid=os.environ.get("XINFERENCE_MODEL_UID", "bge-m3")
            ))
        else:
            raise ValueError(
                f"Unknown EMBEDDER_PROVIDER={EMBEDDER_PROVIDER!r}, expected one of: "
                f"{', '.join(sorted(EMBEDDER_PROVIDER_PRESETS))}, xinference"
            )
        embedding_dims = EMBEDDER_DIMS
        if embedding_dims is None:
            # Probe once to auto-detect (also surfaces auth/network errors at boot).
            embedding_dims = len(embedding_model.embed("dimension self-check"))
            logger.info("Embedding dimension auto-detected: %d", embedding_dims)
    except SystemExit:
        raise
    except Exception as e:
        raise SystemExit(
            f"ERROR: Embedding API call failed ({type(e).__name__}: {e}).\n"
            "  Check your API key and base URL:\n"
            "  - EMBEDDER_API_KEY / EMBEDDER_BASE_URL (SiliconFlow: SILICONFLOW_API_KEY also accepted)\n"
            "  - OPENAI_API_KEY / OPENAI_BASE_URL (OpenAI-compatible LLM)"
        )

    vector_store = create_vector_store(
        "qdrant",
        collection_name="neatmem",
        embedding_model_dims=embedding_dims,
        **({"host": QDRANT_HOST, "port": QDRANT_PORT} if QDRANT_HOST else {"path": QDRANT_PATH}),
        on_disk=False,
    )

    return MemoryStore(
        vector_store=vector_store,
        embedding_model=embedding_model,
        history_db_path=MEMORY_HISTORY_DB_PATH,
    )

# Rerank params all live in neatmem/rerank.py (RERANK_MODE / LLM_RERANK_* / CROSS_ENCODER_*)

# --- Dedup 主参数（三轴正交） ---
# DEDUP_ENABLED  - 是否去重（false = 不去重全写入）
# DEDUP_RESOLVER - 判中后怎么处理：
#   skip     - update 降级 add（新旧共存）
#   replace  - new 直接覆盖 old
#   rewrite  - LLM 融合（pointwise 下固定走 memos resolver，默认）
#   edit     - LLM 生成 patch（F2 prompt）
# DEDUP_DETECTOR - 怎么判：
#   listwise - 1 次 LLM 调用判全批候选
#   listwise_multitarget - listwise 变体：逐候选独立判定，一次调用可更新多个目标（默认）
#   pointwise - MemOS 三分类逐对判定（contradictory/redundant/independent）
DEDUP_ENABLED = os.environ.get("DEDUP_ENABLED", "true").lower() == "true"
DEDUP_RESOLVER = os.environ.get("DEDUP_RESOLVER", "rewrite")
DEDUP_DETECTOR = os.environ.get("DEDUP_DETECTOR", "listwise_multitarget")
if DEDUP_RESOLVER not in ("skip", "replace", "rewrite", "edit"):
    raise ValueError(f"Invalid DEDUP_RESOLVER={DEDUP_RESOLVER!r}, expected skip|replace|rewrite|edit")
if DEDUP_DETECTOR not in ("listwise", "listwise_multitarget", "pointwise"):
    raise ValueError(f"Invalid DEDUP_DETECTOR={DEDUP_DETECTOR!r}, expected listwise|listwise_multitarget|pointwise")

# DEDUP_RECALL_THRESHOLD: 两个 detector 共用的召回截断
# （默认 0.40，基于 bge-m3 分数分布标定；评测可显式调高）
DEDUP_RECALL_THRESHOLD = float(os.environ.get("DEDUP_RECALL_THRESHOLD", "0.40"))
if not 0.0 <= DEDUP_RECALL_THRESHOLD <= 1.0:
    raise ValueError(f"Invalid DEDUP_RECALL_THRESHOLD={DEDUP_RECALL_THRESHOLD}, expected 0-1")

# --- Dedup Advanced 参数 ---
# dedup LLM thinking 开关
DEDUP_THINKING = os.environ.get("DEDUP_THINKING", "false").lower() == "true"

# --- Edit advanced params (only effective when DEDUP_RESOLVER=edit) ---
# edit (patch_diff) LLM thinking switch
EDIT_THINKING = os.environ.get("EDIT_THINKING", "false").lower() == "true"

logger.info("Vector store: Qdrant %s (BM25=%s, Entity=%s)",
             f"server ({QDRANT_HOST}:{QDRANT_PORT})" if QDRANT_HOST else f"local (path={QDRANT_PATH})",
             ENABLE_BM25, ENABLE_ENTITY)
logger.info("Dedup: enabled=%s, resolver=%s, detector=%s, recall_threshold=%.2f",
            DEDUP_ENABLED, DEDUP_RESOLVER, DEDUP_DETECTOR, DEDUP_RECALL_THRESHOLD)
logger.info("Dedup thinking=%s, edit thinking=%s", DEDUP_THINKING, EDIT_THINKING)

# --- 消息历史存储配置 ---
MESSAGES_DB_PATH = os.environ.get(
    "MESSAGES_DB_PATH",
    os.path.join(NEATMEM_DIR, "messages.db"),
)
# sqlite cannot create missing parent directories; make sure the data root
# (and any custom db parents) exist before stores open their files.
os.makedirs(NEATMEM_DIR, exist_ok=True)
for _db in (MESSAGES_DB_PATH, MEMORY_HISTORY_DB_PATH):
    os.makedirs(os.path.dirname(os.path.abspath(_db)), exist_ok=True)
EXTRACT_LAST_K_MESSAGES = int(os.environ.get("EXTRACT_LAST_K_MESSAGES", "10"))
MESSAGE_STORE_BACKEND = os.environ.get("MESSAGE_STORE_BACKEND", "sqlite")  # sqlite / none

# --- Server-side message batching (cursor-driven queue mode) ---
# Master switch for the in-process batch scheduler. When false the server is
# pure sync mode: POST /v1/memories/ extracts on arrival and no background
# task runs. The /v1/messages/add|next-batch|mark-processed/ endpoints work
# regardless of this switch.
MESSAGE_BATCHING_ENABLED = os.environ.get("MESSAGE_BATCHING_ENABLED", "true").lower() == "true"
# Scheduler poll interval.
MESSAGE_BATCHING_CHECK_INTERVAL_SECS = int(os.environ.get("MESSAGE_BATCHING_CHECK_INTERVAL_SECS", "30"))
# Full-batch size, aligned with eval BATCH_SIZE (10 messages per disjoint batch).
MESSAGE_BATCH_SIZE = int(os.environ.get("MESSAGE_BATCH_SIZE", "10"))
# Batch execution deadline: when the oldest pending message exceeds this age,
# a partial batch is flushed even if MESSAGE_BATCH_SIZE is not reached.
MESSAGE_BATCH_DEADLINE_SECS = int(os.environ.get("MESSAGE_BATCH_DEADLINE_SECS", "600"))
# Poison-batch circuit breaker: a batch failing this many consecutive times is
# skipped (cursor advances past it) with an error log, instead of being retried
# forever. Skipped messages stay in the messages table and can be replayed by
# resetting the cursor.
MESSAGE_BATCH_MAX_CONSECUTIVE_FAILURES = int(os.environ.get("MESSAGE_BATCH_MAX_CONSECUTIVE_FAILURES", "10"))

logger.info("Message batching: enabled=%s, interval=%ss, batch_size=%s, deadline=%ss",
            MESSAGE_BATCHING_ENABLED, MESSAGE_BATCHING_CHECK_INTERVAL_SECS,
            MESSAGE_BATCH_SIZE, MESSAGE_BATCH_DEADLINE_SECS)

logger.info("Message history: backend=%s, path=%s (extract_last_k=%s)",
            MESSAGE_STORE_BACKEND, MESSAGES_DB_PATH, EXTRACT_LAST_K_MESSAGES)
logger.info("Memory history: path=%s", MEMORY_HISTORY_DB_PATH)

# --- Client plugin policy (served via GET /v1/config) ---
# Behavior policy for client plugins (claude-code etc.). The server only
# stores and serves these values; enforcement happens in each client's hooks.
INJECT_TIMING = os.environ.get("INJECT_TIMING", "every")  # off | first | every
MIN_QUERY_CHARS = int(os.environ.get("MIN_QUERY_CHARS", "5"))
# Memories produced by the current session and younger than this are excluded
# from automatic prompt injection (explicit search unaffected). 0 = disabled.
RECENT_MEMORY_DELAY_SECONDS = int(os.environ.get("RECENT_MEMORY_DELAY_SECONDS", "1800"))
# claude-code plugin uploads each turn's new messages as they happen (like
# hermes/openclaw) instead of only flushing at session boundaries. On by
# default (flipped 2026-10-09 after the rollout observation window); the
# plugin's local env kill-switch (NEATMEM_CODE_PER_TURN_FORWARD=0) overrides
# this, and setting PER_TURN_FORWARD=false here restores boundary-only flush.
PER_TURN_FORWARD = os.environ.get("PER_TURN_FORWARD", "true").strip().lower() in {
    "1", "true", "yes", "on",
}

# --- Query rewrite (MemOS fine-style rewrite + expansion; plan §5.1) ---
# Off by default during the production-observation rollout. Connection/model
# knobs fall back to the main LLM config at the call site (main.py owns
# LLM_MODEL/LLM_BASE_URL/LLM_API_KEY); thinking stays off by default to match
# the R3-validated semantics and the 3s latency budget.
QUERY_REWRITE_ENABLED = os.environ.get("QUERY_REWRITE_ENABLED", "false").strip().lower() in {
    "1", "true", "yes", "on",
}
QUERY_REWRITE_MODEL = os.environ.get("QUERY_REWRITE_MODEL", "")
QUERY_REWRITE_BASE_URL = os.environ.get("QUERY_REWRITE_BASE_URL", "")
QUERY_REWRITE_API_KEY = os.environ.get("QUERY_REWRITE_API_KEY", "")
QUERY_REWRITE_THINKING = os.environ.get("QUERY_REWRITE_THINKING", "false").strip().lower() in {
    "1", "true", "yes", "on",
}
QUERY_REWRITE_TIMEOUT = float(os.environ.get("QUERY_REWRITE_TIMEOUT", "3"))
QUERY_REWRITE_CONTEXT_TURNS = int(os.environ.get("QUERY_REWRITE_CONTEXT_TURNS", "3"))
QUERY_REWRITE_MAX_EXPANSIONS = int(os.environ.get("QUERY_REWRITE_MAX_EXPANSIONS", "3"))
# Transient-error retries per rewrite call (2026-10-03, R5 batch-1 lesson:
# single-attempt fail-open makes eval batches unmeasurable on a bad-upstream
# day — 29.8% fallback vs the >1% discard rule). Default 0 = production
# single-attempt semantics byte-identical (retries would break the 3s / 5s
# hot-path latency budget); eval harnesses set it explicitly. Deterministic
# failures (parse) are never retried.
QUERY_REWRITE_RETRIES = int(os.environ.get("QUERY_REWRITE_RETRIES", "0"))

# --- Memory usage feedback (plan: 20261005-citation-feedback §5.7) ---
# Injection-feedback loop: server records search/injection events, an offline
# judge batch scores whether injected memories were used, and a per-memory
# double counter (inject_count/used_count) drives an eviction gate
# (inject >= MIN_INJECTIONS and used == 0). Two switches on purpose:
# collection/judging (CAPTURE_ENABLED) and the eviction gate (EVICTION_ENABLED)
# are separate so weeks of observation can run before any behavior change.
# Both default off; off = byte-identical pre-existing behavior.
#
# Rename (plan 20261010): MEMORY_FEEDBACK_ENABLED -> ..._CAPTURE_ENABLED.
# The flag only captures events; judging is offline and eviction is a
# separate switch, so "capture" names what it actually does. The old name
# stays a deprecated alias for one release cycle.
_FEEDBACK_CAPTURE_NEW = "MEMORY_FEEDBACK_CAPTURE_ENABLED"
_FEEDBACK_CAPTURE_OLD = "MEMORY_FEEDBACK_ENABLED"
if _FEEDBACK_CAPTURE_NEW in os.environ:
    MEMORY_FEEDBACK_CAPTURE_ENABLED = os.environ.get(
        _FEEDBACK_CAPTURE_NEW, "false").strip().lower() in {"1", "true", "yes", "on"}
    if _FEEDBACK_CAPTURE_OLD in os.environ:
        _old_val = os.environ[_FEEDBACK_CAPTURE_OLD].strip().lower() in {"1", "true", "yes", "on"}
        if _old_val != MEMORY_FEEDBACK_CAPTURE_ENABLED:
            raise ValueError(
                f"{_FEEDBACK_CAPTURE_NEW} and deprecated {_FEEDBACK_CAPTURE_OLD} "
                f"disagree; set only {_FEEDBACK_CAPTURE_NEW}"
            )
elif _FEEDBACK_CAPTURE_OLD in os.environ:
    logger.warning(
        "%s is deprecated, use %s (alias removal planned next-next minor)",
        _FEEDBACK_CAPTURE_OLD, _FEEDBACK_CAPTURE_NEW,
    )
    MEMORY_FEEDBACK_CAPTURE_ENABLED = os.environ[_FEEDBACK_CAPTURE_OLD].strip().lower() in {
        "1", "true", "yes", "on"}
else:
    MEMORY_FEEDBACK_CAPTURE_ENABLED = False
# Judge model triple: empty = follow the main LLM config at the call site.
MEMORY_FEEDBACK_JUDGE_MODEL = os.environ.get("MEMORY_FEEDBACK_JUDGE_MODEL", "")
MEMORY_FEEDBACK_JUDGE_BASE_URL = os.environ.get("MEMORY_FEEDBACK_JUDGE_BASE_URL", "")
MEMORY_FEEDBACK_JUDGE_API_KEY = os.environ.get("MEMORY_FEEDBACK_JUDGE_API_KEY", "")
# Judge prompt override: file path via the standard prompt loader.
MEMORY_FEEDBACK_JUDGE_PROMPT = os.environ.get("MEMORY_FEEDBACK_JUDGE_PROMPT", "")
# Auto judge (plan 20261010 §3): optional in-serve background thread that runs
# the offline judge batch every INTERVAL_SECONDS — a built-in cron, not an
# online path change. Default off = manual/cron mode only. Boolean and
# interval stay separate knobs (message-batching precedent: enabled+interval).
MEMORY_FEEDBACK_JUDGE_AUTO_ENABLED = os.environ.get(
    "MEMORY_FEEDBACK_JUDGE_AUTO_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
MEMORY_FEEDBACK_JUDGE_INTERVAL_SECONDS = int(
    os.environ.get("MEMORY_FEEDBACK_JUDGE_INTERVAL_SECONDS", "3600"))
# Eviction gate (P2): demote (never delete) memories never used after N
# injections; explicit search still finds them, restore brings them back.
MEMORY_FEEDBACK_EVICTION_ENABLED = os.environ.get("MEMORY_FEEDBACK_EVICTION_ENABLED", "false").strip().lower() in {
    "1", "true", "yes", "on",
}
MEMORY_FEEDBACK_EVICTION_MIN_INJECTIONS = int(os.environ.get("MEMORY_FEEDBACK_EVICTION_MIN_INJECTIONS", "10"))
# Event store location follows the MESSAGES_DB_PATH convention (under
# NEATMEM_DIR); not a serve-flag knob, overridable for tests/ops only.
ACTIVITY_DB_PATH = os.environ.get(
    "ACTIVITY_DB_PATH",
    os.path.join(NEATMEM_DIR, "activity.db"),
)


def validate_feedback_config() -> None:
    """Startup contract for the feedback flags (rule 7: fail loud)."""
    if MEMORY_FEEDBACK_JUDGE_AUTO_ENABLED:
        if not MEMORY_FEEDBACK_CAPTURE_ENABLED:
            raise ValueError(
                "MEMORY_FEEDBACK_JUDGE_AUTO_ENABLED=true requires "
                "MEMORY_FEEDBACK_CAPTURE_ENABLED=true (no capture, nothing to judge)"
            )
        if MEMORY_FEEDBACK_JUDGE_INTERVAL_SECONDS <= 0:
            raise ValueError(
                "MEMORY_FEEDBACK_JUDGE_INTERVAL_SECONDS must be > 0 when "
                "MEMORY_FEEDBACK_JUDGE_AUTO_ENABLED=true"
            )


logger.info(
    "Memory feedback: capture=%s, auto_judge=%s (interval=%ss), eviction=%s "
    "(min_injections=%s), judge_model=%s",
    MEMORY_FEEDBACK_CAPTURE_ENABLED, MEMORY_FEEDBACK_JUDGE_AUTO_ENABLED,
    MEMORY_FEEDBACK_JUDGE_INTERVAL_SECONDS, MEMORY_FEEDBACK_EVICTION_ENABLED,
    MEMORY_FEEDBACK_EVICTION_MIN_INJECTIONS, MEMORY_FEEDBACK_JUDGE_MODEL or "<main llm>",
)

# --- Entity decoupling ---
ENTITY_EXTRACTOR_BACKEND = os.environ.get("ENTITY_EXTRACTOR_BACKEND", "ner")  # ner | llm
ENTITY_STORE_BACKEND = os.environ.get("ENTITY_STORE_BACKEND", "qdrant")  # qdrant

logger.info("Entity: extractor=%s, store=%s", ENTITY_EXTRACTOR_BACKEND, ENTITY_STORE_BACKEND)

# --- 图记忆配置（mem0 1.0.11 忠实复现）---
ENABLE_GRAPH = os.environ.get("ENABLE_GRAPH", "false").lower() == "true"
KUZU_DB_PATH = os.environ.get("KUZU_DB_PATH", "")
GRAPH_THRESHOLD = float(os.environ.get("GRAPH_THRESHOLD", "0.7"))
GRAPH_SEARCH_TOP_K = int(os.environ.get("GRAPH_SEARCH_TOP_K", "5"))
# Graph embedder defaults follow the main embedder config above; set
# GRAPH_EMBEDDER_* to override per-graph. DIMS stays a concrete 1024 because
# the graph schema needs a number (the main side auto-detects at boot).
GRAPH_EMBEDDER_MODEL = os.environ.get("GRAPH_EMBEDDER_MODEL", EMBEDDER_MODEL)
GRAPH_EMBEDDER_DIMS = int(os.environ.get("GRAPH_EMBEDDER_DIMS", "1024"))
GRAPH_EMBEDDER_BASE_URL = os.environ.get("GRAPH_EMBEDDER_BASE_URL", EMBEDDER_BASE_URL)
GRAPH_EMBEDDER_API_KEY = os.environ.get("GRAPH_EMBEDDER_API_KEY") or EMBEDDER_API_KEY

if ENABLE_GRAPH:
    logger.info("Graph memory: ENABLED (kuzu=%s, threshold=%s, top_k=%s, embed=%s/%s)",
                KUZU_DB_PATH or "(unset)", GRAPH_THRESHOLD, GRAPH_SEARCH_TOP_K,
                GRAPH_EMBEDDER_BASE_URL, GRAPH_EMBEDDER_MODEL)
else:
    logger.info("Graph memory: disabled (ENABLE_GRAPH=false)")
