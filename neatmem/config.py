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


# --- Embedding 配置 ---
EMBEDDING_PROVIDER = os.environ.get("EMBEDDING_PROVIDER", "siliconflow")  # siliconflow / openai / dashscope / xinference
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")
# Per-provider presets (batch limits verified in smoke: DashScope hard-errors
# above 10). Explicit EMBEDDING_BASE_URL always wins over the preset.
EMBEDDING_PROVIDER_PRESETS = {
    "siliconflow": {"base_url": "https://api.siliconflow.cn/v1", "batch_size": 100},
    "openai": {"base_url": "https://api.openai.com/v1", "batch_size": 100},
    "dashscope": {"base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "batch_size": 10},
}
_EMB_PRESET = EMBEDDING_PROVIDER_PRESETS.get(EMBEDDING_PROVIDER, {})
EMBEDDING_BASE_URL = os.environ.get(
    "EMBEDDING_BASE_URL",
    _EMB_PRESET.get("base_url", "https://api.siliconflow.cn/v1"),
)
EMBEDDING_BATCH_SIZE = _EMB_PRESET.get("batch_size", 100)
# Generic key name preferred; SILICONFLOW_API_KEY kept as fallback (legacy envs).
EMBEDDING_API_KEY = os.environ.get("EMBEDDING_API_KEY") or os.environ.get("SILICONFLOW_API_KEY", "")
# Explicit dimension override. When unset, the dimension is auto-detected
# from the startup probe embedding (see build_memory_store).
EMBEDDING_DIMS = int(os.environ["EMBEDDING_DIMS"]) if os.environ.get("EMBEDDING_DIMS") else None

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
# Distinct from HISTORY_DB_PATH (chat message store).
MEMORY_HISTORY_DB_PATH = os.environ.get(
    "MEMORY_HISTORY_DB_PATH",
    os.path.join(NEATMEM_DIR, "history.db"),
)


def build_memory_store():
    """Construct the mem0-compatible MemoryStore from env configuration.

    Wires the self-managed parts: OpenAI-compatible embedder + Qdrant vector
    store + SQLite history. Boot-time contract (fail loudly, no silent
    degradation): a probe embedding is always issued at startup -- when
    EMBEDDING_DIMS is set its dimension must match, otherwise the probed
    dimension is auto-detected and used for the collection.
    """
    from neatmem.embeddings import LangchainEmbedder, OpenAIEmbedder
    from neatmem.memory_store import MemoryStore
    from neatmem.storage.vector.factory import create_vector_store

    try:
        if EMBEDDING_PROVIDER in EMBEDDING_PROVIDER_PRESETS:
            embedding_model = OpenAIEmbedder(
                model=EMBEDDING_MODEL,
                api_key=EMBEDDING_API_KEY,
                base_url=EMBEDDING_BASE_URL,
                expected_dims=EMBEDDING_DIMS,
                batch_size=EMBEDDING_BATCH_SIZE,
            )
        elif EMBEDDING_PROVIDER == "xinference":
            # 本地 Xinference Embedding (备用)
            embedding_model = LangchainEmbedder(XinferenceEmbeddings(
                server_url=os.environ.get("XINFERENCE_SERVER_URL", "http://localhost:9997"),
                model_uid=os.environ.get("XINFERENCE_MODEL_UID", "bge-m3")
            ))
        else:
            raise ValueError(
                f"Unknown EMBEDDING_PROVIDER={EMBEDDING_PROVIDER!r}, expected one of: "
                f"{', '.join(sorted(EMBEDDING_PROVIDER_PRESETS))}, xinference"
            )
        embedding_dims = EMBEDDING_DIMS
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
            "  - SILICONFLOW_API_KEY / EMBEDDING_BASE_URL (SiliconFlow)\n"
            "  - OPENAI_API_KEY / OPENAI_BASE_URL (OpenAI-compatible)"
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
#   skip     - update 降级 add（新旧共存，默认）
#   replace  - new 直接覆盖 old
#   rewrite  - LLM 融合（pointwise 下固定走 memos resolver）
#   edit     - LLM 生成 patch（F2 prompt）
# DEDUP_DETECTOR - 怎么判：
#   listwise - 1 次 LLM 调用判全批候选（默认）
#   pointwise - MemOS 三分类逐对判定（contradictory/redundant/independent）
DEDUP_ENABLED = os.environ.get("DEDUP_ENABLED", "true").lower() == "true"
DEDUP_RESOLVER = os.environ.get("DEDUP_RESOLVER", "skip")
DEDUP_DETECTOR = os.environ.get("DEDUP_DETECTOR", "listwise")
if DEDUP_RESOLVER not in ("skip", "replace", "rewrite", "edit"):
    raise ValueError(f"Invalid DEDUP_RESOLVER={DEDUP_RESOLVER!r}, expected skip|replace|rewrite|edit")
if DEDUP_DETECTOR not in ("listwise", "pointwise"):
    raise ValueError(f"Invalid DEDUP_DETECTOR={DEDUP_DETECTOR!r}, expected listwise|pointwise")

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

logger.info("向量存储: Qdrant %s (BM25=%s, Entity=%s)",
             f"server ({QDRANT_HOST}:{QDRANT_PORT})" if QDRANT_HOST else f"本地模式 (path={QDRANT_PATH})",
             ENABLE_BM25, ENABLE_ENTITY)
logger.info("Dedup: enabled=%s, resolver=%s, detector=%s, recall_threshold=%.2f",
            DEDUP_ENABLED, DEDUP_RESOLVER, DEDUP_DETECTOR, DEDUP_RECALL_THRESHOLD)
logger.info("Dedup thinking=%s, edit thinking=%s", DEDUP_THINKING, EDIT_THINKING)

# --- 消息历史存储配置 ---
HISTORY_DB_PATH = os.environ.get(
    "HISTORY_DB_PATH",
    os.path.join(NEATMEM_DIR, "messages.db"),
)
# sqlite cannot create missing parent directories; make sure the data root
# (and any custom db parents) exist before stores open their files.
os.makedirs(NEATMEM_DIR, exist_ok=True)
for _db in (HISTORY_DB_PATH, MEMORY_HISTORY_DB_PATH):
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

logger.info("消息批处理: enabled=%s, interval=%ss, batch_size=%s, deadline=%ss",
            MESSAGE_BATCHING_ENABLED, MESSAGE_BATCHING_CHECK_INTERVAL_SECS,
            MESSAGE_BATCH_SIZE, MESSAGE_BATCH_DEADLINE_SECS)

logger.info("消息历史: backend=%s, path=%s (extract_last_k=%s)",
            MESSAGE_STORE_BACKEND, HISTORY_DB_PATH, EXTRACT_LAST_K_MESSAGES)
logger.info("记忆变更历史: path=%s", MEMORY_HISTORY_DB_PATH)

# --- Entity decoupling ---
ENTITY_EXTRACTOR_BACKEND = os.environ.get("ENTITY_EXTRACTOR_BACKEND", "ner")  # ner | llm
ENTITY_STORE_BACKEND = os.environ.get("ENTITY_STORE_BACKEND", "qdrant")  # qdrant

logger.info("Entity: extractor=%s, store=%s", ENTITY_EXTRACTOR_BACKEND, ENTITY_STORE_BACKEND)

# --- 图记忆配置（mem0 1.0.11 忠实复现）---
ENABLE_GRAPH = os.environ.get("ENABLE_GRAPH", "false").lower() == "true"
KUZU_DB_PATH = os.environ.get("KUZU_DB_PATH", "")
GRAPH_THRESHOLD = float(os.environ.get("GRAPH_THRESHOLD", "0.7"))
GRAPH_SEARCH_TOP_K = int(os.environ.get("GRAPH_SEARCH_TOP_K", "5"))
# 图记忆用的 embedding 与 vector store 同源（siliconflow bge-m3, 1024 维）
GRAPH_EMBEDDING_MODEL = os.environ.get("GRAPH_EMBEDDING_MODEL", "BAAI/bge-m3")
GRAPH_EMBEDDING_DIMS = int(os.environ.get("GRAPH_EMBEDDING_DIMS", "1024"))
GRAPH_EMBEDDING_BASE_URL = os.environ.get("GRAPH_EMBEDDING_BASE_URL", "https://api.siliconflow.cn/v1")
GRAPH_EMBEDDING_API_KEY = os.environ.get("SILICONFLOW_API_KEY", "")

if ENABLE_GRAPH:
    logger.info("图记忆: ENABLED (kuzu=%s, threshold=%s, top_k=%s, embed=%s/%s)",
                KUZU_DB_PATH or "(unset)", GRAPH_THRESHOLD, GRAPH_SEARCH_TOP_K,
                GRAPH_EMBEDDING_BASE_URL, GRAPH_EMBEDDING_MODEL)
else:
    logger.info("图记忆: disabled (ENABLE_GRAPH=false)")
