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
EMBEDDING_PROVIDER = os.environ.get("EMBEDDING_PROVIDER", "siliconflow")  # siliconflow / xinference

if EMBEDDING_PROVIDER == "siliconflow":
    embedding_config = {
        "provider": "openai",
        "embedding_model_dims": 1024,
        "config": {
            "model": "BAAI/bge-m3",
            "openai_base_url": "https://api.siliconflow.cn/v1",
            "api_key": os.environ.get("SILICONFLOW_API_KEY", ""),
        }
    }
else:
    # 本地 Xinference Embedding (备用)
    embedding_model = XinferenceEmbeddings(
        server_url=os.environ.get("XINFERENCE_SERVER_URL", "http://localhost:9997"),
        model_uid=os.environ.get("XINFERENCE_MODEL_UID", "bge-m3")
    )
    embedding_config = {
        "provider": "langchain",
        "embedding_model_dims": 1024,
        "config": {
            "model": embedding_model,
        }
    }

# --- 多信号开关（默认全开，A/B 测试时用环境变量切换）---
QDRANT_HOST = os.environ.get("QDRANT_HOST", "")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
QDRANT_PATH = os.environ.get("QDRANT_PATH", "qdrant_db")
ENABLE_BM25 = os.environ.get("ENABLE_BM25", "true").lower() == "true"
ENABLE_ENTITY = os.environ.get("ENABLE_ENTITY", "false").lower() == "true"

# --- mem0 配置 ---
config = {
    "llm": {
        "provider": "openai",
        "config": {
            "model": os.environ.get("LLM_MODEL", "qwen-max-latest"),
            "temperature": 0,
        }
    },
    "embedder": embedding_config,
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "mem0",
            **({"host": QDRANT_HOST, "port": QDRANT_PORT} if QDRANT_HOST else {"path": QDRANT_PATH}),
            "embedding_model_dims": 1024,
            "on_disk": False,
        }
    }
}

# LLM reranker：二分类过滤，踢掉无关记忆
LLM_RERANK = os.environ.get("LLM_RERANK", "true").lower() == "true"

# --- Dedup 主参数 ---
# DEDUP_MODE 控制去重行为，单参数：
#   off      - 不去重，全写入
#   skip     - listwise，update 降级 add（新旧共存，默认）
#   replace  - listwise，update 时 new 直接覆盖 old
#   rewrite  - listwise，update 时 LLM 重写融合
#   edit     - listwise，update 时 LLM 生成 patch（F2 prompt）
DEDUP_MODE = os.environ.get("DEDUP_MODE", "skip")

# 内部映射：DEDUP_MODE -> (enable_dedup, dedup_strategy, merge_strategy)
if DEDUP_MODE == "off":
    ENABLE_DEDUP = False
    DEDUP_STRATEGY = None
    MERGE_STRATEGY = None
elif DEDUP_MODE == "skip":
    ENABLE_DEDUP = True
    DEDUP_STRATEGY = "skip"        # update 降级 add
    MERGE_STRATEGY = None          # 不调 merge LLM
elif DEDUP_MODE == "replace":
    ENABLE_DEDUP = True
    DEDUP_STRATEGY = "update"
    MERGE_STRATEGY = "replace"     # new 覆盖 old
elif DEDUP_MODE == "rewrite":
    ENABLE_DEDUP = True
    DEDUP_STRATEGY = "update"
    MERGE_STRATEGY = "rewrite"     # LLM 重写
elif DEDUP_MODE == "edit":
    ENABLE_DEDUP = True
    DEDUP_STRATEGY = "update"
    MERGE_STRATEGY = "patch_diff_forward"  # LLM 生成 patch（F2 prompt）
else:
    raise ValueError(f"Invalid DEDUP_MODE={DEDUP_MODE!r}, expected off|skip|replace|rewrite|edit")

# --- Dedup Advanced 参数 ---
# dedup prompt 版本：v7（默认，信息点检查）/ event-check（legacy，已失败）
DEDUP_PROMPT_VERSION = os.environ.get("DEDUP_PROMPT_VERSION", "v7")
# dedup LLM thinking 开关
DEDUP_THINKING = os.environ.get("DEDUP_THINKING", "false").lower() == "true"

# --- Patch_diff Advanced 参数（仅 DEDUP_MODE=edit 时生效）---
# patch_diff prompt 版本：f2（默认，有 relationship）/ f2_norel（legacy，已失败）
PATCH_DIFF_PROMPT_VERSION = os.environ.get("PATCH_DIFF_PROMPT_VERSION", "f2")
# patch_diff LLM thinking 开关
EDIT_THINKING = os.environ.get("EDIT_THINKING", "false").lower() == "true"
# none 时也走 patch_diff（legacy，已失败，默认 false）
NONE_PATCH_DIFF = os.environ.get("NONE_PATCH_DIFF", "false").lower() == "true"

# 可选 reranker：设置 RERANKER_MODEL_PATH 环境变量启用
reranker_model_path = os.environ.get("RERANKER_MODEL_PATH")
if reranker_model_path:
    config["reranker"] = {
        "provider": "sentence_transformer",
        "config": {
            "model": reranker_model_path,
            "device": os.environ.get("RERANKER_DEVICE", "cpu"),
            "batch_size": int(os.environ.get("RERANKER_BATCH_SIZE", "32")),
            "show_progress_bar": True,
            "top_k": int(os.environ.get("RERANKER_TOP_K", "5"))
        }
    }
    logger.info("Reranker 已启用: %s", reranker_model_path)
else:
    logger.info("Reranker 未启用 (纯向量+BM25搜索模式)")

logger.info("向量存储: Qdrant %s (BM25=%s, Entity=%s)",
             f"server ({QDRANT_HOST}:{QDRANT_PORT})" if QDRANT_HOST else f"本地模式 (path={QDRANT_PATH})",
             ENABLE_BM25, ENABLE_ENTITY)
logger.info("Dedup: DEDUP_MODE=%s, enable=%s, strategy=%s, merge=%s",
            DEDUP_MODE, ENABLE_DEDUP, DEDUP_STRATEGY, MERGE_STRATEGY)
logger.info("Dedup prompt: %s (thinking=%s), patch_diff prompt: %s (thinking=%s, none_patch_diff=%s)",
            DEDUP_PROMPT_VERSION, DEDUP_THINKING, PATCH_DIFF_PROMPT_VERSION, EDIT_THINKING, NONE_PATCH_DIFF)

# --- 消息历史存储配置 ---
HISTORY_DB_PATH = os.environ.get(
    "HISTORY_DB_PATH",
    os.path.join(QDRANT_PATH, "history.db"),
)
EXTRACT_LAST_K_MESSAGES = int(os.environ.get("EXTRACT_LAST_K_MESSAGES", "10"))
MESSAGE_STORE_BACKEND = os.environ.get("MESSAGE_STORE_BACKEND", "sqlite")  # sqlite / none

logger.info("消息历史: backend=%s, path=%s (extract_last_k=%s)",
            MESSAGE_STORE_BACKEND, HISTORY_DB_PATH, EXTRACT_LAST_K_MESSAGES)

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
