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
ENABLE_ENTITY = os.environ.get("ENABLE_ENTITY", "true").lower() == "true"

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

# 合并策略：rewrite | patch_diff | off
MERGE_STRATEGY = os.environ.get("MERGE_STRATEGY", "off")

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
