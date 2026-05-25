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
            "path": "qdrant_db",
            "embedding_model_dims": 1024,
            "on_disk": False,
        }
    }
}

# LLM reranker：二分类过滤，踢掉无关记忆
LLM_RERANK = os.environ.get("LLM_RERANK", "true").lower() == "true"

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

logger.info("向量存储: Qdrant 本地模式 (path=qdrant_db, cosine similarity, BM25 enabled)")
