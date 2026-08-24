# The OpenAIEmbedder class in this file is ported from mem0
# (https://github.com/mem0ai/mem0), v2.0.0 (mem0/embeddings/openai.py and
# mem0/embeddings/langchain.py).
# Copyright (c) Mem0
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Modifications:
#   - Config object replaced by plain constructor kwargs.
#   - Optional startup dimension self-check added (expected_dims). This is a
#     NeatMem robustness contract: a misconfigured embedding backend must
#     fail loudly at boot instead of silently degrading search quality.
#   - embed_batch kept for API completeness (used by future batch paths).

import logging
import os
from typing import Literal, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)

MemoryAction = Optional[Literal["add", "search", "update"]]


class OpenAIEmbedder:
    """OpenAI-compatible embedding client with mem0's embed() signature."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        embedding_dims: Optional[int] = None,
        expected_dims: Optional[int] = None,
        batch_size: int = 100,
    ):
        """
        Args:
            model: Embedding model name (e.g. "BAAI/bge-m3").
            api_key: API key. Falls back to OPENAI_API_KEY env var.
            base_url: OpenAI-compatible base URL. Falls back to
                OPENAI_API_BASE / OPENAI_BASE_URL env vars, then api.openai.com.
            embedding_dims: If set, passed to the API as `dimensions`
                (matryoshka models only; non-matryoshka backends reject it).
            expected_dims: If set, a probe embedding is issued at init and its
                dimension must equal this value, otherwise init raises.
            batch_size: Max texts per embeddings API call in embed_batch.
                Provider-specific (DashScope hard-errors above 10); default
                100 preserves the previous fixed behavior.
        """
        self.model = model or "text-embedding-3-small"
        self.embedding_dims = embedding_dims
        self.batch_size = batch_size

        api_key = api_key or os.getenv("OPENAI_API_KEY")
        base_url = (
            base_url
            or os.getenv("OPENAI_API_BASE")
            or os.getenv("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        )
        self.client = OpenAI(api_key=api_key, base_url=base_url)

        if expected_dims is not None:
            probe = self.embed("dimension self-check", "add")
            if len(probe) != expected_dims:
                raise ValueError(
                    f"Embedding dimension mismatch: model {self.model!r} returned "
                    f"{len(probe)} dims, expected {expected_dims}. Check "
                    f"EMBEDDER_MODEL / embedding_model_dims configuration."
                )
            logger.info("Embedding dimension self-check passed (%d dims)", expected_dims)

    def embed(self, text, memory_action: MemoryAction = None):
        """
        Get the embedding for the given text.

        Args:
            text (str): The text to embed.
            memory_action (optional): The type of memory operation
                ("add", "search", or "update"). Accepted for mem0 API
                compatibility; does not change the request.
        Returns:
            list: The embedding vector.
        """
        text = text.replace("\n", " ")
        kwargs = {
            "input": [text],
            "model": self.model,
            "encoding_format": "float",
        }
        if self.embedding_dims is not None:
            kwargs["dimensions"] = self.embedding_dims
        return self.client.embeddings.create(**kwargs).data[0].embedding

    def embed_batch(self, texts, memory_action: MemoryAction = "add"):
        """Embed multiple texts in a single OpenAI API call.

        Chunks into batches of self.batch_size (default 100) to stay within
        API limits.
        """
        texts = [text.replace("\n", " ") for text in texts]
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            chunk = texts[i : i + self.batch_size]
            kwargs = {
                "input": chunk,
                "model": self.model,
                "encoding_format": "float",
            }
            if self.embedding_dims is not None:
                kwargs["dimensions"] = self.embedding_dims
            response = self.client.embeddings.create(**kwargs)
            all_embeddings.extend(item.embedding for item in sorted(response.data, key=lambda x: x.index))
        return all_embeddings


class LangchainEmbedder:
    """Adapter giving a LangChain Embeddings instance mem0's embed() signature."""

    def __init__(self, model):
        """
        Args:
            model: A langchain Embeddings instance (e.g. XinferenceEmbeddings).
        """
        if model is None:
            raise ValueError("`model` parameter is required")
        self.langchain_model = model

    def embed(self, text, memory_action: MemoryAction = None):
        """
        Get the embedding for the given text using Langchain.

        Args:
            text (str): The text to embed.
            memory_action (optional): Accepted for mem0 API compatibility;
                does not change the request.
        Returns:
            list: The embedding vector.
        """
        return self.langchain_model.embed_query(text)
