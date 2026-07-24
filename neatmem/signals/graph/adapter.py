"""Graph memory LLM/embedding adapter.

This bridges NeatMem's OpenAI-compatible clients to the interface mem0 1.0.11's
MemoryGraph expects: ``llm.generate_response(messages, tools)`` and
``embedding_model.embed(text)``.

Per plan section 4.5 (纪律原则): the LLM invocation logic is COPIED from
mem0/llms/openai.py (OpenAILLM.generate_response + _parse_response) and the
embedding logic from mem0/embeddings/openai.py (OpenAIEmbedding.embed). We do
NOT invent message formatting, tool_choice handling, or tool_calls parsing —
those are copied verbatim so the LLM receives identical requests and we parse
identical responses, preserving algorithm equivalence (section 8 test).

Only the client objects are NeatMem-specific glue.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from .prompts import extract_json

logger = logging.getLogger(__name__)


class GraphAdapter:
    """Adapter exposing mem0's LLM/embedding interface on top of NeatMem clients.

    Args:
        llm_client: OpenAI-compatible client for LLM calls (NeatMem's openai_client).
        llm_model: Model name for LLM calls (e.g. "MiniMax-M3").
        embedder_client: OpenAI-compatible client for embedding calls.
        embedder_model: Embedding model name (e.g. siliconflow bge-m3 model id).
        embedding_dims: Embedding dimensionality (for the Kuzu FLOAT[] schema).
        temperature: LLM temperature (mem0's MemoryGraph extraction calls do not
            pass temperature explicitly, so this defaults to 0.0 for determinism.
            If the section-8 equivalence test diverges, revisit against mem0's
            BaseLlmConfig default).
    """

    def __init__(
        self,
        llm_client,
        llm_model: str,
        embedder_client,
        embedder_model: str,
        embedding_dims: int,
        temperature: float = 0.0,
    ):
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.embedder_client = embedder_client
        self.embedder_model = embedder_model
        self.embedding_dims = embedding_dims
        self.temperature = temperature
        # mem0's OpenAIEmbedding sets _pass_dimensions_to_api = (embedding_dims is not None).
        # But non-matryoshka backends (siliconflow bge-m3) reject the `dimensions` param.
        # NeatMem's existing vector-store path does not pass dimensions to siliconflow and
        # works at 0.88, so we mirror that: never pass dimensions to the embedding API.
        self._pass_dimensions_to_api = False

    # ------------------------------------------------------------------
    # LLM — copied from mem0/llms/openai.py OpenAILLM
    # ------------------------------------------------------------------
    def llm_generate(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Any] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        **kwargs,
    ):
        """Mirror of mem0 OpenAILLM.generate_response.

        Returns:
            If tools given: {"content": str, "tool_calls": [{"name": str, "arguments": dict}]}
            Otherwise: str (message content)
        """
        params: Dict[str, Any] = {
            "model": self.llm_model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if response_format:
            params["response_format"] = response_format
        if tools:
            params["tools"] = tools
            params["tool_choice"] = tool_choice
        # Pass through any extra OpenAI params the caller supplied.
        params.update(kwargs)

        response = self.llm_client.chat.completions.create(**params)
        return self._parse_response(response, tools)

    def _parse_response(self, response, tools):
        """Verbatim from mem0/llms/openai.py OpenAILLM._parse_response."""
        if tools:
            processed_response = {
                "content": response.choices[0].message.content,
                "tool_calls": [],
            }
            if response.choices[0].message.tool_calls:
                for tool_call in response.choices[0].message.tool_calls:
                    processed_response["tool_calls"].append(
                        {
                            "name": tool_call.function.name,
                            "arguments": json.loads(extract_json(tool_call.function.arguments)),
                        }
                    )
            return processed_response
        else:
            return response.choices[0].message.content

    # ------------------------------------------------------------------
    # Embedding — copied from mem0/embeddings/openai.py OpenAIEmbedding
    # ------------------------------------------------------------------
    def embed(self, text: str, memory_action: Optional[str] = None) -> List[float]:
        """Mirror of mem0 OpenAIEmbedding.embed.

        Args:
            text: The text to embed.
            memory_action: Unused (kept for API parity with mem0's signature).
        Returns:
            The embedding vector.
        """
        text = text.replace("\n", " ")
        kwargs = {
            "input": [text],
            "model": self.embedder_model,
            "encoding_format": "float",
        }
        if self._pass_dimensions_to_api:
            kwargs["dimensions"] = self.embedding_dims
        return self.embedder_client.embeddings.create(**kwargs).data[0].embedding
