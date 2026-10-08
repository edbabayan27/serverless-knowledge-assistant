"""Text embeddings with Amazon Titan Text Embeddings V2 on Amazon Bedrock.

Ingestion and queries must use the same model and dimensions, otherwise the vectors are not
comparable. Supported dimensions are validated in `config.py` at cold start.

Only the Titan V2 request and response format is implemented. Using another embedding model
(for example Cohere Embed v4) means adding its format here, not just changing the model ID.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Embedding:
    vector: list[float]
    input_tokens: int


class BedrockEmbedder:
    """Calls the embedding model through the Bedrock Runtime InvokeModel API.

    `client` is a boto3 "bedrock-runtime" client. Titan V2 embeds one text per request, so
    `embed_many` sends requests in parallel threads (boto3 clients are thread-safe).
    """

    def __init__(self, client: Any, *, model_id: str, dimensions: int, max_workers: int = 8):
        self._client = client
        self._model_id = model_id
        self._dimensions = dimensions
        self._max_workers = max_workers

    def embed(self, text: str) -> Embedding:
        if not text.strip():
            raise ValueError("Cannot embed empty text")
        response = self._client.invoke_model(
            modelId=self._model_id,
            contentType="application/json",
            accept="application/json",
            # normalize=True returns unit-length vectors (Titan's default, recommended for RAG).
            body=json.dumps({"inputText": text, "dimensions": self._dimensions, "normalize": True}),
        )
        payload = json.loads(response["body"].read())
        vector = payload["embedding"]
        if len(vector) != self._dimensions:
            raise ValueError(f"Expected {self._dimensions} dimensions, got {len(vector)}")
        return Embedding(vector=vector, input_tokens=payload.get("inputTextTokenCount", 0))

    def embed_many(self, texts: Sequence[str]) -> list[Embedding]:
        """Embed several texts in parallel. Results are in the same order as `texts`."""
        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = [pool.submit(self.embed, text) for text in texts]
            try:
                return [future.result() for future in futures]
            except Exception:
                # Fail fast: don't keep calling Bedrock for the rest of the batch.
                for future in futures:
                    future.cancel()
                raise
