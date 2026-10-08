"""Runtime settings, read from environment variables that Terraform sets on each Lambda function.

This module is the contract between the infrastructure and the code: every variable a function
needs is listed here. Settings are validated once at cold start, so a misconfigured deployment
fails immediately with a clear message instead of halfway through a request.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
DEFAULT_EMBEDDING_DIMENSIONS = 1024
# Titan Text Embeddings v2 only supports these output sizes; anything else fails at the first call.
TITAN_V2_DIMENSIONS = frozenset({256, 512, 1024})

# Every environment variable this module reads. Tests clear these so the shell can't leak in.
ENV_VARS = (
    "VECTOR_BUCKET_NAME",
    "VECTOR_INDEX_NAME",
    "EMBEDDING_MODEL_ID",
    "EMBEDDING_DIMENSIONS",
    "DOCUMENTS_BUCKET_NAME",
    "DOCUMENT_URL",
    "DOCUMENT_ID",
    "CHUNK_SIZE_WORDS",
    "CHUNK_OVERLAP_WORDS",
    "GENERATION_MODEL_ID",
    "TOP_K",
    "MAX_ANSWER_TOKENS",
)


class ConfigError(RuntimeError):
    """A required environment variable is missing or has an invalid value."""


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ConfigError(f"Missing required environment variable: {name}")
    return value


def _optional(name: str, default: str) -> str:
    return os.environ.get(name, "").strip() or default


def _int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}, got {value}")
    return value


@dataclass(frozen=True)
class VectorIndexSettings:
    """Where the chunk embeddings are stored (an S3 Vectors bucket and index)."""

    bucket_name: str
    index_name: str

    @classmethod
    def from_env(cls) -> VectorIndexSettings:
        return cls(
            bucket_name=_required("VECTOR_BUCKET_NAME"),
            index_name=_required("VECTOR_INDEX_NAME"),
        )


@dataclass(frozen=True)
class EmbeddingSettings:
    """Bedrock embedding model. Ingestion and queries must use the same model and dimensions."""

    model_id: str
    dimensions: int

    @classmethod
    def from_env(cls) -> EmbeddingSettings:
        settings = cls(
            model_id=_optional("EMBEDDING_MODEL_ID", DEFAULT_EMBEDDING_MODEL_ID),
            dimensions=_int("EMBEDDING_DIMENSIONS", DEFAULT_EMBEDDING_DIMENSIONS),
        )
        # Only the default model is checked; other models have their own limits.
        if (
            settings.model_id == DEFAULT_EMBEDDING_MODEL_ID
            and settings.dimensions not in TITAN_V2_DIMENSIONS
        ):
            allowed = ", ".join(str(d) for d in sorted(TITAN_V2_DIMENSIONS))
            raise ConfigError(
                f"EMBEDDING_DIMENSIONS must be one of {allowed} for {settings.model_id}, "
                f"got {settings.dimensions}"
            )
        return settings


@dataclass(frozen=True)
class IngestSettings:
    """Settings for the ingest function: download the PDF, chunk it, embed it, store vectors."""

    vector_index: VectorIndexSettings
    embedding: EmbeddingSettings
    documents_bucket: str
    document_url: str
    document_id: str
    chunk_size_words: int
    chunk_overlap_words: int

    @classmethod
    def from_env(cls) -> IngestSettings:
        settings = cls(
            vector_index=VectorIndexSettings.from_env(),
            embedding=EmbeddingSettings.from_env(),
            documents_bucket=_required("DOCUMENTS_BUCKET_NAME"),
            document_url=_required("DOCUMENT_URL"),
            document_id=_required("DOCUMENT_ID"),
            chunk_size_words=_int("CHUNK_SIZE_WORDS", 200),
            chunk_overlap_words=_int("CHUNK_OVERLAP_WORDS", 40, minimum=0),
        )
        if not settings.document_url.startswith("https://"):
            raise ConfigError(
                f"DOCUMENT_URL must start with https://, got {settings.document_url!r}"
            )
        if settings.chunk_overlap_words >= settings.chunk_size_words:
            raise ConfigError("CHUNK_OVERLAP_WORDS must be smaller than CHUNK_SIZE_WORDS")
        return settings


@dataclass(frozen=True)
class QuerySettings:
    """Settings for the query function: retrieve relevant chunks and generate an answer."""

    vector_index: VectorIndexSettings
    embedding: EmbeddingSettings
    generation_model_id: str
    top_k: int
    max_answer_tokens: int

    @classmethod
    def from_env(cls) -> QuerySettings:
        return cls(
            vector_index=VectorIndexSettings.from_env(),
            embedding=EmbeddingSettings.from_env(),
            generation_model_id=_required("GENERATION_MODEL_ID"),
            top_k=_int("TOP_K", 5),
            max_answer_tokens=_int("MAX_ANSWER_TOKENS", 800),
        )
