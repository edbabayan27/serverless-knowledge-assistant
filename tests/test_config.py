import inspect
import re

import pytest

from knowledge_assistant import config
from knowledge_assistant.config import (
    DEFAULT_EMBEDDING_DIMENSIONS,
    DEFAULT_EMBEDDING_MODEL_ID,
    ENV_VARS,
    ConfigError,
    IngestSettings,
    QuerySettings,
)

BASE_ENV = {
    "VECTOR_BUCKET_NAME": "ska-vectors",
    "VECTOR_INDEX_NAME": "lens",
}
INGEST_ENV = {
    **BASE_ENV,
    "DOCUMENTS_BUCKET_NAME": "ska-documents",
    "DOCUMENT_URL": "https://example.com/lens.pdf",
    "DOCUMENT_ID": "serverless-lens-it",
}
QUERY_ENV = {
    **BASE_ENV,
    "GENERATION_MODEL_ID": "eu.amazon.nova-lite-v1:0",
}


@pytest.fixture
def env(monkeypatch):
    """Start every test from an environment without any of our variables set."""
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    def apply(values: dict[str, str]) -> None:
        for name, value in values.items():
            monkeypatch.setenv(name, value)

    return apply


def test_query_settings_use_defaults(env):
    env(QUERY_ENV)

    settings = QuerySettings.from_env()

    assert settings.vector_index.bucket_name == "ska-vectors"
    assert settings.vector_index.index_name == "lens"
    assert settings.embedding.model_id == DEFAULT_EMBEDDING_MODEL_ID
    assert settings.embedding.dimensions == DEFAULT_EMBEDDING_DIMENSIONS
    assert settings.generation_model_id == "eu.amazon.nova-lite-v1:0"
    assert settings.top_k == 5


def test_ingest_settings_read_overrides(env):
    env({**INGEST_ENV, "CHUNK_SIZE_WORDS": "300", "CHUNK_OVERLAP_WORDS": "0"})

    settings = IngestSettings.from_env()

    assert settings.document_id == "serverless-lens-it"
    assert settings.chunk_size_words == 300
    assert settings.chunk_overlap_words == 0


def test_missing_required_variable_names_the_variable(env):
    env({k: v for k, v in QUERY_ENV.items() if k != "GENERATION_MODEL_ID"})

    with pytest.raises(ConfigError, match="GENERATION_MODEL_ID"):
        QuerySettings.from_env()


def test_non_integer_value_is_rejected(env):
    env({**QUERY_ENV, "TOP_K": "five"})

    with pytest.raises(ConfigError, match="TOP_K must be an integer"):
        QuerySettings.from_env()


def test_overlap_must_be_smaller_than_chunk_size(env):
    env({**INGEST_ENV, "CHUNK_SIZE_WORDS": "50", "CHUNK_OVERLAP_WORDS": "50"})

    with pytest.raises(ConfigError, match="smaller than CHUNK_SIZE_WORDS"):
        IngestSettings.from_env()


def test_env_vars_lists_every_variable_config_reads():
    source = inspect.getsource(config)
    read = set(re.findall(r'_(?:required|optional|int)\("([A-Z_]+)"', source))

    assert read == set(ENV_VARS)


@pytest.mark.parametrize("dimensions", ["256", "512", "1024"])
def test_titan_v2_accepts_supported_dimensions(env, dimensions):
    env({**QUERY_ENV, "EMBEDDING_DIMENSIONS": dimensions})

    assert QuerySettings.from_env().embedding.dimensions == int(dimensions)


def test_titan_v2_rejects_unsupported_dimensions(env):
    env({**QUERY_ENV, "EMBEDDING_DIMENSIONS": "768"})

    with pytest.raises(ConfigError, match="EMBEDDING_DIMENSIONS must be one of 256, 512, 1024"):
        QuerySettings.from_env()


def test_other_embedding_models_skip_the_titan_dimension_check(env):
    env({**QUERY_ENV, "EMBEDDING_MODEL_ID": "cohere.embed-v4:0", "EMBEDDING_DIMENSIONS": "768"})

    assert QuerySettings.from_env().embedding.dimensions == 768


def test_document_url_must_use_https(env):
    env({**INGEST_ENV, "DOCUMENT_URL": "http://example.com/lens.pdf"})

    with pytest.raises(ConfigError, match="DOCUMENT_URL must start with https://"):
        IngestSettings.from_env()
