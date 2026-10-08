import io

import pytest

from knowledge_assistant.embeddings import BedrockEmbedder

MODEL_ID = "amazon.titan-embed-text-v2:0"


def test_embed_sends_a_titan_v2_request(bedrock):
    embedder = BedrockEmbedder(bedrock, model_id=MODEL_ID, dimensions=256)

    embedding = embedder.embed("How should I handle Lambda cold starts?")

    [call] = bedrock.invocations
    assert call["modelId"] == MODEL_ID
    assert call["body"] == {
        "inputText": "How should I handle Lambda cold starts?",
        "dimensions": 256,
        "normalize": True,
    }
    assert len(embedding.vector) == 256
    assert embedding.input_tokens == 7  # the fake counts words


def test_embed_many_keeps_the_input_order(bedrock):
    embedder = BedrockEmbedder(bedrock, model_id=MODEL_ID, dimensions=256, max_workers=4)
    texts = [f"text number {i}" for i in range(20)]

    embeddings = embedder.embed_many(texts)

    assert [e.vector for e in embeddings] == [embedder.embed(text).vector for text in texts]


def test_empty_text_is_rejected_without_calling_bedrock(bedrock):
    embedder = BedrockEmbedder(bedrock, model_id=MODEL_ID, dimensions=256)

    with pytest.raises(ValueError, match="empty"):
        embedder.embed("   ")
    assert bedrock.invocations == []


def test_a_vector_of_the_wrong_size_is_an_error(bedrock, monkeypatch):
    def short_vector(**kwargs):
        return {"body": io.BytesIO(b'{"embedding": [0.1, 0.2]}')}

    monkeypatch.setattr(bedrock, "invoke_model", short_vector)
    embedder = BedrockEmbedder(bedrock, model_id=MODEL_ID, dimensions=256)

    with pytest.raises(ValueError, match="Expected 256 dimensions, got 2"):
        embedder.embed("text")


def test_embed_many_raises_the_first_error(bedrock, monkeypatch):
    def fail(**kwargs):
        raise RuntimeError("ThrottlingException")

    monkeypatch.setattr(bedrock, "invoke_model", fail)
    embedder = BedrockEmbedder(bedrock, model_id=MODEL_ID, dimensions=256)

    with pytest.raises(RuntimeError, match="ThrottlingException"):
        embedder.embed_many(["a", "b", "c"])
