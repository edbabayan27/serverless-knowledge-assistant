"""Shared test helpers: in-memory fakes for the boto3 clients the code calls.

The fakes record every call so tests can check the exact requests sent to AWS.
"""

import io
import json
import zlib

import pytest


class FakeBedrockRuntime:
    """Stands in for a boto3 "bedrock-runtime" client."""

    def __init__(self):
        self.invocations: list[dict] = []
        self.converse_calls: list[dict] = []
        self.converse_response: dict = {
            "output": {"message": {"role": "assistant", "content": [{"text": "An answer."}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 120, "outputTokens": 30, "totalTokens": 150},
        }

    def invoke_model(self, **kwargs):
        body = json.loads(kwargs["body"])
        self.invocations.append({**kwargs, "body": body})
        # A deterministic vector derived from the text, so different texts get different vectors.
        seed = zlib.crc32(body["inputText"].encode())
        vector = [float((seed >> (i % 24)) % 101) for i in range(body["dimensions"])]
        payload = {"embedding": vector, "inputTextTokenCount": len(body["inputText"].split())}
        return {"body": io.BytesIO(json.dumps(payload).encode())}

    def converse(self, **kwargs):
        self.converse_calls.append(kwargs)
        return self.converse_response


class FakeS3Vectors:
    """Stands in for a boto3 "s3vectors" client, backed by a dict of stored vectors."""

    def __init__(self):
        self.vectors: dict[str, dict] = {}
        self.calls: list[tuple[str, dict]] = []
        self.query_response: dict = {"vectors": [], "distanceMetric": "cosine"}

    def put_vectors(self, **kwargs):
        self.calls.append(("put_vectors", kwargs))
        for vector in kwargs["vectors"]:
            self.vectors[vector["key"]] = vector
        return {}

    def query_vectors(self, **kwargs):
        self.calls.append(("query_vectors", kwargs))
        return self.query_response

    def list_vectors(self, **kwargs):
        self.calls.append(("list_vectors", kwargs))
        keys = sorted(self.vectors)
        start = int(kwargs.get("nextToken", 0))
        end = start + kwargs["maxResults"]
        response = {"vectors": [{"key": key} for key in keys[start:end]]}
        if end < len(keys):
            response["nextToken"] = str(end)
        return response

    def delete_vectors(self, **kwargs):
        self.calls.append(("delete_vectors", kwargs))
        for key in kwargs["keys"]:
            self.vectors.pop(key, None)
        return {}

    def calls_to(self, operation: str) -> list[dict]:
        return [kwargs for name, kwargs in self.calls if name == operation]


@pytest.fixture
def bedrock():
    return FakeBedrockRuntime()


@pytest.fixture
def s3vectors():
    return FakeS3Vectors()
