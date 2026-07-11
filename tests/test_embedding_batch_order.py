"""
Regression test for the batch embedding ordering bug: when a batch mixes
cache hits and misses, returned embeddings must stay aligned with their
input texts (hits used to be appended before misses, misaligning results).
"""

import numpy as np
import pytest

from services import embedding_service as es

VEC_A = np.array([1.0, 0.0], dtype=np.float32)
VEC_B = np.array([0.0, 1.0], dtype=np.float32)


@pytest.fixture(autouse=True)
def clean_cache():
    es.clear_embedding_cache()
    yield
    es.clear_embedding_cache()


@pytest.fixture
def remote_backend_returning_b(monkeypatch):
    monkeypatch.setattr(es, "_should_use_remote_embeddings", lambda: True)
    monkeypatch.setattr(
        es, "_generate_remote_embeddings",
        lambda texts, normalize=True: [VEC_B for _ in texts],
    )


def test_mixed_hits_and_misses_preserve_input_order(remote_backend_returning_b):
    # "A" is a cache hit, "B" is a miss encoded via the (stubbed) backend.
    # With the miss FIRST in the batch, the old append-based code returned
    # [A(hit), B(encoded)] — swapped relative to the input ["B", "A"].
    es._cache_embedding("A", VEC_A)

    embeddings, _ = es.generate_embeddings_batch(["B", "A"], normalize=False)

    assert embeddings[0] == VEC_B.tolist()
    assert embeddings[1] == VEC_A.tolist()


def test_hit_sandwiched_between_misses(remote_backend_returning_b):
    es._cache_embedding("A", VEC_A)

    embeddings, _ = es.generate_embeddings_batch(["B1", "A", "B2"], normalize=False)

    assert embeddings[0] == VEC_B.tolist()
    assert embeddings[1] == VEC_A.tolist()
    assert embeddings[2] == VEC_B.tolist()


def test_all_hits_and_all_misses_still_work(remote_backend_returning_b):
    all_miss, _ = es.generate_embeddings_batch(["B1", "B2"], normalize=False)
    assert all_miss == [VEC_B.tolist(), VEC_B.tolist()]

    es._cache_embedding("A1", VEC_A)
    es._cache_embedding("A2", VEC_A)
    all_hit, _ = es.generate_embeddings_batch(["A1", "A2"], normalize=False)
    assert all_hit == [VEC_A.tolist(), VEC_A.tolist()]


def test_remote_embeddings_use_router_feature_extraction_endpoint(monkeypatch):
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return [[0.1, 0.2]]

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, headers=None, json=None):
            captured["url"] = url
            return FakeResponse()

    monkeypatch.setattr(es.httpx, "Client", FakeClient)
    monkeypatch.setattr(es, "HF_API_TOKEN", "token")
    monkeypatch.setattr(es, "HF_EMBEDDING_MODEL", "model")
    monkeypatch.setattr(es, "HF_INFERENCE_BASE_URL", "https://router.huggingface.co/hf-inference/models")

    embeddings = es._generate_remote_embeddings(["hello"], normalize=False)

    assert captured["url"] == "https://router.huggingface.co/hf-inference/models/model/pipeline/feature-extraction"
    assert embeddings[0].shape == (2,)
