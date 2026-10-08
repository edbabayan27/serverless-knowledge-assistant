import pytest

from knowledge_assistant.hybrid_search import (
    Bm25Index,
    reciprocal_rank_fusion,
    tokenize,
    weighted_fusion,
)

DOCUMENTS = {
    "throttling": "Throttling protects your API. Configure throttling in API Gateway.",
    "dlq": "Send failed events to a dead-letter queue (DLQ) so that they can be retried later.",
    "proxy": "Amazon RDS Proxy pools database connections for Lambda functions.",
}


def test_tokenize_lowercases_and_drops_stopwords():
    assert tokenize("How do I use the RDS Proxy with Lambda?") == ["rds", "proxy", "lambda"]


def test_bm25_ranks_the_document_with_the_query_terms_first():
    index = Bm25Index(DOCUMENTS)

    results = index.search("How should I configure API throttling?", top_k=3)

    assert results[0][0] == "throttling"
    assert all(score > 0 for _key, score in results)


def test_bm25_only_returns_documents_that_share_a_term():
    index = Bm25Index(DOCUMENTS)

    assert [key for key, _score in index.search("DLQ retries", top_k=3)] == ["dlq"]
    assert index.search("kubernetes", top_k=3) == []


def test_bm25_rare_terms_weigh_more_than_common_ones():
    documents = {"a": "lambda lambda lambda", "b": "lambda proxy", "c": "lambda"}

    [(best, _score), *_] = Bm25Index(documents).search("lambda proxy", top_k=3)

    assert best == "b"


def test_weighted_fusion_alpha_one_and_zero_reproduce_each_ranking():
    vector = [("a", 0.9), ("b", 0.8), ("c", 0.1)]
    keyword = [("c", 12.0), ("b", 3.0)]

    assert [key for key, _ in weighted_fusion(vector, keyword, alpha=1.0)] == ["a", "b", "c"]
    assert weighted_fusion(vector, keyword, alpha=0.0)[0][0] == "c"


def test_weighted_fusion_normalises_scores_before_mixing():
    vector = [("a", 0.9), ("b", 0.5)]  # normalised: a=1.0, b=0.0
    keyword = [("b", 20.0), ("a", 10.0)]  # normalised: b=1.0, a=0.0

    fused = dict(weighted_fusion(vector, keyword, alpha=0.75))

    assert fused == {"a": pytest.approx(0.75), "b": pytest.approx(0.25)}


def test_weighted_fusion_rejects_an_invalid_alpha():
    with pytest.raises(ValueError, match="alpha"):
        weighted_fusion([], [], alpha=1.5)


def test_rrf_rewards_chunks_that_both_lists_rank_high():
    vector = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    keyword = [("b", 9.0), ("d", 5.0), ("a", 1.0)]

    fused = reciprocal_rank_fusion([vector, keyword])

    assert [key for key, _ in fused] == ["b", "a", "d", "c"]
    assert fused[0][1] == pytest.approx(1 / 62 + 1 / 61)


def test_rrf_weights_favour_one_list():
    vector = [("a", 0.9), ("b", 0.8)]
    keyword = [("b", 9.0), ("a", 1.0)]

    fused = reciprocal_rank_fusion([vector, keyword], weights=[2.0, 1.0])

    assert fused[0][0] == "a"
