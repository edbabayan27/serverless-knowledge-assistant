"""Keyword search (BM25) and the two ways of merging it with vector search results.

Vector search matches meaning ("stop my API from being overwhelmed" finds "throttling"); keyword
search matches exact terms ("RDS Proxy", "DLQ", "REL 1"). Hybrid search runs both and merges the
rankings:

- `weighted_fusion`: rescales each list's scores to 0..1 and mixes them with a weight `alpha`
  (1.0 = only vectors, 0.0 = only keywords).
- `reciprocal_rank_fusion` (RRF): ignores scores and adds up 1 / (k + rank) for each list, so a
  chunk ranked high by either search ends up near the top.

S3 Vectors only supports vector search, so the keyword index is built here. For one whitepaper
(about 150 chunks) it is a few hundred KB and fits easily in Lambda memory.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence

Ranking = list[tuple[str, float]]  # (chunk key, score), best first

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "has",
        "have",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "should",
        "so",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "this",
        "to",
        "use",
        "used",
        "using",
        "was",
        "we",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "why",
        "will",
        "with",
        "you",
        "your",
    }
)


def tokenize(text: str) -> list[str]:
    """Lowercase words and numbers, without common English stopwords."""
    return [token for token in _TOKEN.findall(text.lower()) if token not in _STOPWORDS]


class Bm25Index:
    """Okapi BM25 keyword search over a fixed set of documents.

    k1 controls how quickly repeated terms stop adding to the score; b controls how much long
    documents are penalised. 1.5 and 0.75 are the usual defaults.
    """

    def __init__(self, documents: Mapping[str, str], *, k1: float = 1.5, b: float = 0.75):
        self._k1 = k1
        self._b = b
        self._term_counts = {key: Counter(tokenize(text)) for key, text in documents.items()}
        self._lengths = {key: sum(counts.values()) for key, counts in self._term_counts.items()}
        self._average_length = sum(self._lengths.values()) / max(len(self._lengths), 1)
        document_frequency = Counter(
            term for counts in self._term_counts.values() for term in counts
        )
        total = len(self._term_counts)
        self._idf = {
            term: math.log(1 + (total - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def search(self, query: str, *, top_k: int) -> Ranking:
        """The `top_k` best-matching documents with a score above zero, best first."""
        terms = [term for term in tokenize(query) if term in self._idf]
        scores: dict[str, float] = {}
        for key, counts in self._term_counts.items():
            length_factor = self._k1 * (
                1 - self._b + self._b * self._lengths[key] / self._average_length
            )
            score = sum(
                self._idf[term] * counts[term] * (self._k1 + 1) / (counts[term] + length_factor)
                for term in terms
                if counts[term]
            )
            if score > 0:
                scores[key] = score
        return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]


def weighted_fusion(vector: Ranking, keyword: Ranking, *, alpha: float) -> Ranking:
    """Mix min-max normalised scores: alpha * vector + (1 - alpha) * keyword.

    Each list is rescaled to 0..1 on its own (best = 1, weakest = 0, as in OpenSearch's min_max
    normalisation). A chunk missing from one list gets 0 from that list.
    """
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    vector_scores, keyword_scores = _min_max(vector), _min_max(keyword)
    keys = vector_scores.keys() | keyword_scores.keys()
    fused = {
        key: alpha * vector_scores.get(key, 0.0) + (1 - alpha) * keyword_scores.get(key, 0.0)
        for key in keys
    }
    return _ranked(fused)


def reciprocal_rank_fusion(
    rankings: Sequence[Ranking], *, weights: Sequence[float] | None = None, k: int = 60
) -> Ranking:
    """Sum weight / (k + rank) over the rankings (rank starts at 1). k=60 is the usual default."""
    weights = weights or [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights and rankings must have the same length")
    fused: dict[str, float] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, (key, _score) in enumerate(ranking, start=1):
            fused[key] = fused.get(key, 0.0) + weight / (k + rank)
    return _ranked(fused)


def _min_max(ranking: Ranking) -> dict[str, float]:
    if not ranking:
        return {}
    scores = [score for _key, score in ranking]
    low, high = min(scores), max(scores)
    if high == low:
        return {key: 1.0 for key, _score in ranking}
    return {key: (score - low) / (high - low) for key, score in ranking}


def _ranked(scores: Mapping[str, float]) -> Ranking:
    # Ties are broken by key so results are deterministic.
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))
