import numpy as np
from typing import Literal, TypedDict

from src.algorisms.algorism_structs import SentenceEmbedding


class BucketResult(TypedDict):
    bucket: str
    sentences: list[str]


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def _dot_product(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate dot product between two vectors."""
    return np.dot(a, b)


def similarity_bucketing(
        data: list[SentenceEmbedding],
        buckets: list[SentenceEmbedding],
        score_method: Literal["dot", "cosine"] = "cosine",
        allow_orphan_bucket: bool = False,
        orphan_threshold: float = 0.5
) -> list[BucketResult]:
    """
    Assign each sentence to the closest bucket based on embedding similarity.

    Args:
        data: List of sentences with embeddings to be bucketed
        buckets: List of bucket definitions with embeddings
        score_method: Similarity metric ("dot" or "cosine")
        allow_orphan_bucket: Whether to create orphan bucket for low-similarity items
        orphan_threshold: Minimum similarity score to assign to a bucket (if allow_orphan_bucket=True)

    Returns:
        List of buckets with assigned sentences
    """
    score_func = _cosine_similarity if score_method == "cosine" else _dot_product

    # Initialize bucket results
    bucket_results: list[BucketResult] = [
        {"bucket": bucket["text"], "sentences": []}
        for bucket in buckets
    ]

    if allow_orphan_bucket:
        bucket_results.append({"bucket": "orphan", "sentences": []})

    # Assign each data point to closest bucket
    for item in data:
        data_emb = np.array(item["embedding"])
        best_score = -float("inf")
        best_bucket_idx = -1

        # Find closest bucket
        for idx, bucket in enumerate(buckets):
            bucket_emb = np.array(bucket["embedding"])
            score = score_func(data_emb, bucket_emb)

            if score > best_score:
                best_score = score
                best_bucket_idx = idx

        # Assign to bucket or orphan
        if allow_orphan_bucket and best_score < orphan_threshold:
            bucket_results[-1]["sentences"].append(item["text"])
        else:
            bucket_results[best_bucket_idx]["sentences"].append(item["text"])

    return bucket_results
