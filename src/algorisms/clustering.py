import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

from src.algorisms.algorism_structs import SentenceEmbedding


def k_means(data: list[SentenceEmbedding], n_clusters: int) -> list[list[str]]:
    """
    Cluster sentences using K-Means on their embeddings.

    Args:
        data: List of dicts with 'text' and 'embedding' keys
        n_clusters: Number of clusters

    Returns:
        List of clusters, each containing sentences in that cluster
    """
    embeddings = np.array([item["embedding"] for item in data])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    clusters: list[list[str]] = [[] for _ in range(n_clusters)]
    for idx, label in enumerate(labels):
        clusters[int(label)].append(data[idx]["text"])

    return clusters


def agglomerative(data: list[SentenceEmbedding], n_clusters: int, linkage: str = "ward") -> list[list[str]]:
    """
    Cluster sentences using Agglomerative Clustering on their embeddings.

    Args:
        data: List of dicts with 'text' and 'embedding' keys
        n_clusters: Number of clusters
        linkage: Linkage criterion ('ward', 'complete', 'average', 'single')

    Returns:
        List of clusters, each containing sentences in that cluster
    """
    embeddings = np.array([item["embedding"] for item in data])
    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = agg.fit_predict(embeddings)

    clusters: list[list[str]] = [[] for _ in range(n_clusters)]
    for idx, label in enumerate(labels):
        clusters[int(label)].append(data[idx]["text"])

    return clusters


def auto_agglomerative(data: list[SentenceEmbedding], max_clusters: int, linkage: str = "ward") -> list[list[str]]:
    """
    Automatically determine optimal number of clusters using silhouette score.

    Args:
        data: List of dicts with 'text' and 'embedding' keys
        max_clusters: Maximum number of clusters to try
        linkage: Linkage criterion ('ward', 'complete', 'average', 'single')

    Returns:
        List of clusters, each containing sentences in that cluster
    """
    embeddings = np.array([item["embedding"] for item in data])

    best_score = -1
    best_n_clusters = 2
    best_labels = None

    # Try different numbers of clusters
    for n_clusters in range(2, min(max_clusters + 1, len(data))):
        agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        labels = agg.fit_predict(embeddings)

        score = silhouette_score(embeddings, labels)
        if score > best_score:
            best_score = score
            best_n_clusters = n_clusters
            best_labels = labels

    print(f"Optimal clusters: {best_n_clusters} (silhouette score: {best_score:.4f})")

    # Build clusters with best labels
    clusters: list[list[str]] = [[] for _ in range(best_n_clusters)]
    for idx, label in enumerate(best_labels):
        clusters[int(label)].append(data[idx]["text"])

    return clusters