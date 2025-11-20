import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering

from src.algorisms.algorism_structs import SentenceEmbedding


def k_means(data: list[SentenceEmbedding], n_clusters: int) -> list[list[str]]:
    """
    Cluster sentences using K-Means on their embeddings.

    Args:
        data: List of dicts with 'sentence' and 'embedding' keys
        n_clusters: Number of clusters

    Returns:
        List of clusters, each containing sentences in that cluster
    """
    embeddings = np.array([item["embedding"] for item in data])
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(embeddings)

    clusters = [[] for _ in range(n_clusters)]
    for idx, label in enumerate(labels):
        clusters[label].append(data[idx]["sentence"])

    return clusters


def agglomerative(data: list[SentenceEmbedding], n_clusters: int, linkage: str = "ward") -> list[list[str]]:
    """
    Cluster sentences using Agglomerative Clustering on their embeddings.

    Args:
        data: List of dicts with 'sentence' and 'embedding' keys
        n_clusters: Number of clusters
        linkage: Linkage criterion ('ward', 'complete', 'average', 'single')

    Returns:
        List of clusters, each containing sentences in that cluster
    """
    embeddings = np.array([item["embedding"] for item in data])
    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = agg.fit_predict(embeddings)

    clusters = [[] for _ in range(n_clusters)]
    for idx, label in enumerate(labels):
        clusters[label].append(data[idx]["sentence"])

    return clusters