import json
from pathlib import Path

from src.algorisms.clustering import k_means, agglomerative, auto_agglomerative


def main():
    # Load embeddings
    data_path = Path(__file__).parent.parent / "test_data" / "chinese_dialogue_embeddings.json"
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} sentences")

    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    # K-Means clustering
    n_clusters = 5
    print(f"\nRunning K-Means with {n_clusters} clusters...")
    kmeans_clusters = k_means(data, n_clusters)

    kmeans_output = {"clusters": kmeans_clusters}
    kmeans_path = output_dir / "k_means.json"
    with open(kmeans_path, "w", encoding="utf-8") as f:
        json.dump(kmeans_output, f, ensure_ascii=False, indent=2)
    print(f"K-Means saved to {kmeans_path}")

    # Agglomerative clustering
    print(f"\nRunning Agglomerative clustering with {n_clusters} clusters...")
    agg_clusters = agglomerative(data, n_clusters, linkage="ward")

    agg_output = {"clusters": agg_clusters}
    agg_path = output_dir / "agglomerative.json"
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(agg_output, f, ensure_ascii=False, indent=2)
    print(f"Agglomerative saved to {agg_path}")

    # Auto Agglomerative clustering
    max_clusters = 10
    print(f"\nRunning Auto Agglomerative clustering (max {max_clusters} clusters)...")
    auto_agg_clusters = auto_agglomerative(data, max_clusters, linkage="ward")

    auto_agg_output = {"clusters": auto_agg_clusters}
    auto_agg_path = output_dir / "auto_agglomerative.json"
    with open(auto_agg_path, "w", encoding="utf-8") as f:
        json.dump(auto_agg_output, f, ensure_ascii=False, indent=2)
    print(f"Auto Agglomerative saved to {auto_agg_path}")

    # Print summary
    print("\nK-Means cluster sizes:")
    for i, cluster in enumerate(kmeans_clusters):
        print(f"  Cluster {i}: {len(cluster)} sentences")

    print("\nAgglomerative cluster sizes:")
    for i, cluster in enumerate(agg_clusters):
        print(f"  Cluster {i}: {len(cluster)} sentences")

    print("\nAuto Agglomerative cluster sizes:")
    for i, cluster in enumerate(auto_agg_clusters):
        print(f"  Cluster {i}: {len(cluster)} sentences")


if __name__ == "__main__":
    main()
