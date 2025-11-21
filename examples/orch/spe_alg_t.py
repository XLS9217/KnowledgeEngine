import asyncio
import json
from pathlib import Path

from src.task_orchestrator.orchestrator_interface import OrchestratorInterface


async def main():
    # Initialize the single process engine
    OrchestratorInterface.initialize("single_process_engine")

    # Load the data
    data_path = Path(__file__).parent.parent / "test_data" / "chinese_dialogue_embeddings.json"
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} sentences")

    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    # K-Means clustering
    print("\nRunning K-Means clustering...")
    kmeans_clusters = await OrchestratorInterface.k_means(data, 3)

    kmeans_output = {"clusters": kmeans_clusters}
    kmeans_path = output_dir / "orch_k_means.json"
    with open(kmeans_path, "w", encoding="utf-8") as f:
        json.dump(kmeans_output, f, ensure_ascii=False, indent=2)
    print(f"K-Means saved to {kmeans_path}")

    # Agglomerative clustering (the rest param will adjust in engine)
    print("\nRunning Agglomerative clustering...")
    agg_clusters = await OrchestratorInterface.agglomerative(data, 3)

    agg_output = {"clusters": agg_clusters}
    agg_path = output_dir / "orch_agglomerative.json"
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(agg_output, f, ensure_ascii=False, indent=2)
    print(f"Agglomerative saved to {agg_path}")

    # Auto Agglomerative clustering
    print("\nRunning Auto Agglomerative clustering...")
    auto_agg_clusters = await OrchestratorInterface.auto_agglomerative(data, 10)

    auto_agg_output = {"clusters": auto_agg_clusters}
    auto_agg_path = output_dir / "orch_auto_agglomerative.json"
    with open(auto_agg_path, "w", encoding="utf-8") as f:
        json.dump(auto_agg_output, f, ensure_ascii=False, indent=2)
    print(f"Auto Agglomerative saved to {auto_agg_path}")

    # Similarity bucketing
    print("\nPreparing bucket topics...")
    bucket_topics = [
        "讨论popping舞蹈动作技巧、练习和freeze动作的内容",
        "关于去KFC吃鸡翅、薯条和可乐的讨论",
        "使用Sora2生成视频、合成草稿、高帧率拍摄和后期制作的技术讨论"
    ]

    # Get embeddings for bucket topics
    buckets = []
    for topic in bucket_topics:
        embedding = await OrchestratorInterface.get_embedding(topic)
        buckets.append({
            "text": topic,
            "embedding": embedding.tolist()
        })

    print("\nRunning similarity bucketing...")
    bucket_results = await OrchestratorInterface.similarity_bucketing(
        data=data,
        buckets=buckets,
        score_method="cosine",
        allow_orphan_bucket=True,
        orphan_threshold=0.2
    )

    bucket_output = {"buckets": bucket_results}
    bucket_path = output_dir / "orch_similarity_bucketing.json"
    with open(bucket_path, "w", encoding="utf-8") as f:
        json.dump(bucket_output, f, ensure_ascii=False, indent=2)
    print(f"Similarity bucketing saved to {bucket_path}")

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

    print("\nBucket summary:")
    for bucket in bucket_results:
        bucket_name = bucket["bucket"]
        sentence_count = len(bucket["sentences"])
        if bucket_name == "orphan":
            print(f"[Orphan Bucket] - {sentence_count} sentences")
        else:
            print(f"[{bucket_name[:50]}...] - {sentence_count} sentences")


if __name__ == "__main__":
    asyncio.run(main())