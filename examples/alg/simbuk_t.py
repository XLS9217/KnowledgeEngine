import asyncio
import json
from pathlib import Path

from src.task_orchestrator.orchestrator_interface import OrchestratorInterface
from src.algorisms.bucketing import similarity_bucketing


async def main():
    # Initialize engine
    OrchestratorInterface.initialize("single_process_engine")

    # Load embeddings
    data_path = Path(__file__).parent.parent / "test_data" / "chinese_dialogue_embeddings.json"
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} sentences")

    # Define bucket topics based on the dialogue themes
    bucket_topics = [
        "讨论popping舞蹈动作技巧、练习和freeze动作的内容",  # Popping dance technique and practice
        "关于去KFC吃鸡翅、薯条和可乐的讨论",  # KFC food discussion
        "使用Sora2生成视频、合成草稿、高帧率拍摄和后期制作的技术讨论"  # Sora2 video generation and technical production
    ]

    # Get embeddings for bucket topics
    print("\nGenerating bucket embeddings...")
    buckets = []
    for topic in bucket_topics:
        embedding = await OrchestratorInterface.get_embedding(topic)
        buckets.append({
            "text": topic,
            "embedding": embedding.tolist()
        })

    # Run similarity bucketing with orphan bucket
    print("\nRunning similarity bucketing (cosine similarity, allow orphan)...")
    results = similarity_bucketing(
        data=data,
        buckets=buckets,
        score_method="cosine",
        allow_orphan_bucket=True,
        orphan_threshold=0.2
    )

    # Save results
    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "similarity_bucketing.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"buckets": results}, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to {output_path}")

    # Print summary
    print("\nBucket summary:")
    for bucket in results:
        bucket_name = bucket["bucket"]
        sentence_count = len(bucket["sentences"])

        if bucket_name == "orphan":
            print(f"\n[Orphan Bucket] - {sentence_count} sentences:")
        else:
            print(f"\n[{bucket_name[:50]}...] - {sentence_count} sentences:")

        for sentence in bucket["sentences"][:3]:  # Show first 3
            print(f"  - {sentence[:60]}...")

        if sentence_count > 3:
            print(f"  ... and {sentence_count - 3} more")


if __name__ == "__main__":
    asyncio.run(main())
