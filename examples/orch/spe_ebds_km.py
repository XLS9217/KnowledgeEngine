import asyncio
import json
from pathlib import Path

from src.task_orchestrator.orchestrator_interface import OrchestratorInterface


async def main():
    # Initialize the single process engine
    OrchestratorInterface.initialize("single_process_engine")

    # 10 sentences with 2 clear topics (5 about animals, 5 about technology)
    sentences = [
        # Topic 1: Animals
        "The cat is sleeping peacefully on the couch",
        "Dogs are loyal companions and great pets",
        "The elephant is the largest land animal",
        "Birds migrate south for the winter season",
        "Fish swim gracefully in the ocean",

        # Topic 2: Technology
        "Machine learning algorithms improve with data",
        "Cloud computing enables scalable applications",
        "Artificial intelligence is transforming industries",
        "Blockchain technology ensures secure transactions",
        "Neural networks mimic the human brain"
    ]

    print(f"Getting embeddings for {len(sentences)} sentences...")
    print("=" * 80)

    # Get embeddings for all sentences
    embeddings = []
    for i, sentence in enumerate(sentences):
        embedding = await OrchestratorInterface.get_embedding(sentence)
        embeddings.append({
            "id": f"sentence_{i}",
            "text": sentence,
            "embedding": embedding
        })
        print(f"{i+1}. {sentence}")

    # Perform K-Means clustering with k=2 (expecting 2 topics)
    print("\n" + "=" * 80)
    print("Running K-Means clustering (k=2)...")
    clusters = await OrchestratorInterface.k_means(embeddings, 2)

    # Print results
    print("\nClustering Results:")
    print("=" * 80)
    print(clusters)

if __name__ == "__main__":
    asyncio.run(main())
