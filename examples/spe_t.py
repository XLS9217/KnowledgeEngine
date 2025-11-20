import asyncio
from pathlib import Path
import numpy as np
from PIL import Image
from src.task_orchestrator.orchestrator_interface import OrchestratorInterface


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


async def main():
    # Initialize the single process engine
    OrchestratorInterface.initialize("single_process_engine")

    # Example 1: Get embeddings and compute cosine similarity
    texts = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks.",
        "The weather is sunny today."
    ]

    embeddings = []
    for text in texts:
        emb = await OrchestratorInterface.get_embedding(text)
        embeddings.append(emb)

    print("Cosine Similarities:")
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            print(f"  '{texts[i][:40]}...' vs '{texts[j][:40]}...': {sim:.4f}")

    # Example 2: Rerank documents
    query = "What is machine learning?"
    documents = [
        "Machine learning is a subset of artificial intelligence.",
        "The weather is sunny today.",
        "Deep learning uses neural networks for complex tasks.",
        "Python is a popular programming language."
    ]
    top_k = 2
    reranked = await OrchestratorInterface.rerank(query, documents, top_k)
    print(f"\nReranked results (top {top_k}):")
    for i, result in enumerate(reranked):
        print(f"  {i+1}. {result}")

    # Example 3: CLIP image-text similarity
    image_path = Path(__file__).parent / "page.png"
    image = Image.open(image_path)

    clip_texts = [
        "a photo of a cat",
        "a photo of a dog",
        "a document with text",
        "a webpage full of text",
    ]

    results = await OrchestratorInterface.get_clip_scores(image, clip_texts)

    print("\nImage-Text Similarity Scores:")
    for text, score in results:
        print(f"  {text:30s} -> {score:.4f}")

    best = max(results, key=lambda x: x[1])
    print(f"\nBest match: '{best[0]}' with {best[1]:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
