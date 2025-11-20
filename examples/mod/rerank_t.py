from src.model_objects.model_loader import ModelLoader

model = ModelLoader.load_model(
    "Qwen/Qwen3-Reranker-0.6B",
    device="cuda"
)

query = "What is machine learning?"
documents = [
    "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
    "The weather today is sunny with a high of 75 degrees.",
    "Deep learning uses neural networks with many layers to process complex patterns.",
    "Python is a popular programming language for data science.",
]

results = model.rerank(query, documents, top_k=3)

print(f"Query: {query}\n")
print("Ranked results:")
for idx, score, doc in results:
    print(f"  [{idx}] {score:.4f}: {doc[:60]}...")
