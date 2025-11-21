import requests


BASE_URL = "http://localhost:7009/api/v1"


def test_rerank_http():
    """Test Rerank API with a query and documents."""
    print(f"\n{'='*60}")
    print(f"Testing Rerank API")
    print(f"{'='*60}")

    try:
        query = "What is machine learning?"
        documents = [
            "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
            "The weather today is sunny with a high of 75 degrees.",
            "Deep learning uses neural networks with many layers to process complex patterns.",
            "Python is a popular programming language for data science.",
        ]

        response = requests.post(
            f"{BASE_URL}/rerank",
            json={
                "query": query,
                "documents": documents,
                "top_k": 3
            }
        )

        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.text}")
            return

        results = response.json()["results"]

        print(f"Query: {query}\n")
        print("Ranked results:")
        for idx, score, doc in results:
            print(f"  [{idx}] {score:.4f}: {doc[:60]}...")

        print(f"\nTest PASSED")

    except Exception as e:
        print(f"Test FAILED")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Make sure the server is running on http://localhost:7009")
    print("Start server with: python main.py")

    test_rerank_http()