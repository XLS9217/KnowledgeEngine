import json

import requests
import numpy as np


BASE_URL = "http://localhost:7009"


def cosine_similarity(vec1, vec2) -> float:
    """Calculate cosine similarity between two vectors."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))


def test_embedding_http():
    """Test embedding API with sample texts in English and Chinese."""
    print(f"\n{'='*60}")
    print(f"Testing Embedding API")
    print(f"{'='*60}")

    try:
        # Test texts
        text1_en = "hello world"
        text2_zh = "你好世界"
        text3_zh = "傻逼玩意儿"

        # Get embedding for text1
        response = requests.post(
            f"{BASE_URL}/embedding",
            json={"text": text1_en}
        )
        response.raise_for_status()
        resp1 = response.json()
        ebd1 = resp1.get("embedding")
        model1 = resp1.get("model_name")

        # Get embedding for text2
        response = requests.post(
            f"{BASE_URL}/embedding",
            json={"text": text2_zh}
        )
        response.raise_for_status()
        resp2 = response.json()
        ebd2 = resp2.get("embedding")
        model2 = resp2.get("model_name")

        # Get embedding for text3
        response = requests.post(
            f"{BASE_URL}/embedding",
            json={"text": text3_zh}
        )
        response.raise_for_status()
        resp3 = response.json()
        ebd3 = resp3.get("embedding")
        model3 = resp3.get("model_name")

        # Calculate similarities
        sim_en_zh = cosine_similarity(ebd1, ebd2)
        sim_zh_zh = cosine_similarity(ebd3, ebd2)

        # Print results
        print(f"Text 1 (EN): '{text1_en}' | model: {model1 or 'unknown'}")
        print(f"Text 2 (ZH): '{text2_zh}' | model: {model2 or 'unknown'}")
        print(f"Text 3 (ZH): '{text3_zh}' | model: {model3 or 'unknown'}")
        print(f"\nSimilarity (EN-ZH): {sim_en_zh:.4f}")
        print(f"Similarity (ZH-ZH): {sim_zh_zh:.4f}")
        print(f"\nTest PASSED")

    except Exception as e:
        print(f"Test FAILED")
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    print("Make sure the server is running on http://localhost:7009")
    print("Start server with: python main.py")

    test_embedding_http()

