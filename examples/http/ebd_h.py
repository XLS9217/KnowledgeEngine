import requests
import numpy as np


def cosine_similarity(vec1, vec2) -> float:
    """Calculate cosine similarity between two vectors."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))


# Base URL of the API
BASE_URL = "http://localhost:7009/api/v1"


def get_embedding(text: str):
    """Get embedding for a single text via HTTP API."""
    response = requests.post(
        f"{BASE_URL}/embedding",
        json={"text": text}
    )
    response.raise_for_status()
    return response.json()["embedding"]


# Test the embedding endpoint with the same logic as ebd_t.py
print("Testing embedding endpoint...")

ebd1 = get_embedding("hello world")
print(f"Generated embedding for 'hello world' with dimension: {len(ebd1)}")

ebd2 = get_embedding("hello there")
similarity_12 = cosine_similarity(ebd1, ebd2)
print(f"Cosine similarity between 'hello world' and 'hello there': {similarity_12:.4f}")

ebd3 = get_embedding("goodbye")
similarity_23 = cosine_similarity(ebd3, ebd2)
print(f"Cosine similarity between 'goodbye' and 'hello there': {similarity_23:.4f}")