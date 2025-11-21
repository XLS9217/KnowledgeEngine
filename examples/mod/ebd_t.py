import os
import numpy as np

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

from src.model_objects.model_loader import ModelLoader
from src.utils.device_watcher import DeviceWatcher


def cosine_similarity(vec1, vec2) -> float:
    """Calculate cosine similarity between two vectors."""
    # Convert to numpy if tensors
    if hasattr(vec1, 'cpu'):
        vec1 = vec1.cpu().numpy()
    if hasattr(vec2, 'cpu'):
        vec2 = vec2.cpu().numpy()
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))


def test_embedding_model(model_name: str, device: str = "cuda"):
    """Test an embedding model with sample texts in English and Chinese."""
    print(f"\n{'='*60}")
    print(f"Testing model: {model_name}")
    print(f"{'='*60}")

    try:
        # Load model
        model = ModelLoader.load_model(model_name, device=device)

        # Test texts
        text1_en = "hello world"
        text2_zh = "你好世界"
        text3_zh = "傻逼玩意儿"

        # Get embeddings
        ebd1 = model.get_embedding(text1_en)
        ebd2 = model.get_embedding(text2_zh)
        ebd3 = model.get_embedding(text3_zh)

        # Calculate similarities
        sim_en_zh = cosine_similarity(ebd1, ebd2)
        sim_zh_zh = cosine_similarity(ebd3, ebd2)

        # Print results
        print(f"Text 1 (EN): '{text1_en}'")
        print(f"Text 2 (ZH): '{text2_zh}'")
        print(f"Text 3 (ZH): '{text3_zh}'")
        print(f"\nSimilarity (EN-ZH): {sim_en_zh:.4f}")
        print(f"Similarity (ZH-ZH): {sim_zh_zh:.4f}")
        print(f"\nTest PASSED for {model_name}")

    except Exception as e:
        print(f"Test FAILED for {model_name}")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Show available device
    print(f"Available device: {DeviceWatcher.get_available_device()}")

    # List of all embedding models to test
    embedding_models = [
        "Qwen/Qwen3-Embedding-0.6B",
        "jinaai/jina-embeddings-v3",
        "jinaai/jina-embeddings-v2-base-zh",
        "jinaai/jina-embeddings-v4",
    ]

    # Test each model
    for model_name in embedding_models:
        test_embedding_model(model_name)



