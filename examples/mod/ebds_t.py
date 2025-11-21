import os
import time
import numpy as np
from sklearn.cluster import KMeans

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

from src.model_objects.model_loader import ModelLoader
from src.utils.device_watcher import DeviceWatcher


# Test datasets
english_sentences = [
    "The cat is sleeping on the couch",
    "A feline is resting on the sofa",
    "The dog is barking loudly",
    "I love eating pizza for dinner",
    "The weather is beautiful today",
    "Machine learning is fascinating",
    "The kitten is napping on the furniture",
    "Python is a programming language",
    "The sun is shining brightly",
    "Neural networks are powerful tools"
]

chinese_sentences = [
    "猫在沙发上睡觉",
    "一只猫科动物正在沙发上休息",
    "狗在大声吠叫",
    "我喜欢晚餐吃披萨",
    "今天天气很好",
    "机器学习很迷人",
    "小猫在家具上打盹",
    "Python是一种编程语言",
    "阳光明媚地照耀着",
    "神经网络是强大的工具"
]


def test_embeddings_with_clustering(model_name: str, sentences: list[str], lang: str, device: str = "cuda"):
    """Test embeddings and perform k-means clustering."""
    print(f"\n{'='*70}")
    print(f"Model: {model_name} | Language: {lang}")
    print(f"{'='*70}")

    try:
        # Load model
        model = ModelLoader.load_model(model_name, device=device)

        # Get embeddings and measure time
        start_time = time.time()
        results = model.get_embeddings(sentences)
        embed_time = time.time() - start_time

        # Extract embeddings
        embeddings = np.array([r['embedding'] for r in results])

        # K-means clustering (3 clusters)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        # Print results
        print(f"Embedding time: {embed_time:.3f}s")
        print(f"\nClusters:")
        for cluster_id in range(3):
            indices = np.where(labels == cluster_id)[0]
            print(f"\nCluster {cluster_id}:")
            for idx in indices:
                print(f"  - {sentences[idx]}")

    except Exception as e:
        print(f"FAILED: {str(e)}")


if __name__ == "__main__":
    print(f"Device: {DeviceWatcher.get_available_device()}")

    models = [
        "Qwen/Qwen3-Embedding-0.6B",
        "jinaai/jina-embeddings-v3",
        "jinaai/jina-embeddings-v2-base-zh",
        "jinaai/jina-embeddings-v4",
    ]

    for model_name in models:
        test_embeddings_with_clustering(model_name, english_sentences, "EN")
        test_embeddings_with_clustering(model_name, chinese_sentences, "ZH")