import requests


BASE_URL = "http://localhost:7009/api/v1"


def test_batch_embedding_and_clustering(sentences: list[str], lang: str):
    """Test batch embeddings followed by k-means, agglomerative, and auto-agglomerative clustering."""
    print(f"\n{'='*70}")
    print(f"Testing Batch Embeddings + Clustering APIs ({lang})")
    print(f"{'='*70}")

    try:

        print(f"\n1. Getting embeddings for {len(sentences)} sentences...")
        print("-" * 70)

        # Get batch embeddings
        response = requests.post(
            f"{BASE_URL}/embeddings",
            json={"text_list": sentences}
        )

        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.text}")
            return

        embeddings_result = response.json()["embeddings"]

        # Prepare data for clustering algorithms (needs "text" and "embedding" fields)
        data = [
            {"text": item["text"], "embedding": item["embedding"]}
            for item in embeddings_result
        ]

        print(f"✓ Got {len(data)} embeddings\n")

        # Test K-Means
        print("2. Testing K-Means Clustering (n_clusters=3)...")
        print("-" * 70)

        response = requests.post(
            f"{BASE_URL}/algorithm/k-means",
            json={
                "data": data,
                "n_clusters": 3
            }
        )

        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.text}")
            return

        kmeans_result = response.json()["result"]

        print("K-Means Clusters:")
        for cluster_id, cluster_texts in enumerate(kmeans_result):
            print(f"\nCluster {cluster_id}:")
            for text in cluster_texts:
                print(f"  - {text}")

        print(f"\n✓ K-Means completed\n")

        # Test Agglomerative Clustering
        print("3. Testing Agglomerative Clustering (n_clusters=3, linkage=ward)...")
        print("-" * 70)

        response = requests.post(
            f"{BASE_URL}/algorithm/agglomerative",
            json={
                "data": data,
                "n_clusters": 3,
                "linkage": "ward"
            }
        )

        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.text}")
            return

        agg_result = response.json()["result"]

        print("Agglomerative Clusters:")
        for cluster_id, cluster_texts in enumerate(agg_result):
            print(f"\nCluster {cluster_id}:")
            for text in cluster_texts:
                print(f"  - {text}")

        print(f"\n✓ Agglomerative completed\n")

        # Test Auto-Agglomerative Clustering
        print("4. Testing Auto-Agglomerative Clustering (max_clusters=5, linkage=ward)...")
        print("-" * 70)

        response = requests.post(
            f"{BASE_URL}/algorithm/auto-agglomerative",
            json={
                "data": data,
                "max_clusters": 5,
                "linkage": "ward"
            }
        )

        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.text}")
            return

        auto_agg_result = response.json()["result"]

        print(f"Auto-Agglomerative Clusters:")
        for cluster_id, cluster_texts in enumerate(auto_agg_result):
            print(f"\nCluster {cluster_id}:")
            for text in cluster_texts:
                print(f"  - {text}")

        print(f"\n✓ Auto-Agglomerative completed")
        print(f"\n{'='*70}")
        print(f"All tests PASSED")
        print(f"{'='*70}")

    except Exception as e:
        print(f"\nTest FAILED")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Make sure the server is running on http://localhost:7009")
    print("Start server with: python main.py")

    # English test data
    english_sentences = [
        # Animal-related (cluster 1)
        "The cat is sleeping on the couch",
        "Dogs are loyal companions",
        "A bird is singing in the tree",
        # Food-related (cluster 2)
        "I love eating pizza for dinner",
        "Fresh vegetables are healthy",
        "This chocolate cake is delicious",
        # Tech-related (cluster 3)
        "Machine learning is fascinating",
        "Python is a programming language",
        "Neural networks are powerful tools",
    ]

    # Chinese test data
    chinese_sentences = [
        # 动物相关 (cluster 1)
        "猫在沙发上睡觉",
        "狗是忠诚的伴侣",
        "一只鸟在树上唱歌",
        # 食物相关 (cluster 2)
        "我喜欢晚餐吃披萨",
        "新鲜蔬菜很健康",
        "这个巧克力蛋糕很美味",
        # 科技相关 (cluster 3)
        "机器学习很迷人",
        "Python是一种编程语言",
        "神经网络是强大的工具",
    ]

    test_batch_embedding_and_clustering(english_sentences, "EN")
    test_batch_embedding_and_clustering(chinese_sentences, "ZH")