import requests


BASE_URL = "http://localhost:7009/api/v1"


def test_similarity_bucketing(sentences: list[str], bucket_topics: list[str], lang: str):
    """Test similarity bucketing API."""
    print(f"\n{'='*70}")
    print(f"Testing Similarity Bucketing API ({lang})")
    print(f"{'='*70}")

    try:
        print(f"\n1. Getting embeddings for {len(sentences)} sentences...")
        print("-" * 70)

        # Get embeddings for sentences
        response = requests.post(
            f"{BASE_URL}/embeddings",
            json={"text_list": sentences}
        )

        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.text}")
            return

        embeddings_result = response.json()["embeddings"]
        data = [
            {"text": item["text"], "embedding": item["embedding"]}
            for item in embeddings_result
        ]

        print(f"Got {len(data)} sentence embeddings\n")

        print(f"2. Getting embeddings for {len(bucket_topics)} bucket topics...")
        print("-" * 70)

        # Get embeddings for bucket topics
        response = requests.post(
            f"{BASE_URL}/embeddings",
            json={"text_list": bucket_topics}
        )

        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.text}")
            return

        bucket_embeddings_result = response.json()["embeddings"]
        buckets = [
            {"text": item["text"], "embedding": item["embedding"]}
            for item in bucket_embeddings_result
        ]

        print(f"Got {len(buckets)} bucket embeddings\n")

        print(f"3. Running similarity bucketing...")
        print("-" * 70)

        # Run similarity bucketing
        response = requests.post(
            f"{BASE_URL}/algorithm/similarity-bucketing",
            json={
                "data": data,
                "buckets": buckets,
                "score_method": "cosine",
                "allow_orphan_bucket": True,
                "orphan_threshold": 0.1
            }
        )

        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.text}")
            return

        results = response.json()["result"]

        # Print results
        print("\nBucketing Results:")
        print("=" * 70)
        for bucket in results:
            bucket_name = bucket["bucket"]
            bucket_sentences = bucket["sentences"]

            if bucket_name == "orphan":
                print(f"\n[Orphan Bucket] - {len(bucket_sentences)} sentences:")
            else:
                print(f"\n[{bucket_name[:50]}...] - {len(bucket_sentences)} sentences:")

            for sentence in bucket_sentences:
                print(f"  - {sentence}")

        print(f"\n{'='*70}")
        print(f"Test PASSED")
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
        "The product quality is excellent",
        "Great customer service experience",
        "Very satisfied with the purchase",
        "The delivery was very slow",
        "Terrible shipping experience",
        "Package arrived damaged",
        "The price is too high",
        "Not worth the money",
        "Overpriced for what you get",
    ]

    english_bucket_topics = [
        "positive feedback about product quality and service",
        "negative feedback about shipping and delivery",
        "complaints about pricing and value",
    ]

    # Chinese test data
    chinese_sentences = [
        "产品质量非常好",
        "客户服务体验很棒",
        "对购买非常满意",
        "送货速度太慢了",
        "物流体验很糟糕",
        "包裹到达时已损坏",
        "价格太高了",
        "不值这个价钱",
        "性价比太低",
    ]

    chinese_bucket_topics = [
        "关于产品质量和服务的正面反馈",
        "关于物流配送的负面反馈",
        "关于价格和性价比的投诉",
    ]

    test_similarity_bucketing(english_sentences, english_bucket_topics, "EN")
    test_similarity_bucketing(chinese_sentences, chinese_bucket_topics, "ZH")