import requests
from pathlib import Path


BASE_URL = "http://localhost:7009/api/v1"


def test_clip_http():
    """Test CLIP API with image and multiple texts."""
    print(f"\n{'='*60}")
    print(f"Testing CLIP API")
    print(f"{'='*60}")

    try:
        # Load image
        image_path = Path(__file__).parent.parent / "page.png"

        if not image_path.exists():
            print(f"Error: Image not found at {image_path}")
            return

        # Test texts
        texts = [
            "a photo of a cat",
            "a photo of a dog",
            "a document with text",
            "a webpage full of text",
        ]

        # Prepare multipart form data
        with open(image_path, 'rb') as img_file:
            files = {'image': img_file}
            data = {'texts': ','.join(texts)}

            response = requests.post(
                f"{BASE_URL}/clip/scores",
                files=files,
                data=data
            )

        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.text}")
            return

        results = response.json()["scores"]

        # Print results
        print("Image-Text Similarity Scores:")
        for text, score in results:
            print(f"  {text:30s} -> {score:.4f}")

        # Best match
        best = max(results, key=lambda x: x[1])
        print(f"\nBest match: '{best[0]}' with {best[1]:.4f}")
        print(f"\nTest PASSED")

    except Exception as e:
        print(f"Test FAILED")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Make sure the server is running on http://localhost:7009")
    print("Start server with: python main.py")

    test_clip_http()