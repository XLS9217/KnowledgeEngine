from pathlib import Path
from PIL import Image
from src.model_objects.model_loader import ModelLoader

model = ModelLoader.load_model(
    "openai/clip-vit-base-patch32",
    device="cuda"
)

# Load image
image_path = Path(__file__).parent.parent / "page.png"
image = Image.open(image_path)

# Test texts
texts = [
    "a photo of a cat",
    "a photo of a dog",
    "a document with text",
    "a webpage full of text",
]

# Get scores
results = model.get_clip_scores(image, texts)

print("Image-Text Similarity Scores:")
for text, score in results:
    print(f"  {text:30s} -> {score:.4f}")

# Best match
best = max(results, key=lambda x: x[1])
print(f"\nBest match: '{best[0]}' with {best[1]:.4f}")
