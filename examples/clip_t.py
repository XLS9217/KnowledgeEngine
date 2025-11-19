from pathlib import Path
import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Setup cache
CACHE_DIR = Path(r"E:\model_cache")
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HOME'] = str(CACHE_DIR)
os.environ['HF_HUB_CACHE'] = str(CACHE_DIR)

# Load model and processor
model_id = "openai/clip-vit-base-patch32"

print(f"Loading {model_id} from {CACHE_DIR}")
model = CLIPModel.from_pretrained(
    model_id,
    cache_dir=str(CACHE_DIR),
    local_files_only=True
)
processor = CLIPProcessor.from_pretrained(
    model_id,
    cache_dir=str(CACHE_DIR),
    local_files_only=True
)

# Load an image
image_path = Path(__file__).parent / "page.png"
image = Image.open(image_path)

# Define text descriptions
texts = [
    "a photo of a cat",
    "a photo of a dog",
    "a document with text",
    "a webpage full of text",
]

# Process inputs
inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)

# Get predictions
with torch.no_grad():
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # image-text similarity scores
    probs = logits_per_image.softmax(dim=1)  # convert to probabilities

# Print results
print("\nImage-Text Similarity Scores:")
for text, prob in zip(texts, probs[0]):
    print(f"{text:30s} -> {prob.item():.4f}")

# Find best match
best_idx = probs.argmax().item()
print(f"\nBest match: '{texts[best_idx]}' with {probs[0][best_idx].item():.4f} confidence")

# You can also get embeddings
print("\n--- Getting Embeddings ---")
with torch.no_grad():
    image_features = model.get_image_features(pixel_values=inputs.pixel_values)
    text_features = model.get_text_features(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)

print(f"Image embedding shape: {image_features.shape}")
print(f"Text embeddings shape: {text_features.shape}")

# Compute similarity manually
image_features = image_features / image_features.norm(dim=-1, keepdim=True)
text_features = text_features / text_features.norm(dim=-1, keepdim=True)
similarity = (image_features @ text_features.T)

print(f"\nManual similarity scores: {similarity[0].tolist()}")
