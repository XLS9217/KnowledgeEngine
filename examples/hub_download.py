from pathlib import Path
import os

# Set cache directory
CACHE_DIR = Path(r"E:\model_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Set HuggingFace cache to use our custom cache directory
# os.environ['HF_HUB_OFFLINE'] = '1'
# os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HOME'] = str(CACHE_DIR)
os.environ["HF_HUB_CACHE"] = str(CACHE_DIR)

from huggingface_hub import snapshot_download

# Download microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank
model_id = "Qwen/Qwen3-Embedding-0.6B"

print(f"Downloading {model_id} to {CACHE_DIR}")
snapshot_path = snapshot_download(
    repo_id=model_id,
    cache_dir=str(CACHE_DIR)
)

print(f"\nModel downloaded to: {snapshot_path}")
print("\nFiles in snapshot:")
for file in Path(snapshot_path).rglob("*"):
    if file.is_file():
        print(f"  - {file.name} ({file.stat().st_size} bytes)")
