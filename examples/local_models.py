from pathlib import Path
import os
import torch
from transformers import AutoModel
from numpy.linalg import norm

CACHE_DIR = Path(__file__).parent.parent / "model_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HOME'] = str(CACHE_DIR)
os.environ['HF_HUB_CACHE'] = str(CACHE_DIR)

model_id = "jinaai/jina-embeddings-v4"
print(f"Loading {model_id} from local cache: {CACHE_DIR}")
model = AutoModel.from_pretrained(
    model_id,
    cache_dir=str(CACHE_DIR),
    trust_remote_code=True,
    local_files_only=True,
    dtype=torch.bfloat16
)

# Use encode_text method with task parameter
text = "hello world"
print(f"\nEmbedding text: '{text}'")

# Encode using text-matching task
embeddings = model.encode_text(
    texts=[text],
    task="text-matching"
)

print(f"Embedding : {embeddings}")
