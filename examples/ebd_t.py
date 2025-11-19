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


print(DeviceWatcher.get_available_device())

model = ModelLoader.load_model(
    "jinaai/jina-embeddings-v3",
    device="cuda"
)

ebd1 =  model.get_embedding("hello world")

ebd2 =  model.get_embedding("你好世界")
print(cosine_similarity(ebd1 , ebd2))

ebd3 =  model.get_embedding("傻逼玩意儿")
print(cosine_similarity(ebd3 , ebd2))



