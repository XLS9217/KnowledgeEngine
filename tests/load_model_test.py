import os

# os.environ['HF_HUB_OFFLINE'] = '1'
# os.environ['TRANSFORMERS_OFFLINE'] = '1'

from src.model_objects.model_loader import ModelLoader
from src.utils.device_watcher import DeviceWatcher


print(DeviceWatcher.get_available_device())

model = ModelLoader.load_model("jinaai/jina-embeddings-v3" , device="cuda",force_local_only = False )

ebd1 =  model.get_embedding("hello world")
print(ebd1)
ebd2 =  model.get_embedding("你好世界")
print(ebd2)


