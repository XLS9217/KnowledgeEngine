from src.model_objects.model_loader import ModelLoader
from src.utils.device_watcher import DeviceWatcher


print(DeviceWatcher.get_available_device())

ModelLoader.load_model("jinaai/jina-embeddings-v3" , device="cuda")