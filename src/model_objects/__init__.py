# Import ModelLoader and register_model
from src.model_objects.model_loader import ModelLoader, register_model

# Import all model adapters to trigger registration
from src.model_objects.semantic.jina_series_adapters import JinaEmbeddingsV3

__all__ = ["ModelLoader", "register_model", "JinaEmbeddingsV3"]