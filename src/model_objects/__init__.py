# Import ModelLoader and register_model
from src.model_objects.model_loader import ModelLoader, register_model

# Import all model adapters to trigger registration
from src.model_objects.semantic.jina_series_adapters import JinaEmbeddingsV3, JinaEmbeddingsV2BaseZH, JinaEmbeddingsV4
from src.model_objects.semantic.qwen_series_adapters import Qwen3Embedding06B, Qwen3Reranker06B
from src.model_objects.semantic.openai_series_adapters import CLIPVITBasePatch32

__all__ = ["ModelLoader", "register_model", "JinaEmbeddingsV3", "JinaEmbeddingsV2BaseZH", "JinaEmbeddingsV4", "Qwen3Embedding06B", "Qwen3Reranker06B", "CLIPVITBasePatch32"]