from src.model_objects.model_loader import register_model
from src.model_objects.semantic.model_bases import EmbeddingModelBase


@register_model
class JinaEmbeddingsV3(EmbeddingModelBase):

    model_id = "jinaai/jina-embeddings-v3"

    def __init__(self , device:str):
        self.device = device
        pass

    def get_embedding(self, text: str):
        pass