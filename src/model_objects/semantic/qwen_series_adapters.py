from src.model_objects import register_model
from src.model_objects.semantic.model_bases import EmbeddingModelBase


@register_model
class Qwen3Embedding06B(EmbeddingModelBase):
    model_id = "Qwen/Qwen3-Embedding-0.6B"

    def get_embedding(self, text: str):
        """
        Generate embeddings for input text.

        Args:
            text: Single text string to encode

        Returns:
            Embeddings vector
        """
        embeddings = self.model.encode(text)
        return embeddings