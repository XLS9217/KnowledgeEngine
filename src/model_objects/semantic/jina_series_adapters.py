from typing import Optional, Literal, Union, List
from src.model_objects.model_loader import register_model
from src.model_objects.semantic.model_bases import EmbeddingModelBase


@register_model
class JinaEmbeddingsV3(EmbeddingModelBase):

    model_id = "jinaai/jina-embeddings-v3"

    # Supported tasks for LoRA adapters
    SUPPORTED_TASKS = [
        "retrieval.query",
        "retrieval.passage",
        "separation",
        "classification",
        "text-matching"
    ]

    # Supported embedding dimensions (Matryoshka embeddings)
    SUPPORTED_DIMENSIONS = [32, 64, 128, 256, 512, 768, 1024]

    def __init__(self, device: str, model):
        super().__init__(device, model)
        self.task: Optional[str] = None
        self.embedding_size: int = 512

    def set_task(self, task: Optional[Literal["retrieval.query", "retrieval.passage", "separation", "classification", "text-matching"]]):
        """Set the task type for LoRA adapter. Set to None to disable task-specific encoding."""
        if task is not None and task not in self.SUPPORTED_TASKS:
            raise ValueError(
                f"task must be one of {self.SUPPORTED_TASKS} or None, got {task}"
            )
        self.task = task

    def set_embedding_size(self, size: int):
        """Set the output embedding dimension (Matryoshka embeddings)."""
        if size not in self.SUPPORTED_DIMENSIONS:
            raise ValueError(
                f"embedding_size must be one of {self.SUPPORTED_DIMENSIONS}, got {size}"
            )
        self.embedding_size = size

    def get_embedding(self, text: str):
        """
        Generate embeddings for input text using current task and embedding_size settings.

        Args:
            text: Single text string to encode

        Returns:
            Embeddings with shape (embedding_size,)
        """
        # Prepare model encoding parameters
        encode_kwargs = {
            'truncate_dim': self.embedding_size
        }

        # Only add task parameter if it's set
        if self.task is not None:
            encode_kwargs['task'] = self.task

        # Encode text
        embeddings = self.model.encode(text, **encode_kwargs)

        return embeddings


@register_model
class JinaEmbeddingsV2BaseZH(EmbeddingModelBase):
    model_id = "jinaai/jina-embeddings-v2-base-zh"

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