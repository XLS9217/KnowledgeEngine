from typing import Optional, Literal
from transformers import AutoModel
from src.model_objects.model_loader import register_model
from src.model_objects.model_bases import EmbeddingModelBase


@register_model
class JinaEmbeddingsV3(EmbeddingModelBase):

    model_id = "jinaai/jina-embeddings-v3"

    SUPPORTED_TASKS = [
        "retrieval.query",
        "retrieval.passage",
        "separation",
        "classification",
        "text-matching"
    ]

    SUPPORTED_DIMENSIONS = [32, 64, 128, 256, 512, 768, 1024]

    def __init__(self):
        super().__init__()
        self.task: Optional[str] = None
        self.embedding_size: int = 512

    def initialize(self, model_name: str, device: str, model_path: str):
        self.device = device
        self.model = AutoModel.from_pretrained(
            model_name,
            cache_dir=model_path,
            local_files_only=True,
            trust_remote_code=True
        ).to(device)

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
            List representation of embedding with shape (embedding_size,)
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

        return embeddings.tolist()

    def get_embeddings(self, text_list: list[str]):
        """
        Generate embeddings for multiple texts using current task and embedding_size settings.

        Args:
            text_list: List of text strings to encode

        Returns:
            List of dicts with 'text' and 'embedding' keys
        """
        # Prepare model encoding parameters
        encode_kwargs = {
            'truncate_dim': self.embedding_size
        }

        # Only add task parameter if it's set
        if self.task is not None:
            encode_kwargs['task'] = self.task

        # Encode texts
        embeddings = self.model.encode(text_list, **encode_kwargs)

        return [
            {"text": text, "embedding": embedding.tolist()}
            for text, embedding in zip(text_list, embeddings)
        ]


@register_model
class JinaEmbeddingsV2BaseZH(EmbeddingModelBase):
    model_id = "jinaai/jina-embeddings-v2-base-zh"

    def initialize(self, model_name: str, device: str, model_path: str):
        self.device = device
        self.model = AutoModel.from_pretrained(
            model_name,
            cache_dir=model_path,
            local_files_only=True,
            trust_remote_code=True
        ).to(device)

    def get_embedding(self, text: str):
        embeddings = self.model.encode(text)
        return embeddings.tolist()

    def get_embeddings(self, text_list: list[str]):
        embeddings = self.model.encode(text_list)
        return [
            {"text": text, "embedding": embedding.tolist()}
            for text, embedding in zip(text_list, embeddings)
        ]


@register_model
class JinaEmbeddingsV4(EmbeddingModelBase):
    model_id = "jinaai/jina-embeddings-v4"

    def initialize(self, model_name: str, device: str, model_path: str):
        self.device = device
        self.model = AutoModel.from_pretrained(
            model_name,
            cache_dir=model_path,
            local_files_only=True,
            trust_remote_code=True
        ).to(device)

    def get_embedding(self, text: str):
        embeddings = self.model.encode_text(texts=[text], task="text-matching")
        return embeddings[0].tolist()

    def get_embeddings(self, text_list: list[str]):
        embeddings = self.model.encode_text(texts=text_list, task="text-matching")
        return [
            {"text": text, "embedding": embedding.tolist()}
            for text, embedding in zip(text_list, embeddings)
        ]