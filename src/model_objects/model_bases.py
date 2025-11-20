from abc import ABC, abstractmethod
from typing import List

from PIL.Image import Image

class ModelBase(ABC):

    def __init__(self):
        self.type = "model"
        self.device = None
        self.model = None

    @abstractmethod
    def initialize(self, model_name:str, device:str, model_path:str):
        pass

class EmbeddingModelBase(ModelBase):

    def __init__(self):
        super().__init__()
        self.type = "embedding"
        self.tokenizer = None

    @abstractmethod
    def get_embedding(self, text: str):
        pass

    @abstractmethod
    def get_embeddings(self, text_list: List[str]) -> List[dict]:
        """
        Generate embeddings for multiple texts.

        Args:
            text_list: List of text strings to encode

        Returns:
            List of dicts with 'text' and 'embedding' keys
        """
        pass


class RerankerModelBase(ModelBase):

    def __init__(self):
        super().__init__()
        self.type = "reranker"
        self.tokenizer = None

    @abstractmethod
    def rerank(self, query:str , document:list[str], top_k:int):
        # return the top k from documents
        pass

class CLIPModelBase(ModelBase):

    def __init__(self):
        super().__init__()
        self.type = "clip"
        self.processor = None

    @abstractmethod
    def get_clip_score(self, img: Image , text: str):
        pass

    @abstractmethod
    def get_clip_scores(self, img: Image, texts: list[str]):
        pass