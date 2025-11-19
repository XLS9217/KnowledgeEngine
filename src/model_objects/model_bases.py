from abc import ABC, abstractmethod
from PIL.Image import Image


class EmbeddingModelBase(ABC):

    def __init__(self):
        self.type = "embedding"
        self.device = None
        self.model = None
        self.tokenizer = None

    @abstractmethod
    def initialize(self, model_name:str, device:str, model_path:str):
        pass

    @abstractmethod
    def get_embedding(self, text: str):
        pass


class RerankerModelBase(ABC):

    def __init__(self):
        self.type = "reranker"
        self.device = None
        self.model = None
        self.tokenizer = None

    @abstractmethod
    def initialize(self, model_name:str, device:str, model_path:str):
        pass

    @abstractmethod
    def rerank(self, query:str , document:list[str], top_k:int):
        # return the top k from documents
        pass

class CLIPModelBase(ABC):

    def __init__(self):
        self.type = "clip"
        self.device = None
        self.model = None
        self.processor = None

    @abstractmethod
    def initialize(self, model_name: str, device: str, model_path: str):
        pass

    @abstractmethod
    def get_clip_score(self, img: Image , text: str):
        pass

    @abstractmethod
    def get_clip_scores(self, img: Image, texts: list[str]):
        pass