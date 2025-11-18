from abc import ABC, abstractmethod


class EmbeddingModelBase(ABC):

    def __init__(self, device:str , model):
        self.device = device
        self.model = model

    @abstractmethod
    def get_embedding(self, text: str):
        pass


class RerankerModelBase(ABC):

    def __init__(self, device:str , model):
        self.device = device
        self.model = model

    @abstractmethod
    def rerank(self, text_list: list[str]):
        pass