from abc import ABC, abstractmethod


class EmbeddingModelBase(ABC):

    def __init__(self, device:str , model):
        self.device = device
        self.model = model

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def get_embedding(self, text: str):
        pass


class RerankerModelBase(ABC):

    def __init__(self, device:str , model):
        self.device = device
        self.model = model

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def rerank(self, query:str , document:list[str], top_k:int):
        # return the top k from documents
        pass