from abc import ABC, abstractmethod


class EmbeddingModelBase(ABC):
    @abstractmethod
    def get_embedding(self, text: str):
        pass


class RerankerModelBase(ABC):
    @abstractmethod
    def rerank(self, text_list: list[str]):
        pass