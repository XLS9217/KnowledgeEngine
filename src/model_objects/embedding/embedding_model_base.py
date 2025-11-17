from abc import ABC, abstractmethod

class EmbeddingModelBase(ABC):

    def __init__(
            self,
            model_name: str,
    ):
        pass

    @abstractmethod
    def get_embedding(self, text: str):
        pass


class CustomModelAdapter(EmbeddingModelBase):
    """
    Do not implement yet
    """
    def get_embedding(self, text: str):
        pass


