from abc import ABC, abstractmethod
from transformers import AutoModel



class ModelLoaderBase(ABC):
    """
    All children should only use class methods for singleton pattern
    """

    @abstractmethod
    def load_model(
            self,
            model_name: str,
            local_only: bool = False,
    ):
        raise NotImplementedError("need to implement")



class EmbeddingModelLoader(ModelLoaderBase):
    @abstractmethod
    def load_model(
            self,
            model_name: str,
            local_only: bool = False,
    ):
        raise NotImplementedError("need to implement")