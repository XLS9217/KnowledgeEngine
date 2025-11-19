from abc import ABC , abstractmethod

class OrchestratorEngineBase(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def start_engine(self, model_list:list[str]):
        pass

    @abstractmethod
    def load_embedding_model(self, model_name:str , **kwargs):
        pass

    @abstractmethod
    def load_reranker_model(self, model_name: str, **kwargs):
        pass

    @abstractmethod
    def load_clip_model(self, model_name: str, **kwargs):
        pass