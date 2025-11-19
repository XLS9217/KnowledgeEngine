from abc import ABC , abstractmethod

class OrchestratorEngineBase(ABC):

    def __init__(self):
        pass

    def start_engine(self, **kwargs):
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