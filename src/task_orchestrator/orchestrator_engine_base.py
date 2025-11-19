from abc import ABC, abstractmethod

from src.task_orchestrator.engine_request_struct import LoadRequestStruct, TaskRequestStruct


class OrchestratorEngineBase(ABC):

    def __init__(self):
        self._models = {}

    def start_engine(self, model_list: list[LoadRequestStruct]):
        for request in model_list:
            if request.model_type == "embedding":
                self.load_embedding_model(request.model_name, request.device, request.extra_params)
            elif request.model_type == "reranker":
                self.load_reranker_model(request.model_name, request.device, request.extra_params)
            elif request.model_type == "clip":
                self.load_clip_model(request.model_name, request.device, request.extra_params)

    @abstractmethod
    def load_embedding_model(self, model_name: str, device: str = "cpu", extra_params: dict = None):
        pass

    @abstractmethod
    def load_reranker_model(self, model_name: str, device: str = "cpu", extra_params: dict = None):
        pass

    @abstractmethod
    def load_clip_model(self, model_name: str, device: str = "cpu", extra_params: dict = None):
        pass

    @abstractmethod
    def add_task(self, task: TaskRequestStruct):
        pass