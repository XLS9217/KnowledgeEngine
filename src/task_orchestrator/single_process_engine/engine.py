from src.task_orchestrator.engine_request_struct import TaskRequestStruct
from src.task_orchestrator.orchestrator_engine_base import OrchestratorEngineBase
from src.model_objects.model_loader import ModelLoader


class SingleProcessEngine(OrchestratorEngineBase):

    def __init__(self):
        super().__init__()

    def load_embedding_model(self, model_name: str, device: str = "cpu", extra_params: dict = None):
        model = ModelLoader.load_model(model_name, device)
        self._models[model_name] = model
        return model

    def load_reranker_model(self, model_name: str, device: str = "cpu", extra_params: dict = None):
        model = ModelLoader.load_model(model_name, device)
        self._models[model_name] = model
        return model

    def load_clip_model(self, model_name: str, device: str = "cpu", extra_params: dict = None):
        model = ModelLoader.load_model(model_name, device)
        self._models[model_name] = model
        return model

    def add_task(self, task: TaskRequestStruct):
        pass
