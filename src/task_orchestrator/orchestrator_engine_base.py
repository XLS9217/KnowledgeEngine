from abc import ABC, abstractmethod

from src.task_orchestrator.engine_structs import LoadRequestStruct, TaskRequestStruct


class OrchestratorEngineBase(ABC):

    def __init__(self):
        self.engine_name = "base_engine"

    def start_engine(self):
        """Initialize engine and load default models"""
        self._engine_init()

    @abstractmethod
    def _engine_init(self):
        """Engine-specific initialization - load models here"""
        pass

    @abstractmethod
    def load_model(self, model_request: LoadRequestStruct):
        """Load a single model based on request"""
        pass

    @abstractmethod
    async def execute_task(self, task: TaskRequestStruct):
        pass