from src.task_orchestrator.orchestrator_engine_base import OrchestratorEngineBase


class SingleProcessEngine(OrchestratorEngineBase):

    def __init__(self):
        super().__init__()
