


class OrchestratorInterface:

    engine =  None

    @classmethod
    def initialize(cls, engine_name:str):

        #according to folders, import by if
        if engine_name == "asyncio_ray_engine":
            pass

        elif engine_name == "single_process_engine":
            from src.task_orchestrator.single_process_engine.engine import SingleProcessEngine
            cls.engine = SingleProcessEngine()
            pass

