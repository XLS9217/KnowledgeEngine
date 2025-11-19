from PIL.Image import Image
from src.task_orchestrator.engine_request_struct import TaskRequestStruct


class OrchestratorInterface:

    engine = None

    @classmethod
    def initialize(cls, engine_name: str):
        if engine_name == "asyncio_ray_engine":
            pass
        elif engine_name == "single_process_engine":
            from src.task_orchestrator.single_process_engine.engine import SingleProcessEngine
            cls.engine = SingleProcessEngine()

    # Embedding tasks
    @classmethod
    def get_embedding(cls, text: str):
        return cls.engine.add_task(TaskRequestStruct(
            task_type="embedding",
            task_name="get_embedding",
            task_params={"text": text}
        ))

    # Reranker tasks
    @classmethod
    def rerank(cls, query: str, documents: list[str], top_k: int):
        return cls.engine.add_task(TaskRequestStruct(
            task_type="reranker",
            task_name="rerank",
            task_params={"query": query, "documents": documents, "top_k": top_k}
        ))

    # CLIP tasks
    @classmethod
    def get_clip_score(cls, img: Image, text: str):
        return cls.engine.add_task(TaskRequestStruct(
            task_type="clip",
            task_name="get_clip_score",
            task_params={"img": img, "text": text}
        ))

    @classmethod
    def get_clip_scores(cls, img: Image, texts: list[str]):
        return cls.engine.add_task(TaskRequestStruct(
            task_type="clip",
            task_name="get_clip_scores",
            task_params={"img": img, "texts": texts}
        ))