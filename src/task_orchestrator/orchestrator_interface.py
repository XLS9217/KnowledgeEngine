from PIL.Image import Image
from src.task_orchestrator.engine_structs import TaskRequestStruct, LoadRequestStruct


class OrchestratorInterface:

    engine = None

    @classmethod
    def initialize(cls, engine_name: str):
        if engine_name == "asyncio_ray_engine":
            pass
        elif engine_name == "single_process_engine":
            from src.task_orchestrator.single_process_engine.engine import SingleProcessEngine
            cls.engine = SingleProcessEngine()
            cls.engine.start_engine()

    # Embedding tasks
    @classmethod
    async def get_embedding(cls, text: str):
        return await cls.engine.execute_task(TaskRequestStruct(
            task_type="embedding",
            task_name="get_embedding",
            task_params={"text": text}
        ))

    @classmethod
    async def get_embeddings(cls, text_list: list[str]):
        return await cls.engine.execute_task(TaskRequestStruct(
            task_type="embedding",
            task_name="get_embeddings",
            task_params={"text_list": text_list}
        ))

    # Reranker tasks
    @classmethod
    async def rerank(cls, query: str, documents: list[str], top_k: int):
        return await cls.engine.execute_task(TaskRequestStruct(
            task_type="rerank",
            task_name="rerank",
            task_params={"query": query, "documents": documents, "top_k": top_k}
        ))

    # CLIP tasks
    @classmethod
    async def get_clip_score(cls, img: Image, text: str):
        return await cls.engine.execute_task(TaskRequestStruct(
            task_type="clip",
            task_name="get_clip_score",
            task_params={"img": img, "text": text}
        ))

    @classmethod
    async def get_clip_scores(cls, img: Image, texts: list[str]):
        return await cls.engine.execute_task(TaskRequestStruct(
            task_type="clip",
            task_name="get_clip_scores",
            task_params={"img": img, "texts": texts}
        ))

    # Algorithm tasks - Clustering
    @classmethod
    async def k_means(cls, data: list[dict], n_clusters: int):
        return await cls.engine.execute_task(TaskRequestStruct(
            task_type="algorithm",
            task_name="k_means",
            task_params={"data": data, "n_clusters": n_clusters}
        ))

    @classmethod
    async def agglomerative(cls, data: list[dict], n_clusters: int, linkage: str = "ward"):
        return await cls.engine.execute_task(TaskRequestStruct(
            task_type="algorithm",
            task_name="agglomerative",
            task_params={"data": data, "n_clusters": n_clusters, "linkage": linkage}
        ))

    @classmethod
    async def auto_agglomerative(cls, data: list[dict], max_clusters: int, linkage: str = "ward"):
        return await cls.engine.execute_task(TaskRequestStruct(
            task_type="algorithm",
            task_name="auto_agglomerative",
            task_params={"data": data, "max_clusters": max_clusters, "linkage": linkage}
        ))

    # Algorithm tasks - Bucketing
    @classmethod
    async def similarity_bucketing(cls, data: list[dict], buckets: list[dict], score_method: str = "cosine",
                                    allow_orphan_bucket: bool = False, orphan_threshold: float = 0.5):
        return await cls.engine.execute_task(TaskRequestStruct(
            task_type="algorithm",
            task_name="similarity_bucketing",
            task_params={
                "data": data,
                "buckets": buckets,
                "score_method": score_method,
                "allow_orphan_bucket": allow_orphan_bucket,
                "orphan_threshold": orphan_threshold
            }
        ))