from PIL.Image import Image
from src.model_objects.model_loader import ModelLoader
from src.algorisms.clustering import k_means as k_means_algo, agglomerative as agglomerative_algo, auto_agglomerative as auto_agglomerative_algo
from src.algorisms.bucketing import similarity_bucketing as similarity_bucketing_algo


class OrchestratorInterface:
    """
    Simplified orchestrator that directly calls model methods without task engines.
    Keeps a minimal compatibility attribute `engine` that points to this class,
    so existing router fallbacks like OrchestratorInterface.engine.embedding_model still work.
    """

    # Backward-compat compatibility handle for routers
    engine = None  # will be set to this class in initialize

    # Direct model references
    embedding_model = None
    reranker_model = None
    clip_model = None

    @classmethod
    def initialize(cls):
        """Initialize and load default models. engine_name is ignored after refactor."""
        # Load default models directly
        cls.embedding_model = ModelLoader.load_model("Qwen/Qwen3-Embedding-0.6B", "cuda")
        cls.reranker_model = ModelLoader.load_model("Qwen/Qwen3-Reranker-0.6B", "cuda")
        cls.clip_model = ModelLoader.load_model("openai/clip-vit-base-patch32", "cuda")
        # Point engine to this class for backward compatibility
        cls.engine = cls

    # Embedding tasks
    @classmethod
    async def get_embedding(cls, text: str):
        if cls.embedding_model is None:
            raise ValueError("No embedding model loaded")
        res = cls.embedding_model.get_embedding(text)
        return {"model_name": getattr(cls.embedding_model, "model_name", None), "result": res}

    @classmethod
    async def get_embeddings(cls, text_list: list[str]):
        if cls.embedding_model is None:
            raise ValueError("No embedding model loaded")
        res = cls.embedding_model.get_embeddings(text_list)
        return {"model_name": getattr(cls.embedding_model, "model_name", None), "result": res}

    # Reranker tasks
    @classmethod
    async def rerank(cls, query: str, documents: list[str], top_k: int):
        if cls.reranker_model is None:
            raise ValueError("No reranker model loaded")
        res = cls.reranker_model.rerank(query, documents, top_k)
        return {"model_name": getattr(cls.reranker_model, "model_name", None), "result": res}

    # CLIP tasks
    @classmethod
    async def get_clip_score(cls, img: Image, text: str):
        if cls.clip_model is None:
            raise ValueError("No CLIP model loaded")
        res = cls.clip_model.get_clip_score(img, text)
        return {"model_name": getattr(cls.clip_model, "model_name", None), "result": res}

    @classmethod
    async def get_clip_scores(cls, img: Image, texts: list[str]):
        if cls.clip_model is None:
            raise ValueError("No CLIP model loaded")
        res = cls.clip_model.get_clip_scores(img, texts)
        return {"model_name": getattr(cls.clip_model, "model_name", None), "result": res}

    # Algorithm tasks - Clustering
    @classmethod
    async def k_means(cls, data: list[dict], n_clusters: int):
        return k_means_algo(data=data, n_clusters=n_clusters)

    @classmethod
    async def agglomerative(cls, data: list[dict], n_clusters: int, linkage: str = "ward"):
        return agglomerative_algo(data=data, n_clusters=n_clusters, linkage=linkage)

    @classmethod
    async def auto_agglomerative(cls, data: list[dict], max_clusters: int, linkage: str = "ward"):
        return auto_agglomerative_algo(data=data, max_clusters=max_clusters, linkage=linkage)

    # Algorithm tasks - Bucketing
    @classmethod
    async def similarity_bucketing(cls, data: list[dict], buckets: list[dict], score_method: str = "cosine",
                                   allow_orphan_bucket: bool = False, orphan_threshold: float = 0.5):
        return similarity_bucketing_algo(
            data=data,
            buckets=buckets,
            score_method=score_method,
            allow_orphan_bucket=allow_orphan_bucket,
            orphan_threshold=orphan_threshold
        )