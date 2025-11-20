import asyncio
from concurrent.futures import ThreadPoolExecutor
from src.task_orchestrator.engine_structs import TaskRequestStruct, LoadRequestStruct
from src.task_orchestrator.orchestrator_engine_base import OrchestratorEngineBase
from src.model_objects.model_loader import ModelLoader


class SingleProcessEngine(OrchestratorEngineBase):

    def __init__(self):
        super().__init__()
        self.engine_name = "single_process_engine"
        self._executor = ThreadPoolExecutor(max_workers=1)
        self.embedding_model = None
        self.reranker_model = None
        self.clip_model = None

    def _engine_init(self):
        """Load default models for single process engine"""
        self.embedding_model = ModelLoader.load_model("jinaai/jina-embeddings-v3", "cuda")
        self.reranker_model = ModelLoader.load_model("Qwen/Qwen3-Reranker-0.6B", "cuda")
        self.clip_model = ModelLoader.load_model("openai/clip-vit-base-patch32", "cuda")

    def load_model(self, model_request: LoadRequestStruct):
        """Load a model based on the request"""
        model = ModelLoader.load_model(model_request.model_name, model_request.device)

        if model_request.model_type == "embedding":
            self.embedding_model = model
        elif model_request.model_type == "reranker":
            self.reranker_model = model
        elif model_request.model_type == "clip":
            self.clip_model = model

        return model

    async def execute_task(self, task: TaskRequestStruct):
        """Execute task asynchronously by running blocking model call in thread pool"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._execute_sync,
            task
        )

    def _execute_sync(self, task: TaskRequestStruct):
        """Synchronously execute task by routing to appropriate model method"""
        # Route to appropriate model based on task_type
        if task.task_type == "embedding":
            if self.embedding_model is None:
                raise ValueError("No embedding model loaded")
            return self.embedding_model.get_embedding(task.task_params["text"])

        elif task.task_type == "reranker":
            if self.reranker_model is None:
                raise ValueError("No reranker model loaded")
            return self.reranker_model.rerank(
                task.task_params["query"],
                task.task_params["documents"],
                task.task_params["top_k"]
            )

        elif task.task_type == "clip":
            if self.clip_model is None:
                raise ValueError("No CLIP model loaded")

            if task.task_name == "get_clip_score":
                return self.clip_model.get_clip_score(
                    task.task_params["img"],
                    task.task_params["text"]
                )
            elif task.task_name == "get_clip_scores":
                return self.clip_model.get_clip_scores(
                    task.task_params["img"],
                    task.task_params["texts"]
                )
            else:
                raise ValueError(f"Unknown CLIP task_name: {task.task_name}")

        else:
            raise ValueError(f"Unknown task_type: {task.task_type}")
