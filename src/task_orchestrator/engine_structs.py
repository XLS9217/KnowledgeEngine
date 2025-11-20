from typing import Literal
from pydantic import BaseModel

# Define valid model/task types
ModelType = Literal["embedding", "reranker", "clip"]
TaskType = Literal["embedding", "rerank", "clip" , "algorithm"]


class LoadRequestStruct(BaseModel):
    model_name: str
    model_type: ModelType
    device: str
    extra_params: dict


class TaskRequestStruct(BaseModel):
    task_type: ModelType  # Must be one of: "embedding", "reranker", "clip"
    task_name: str
    task_params: dict

