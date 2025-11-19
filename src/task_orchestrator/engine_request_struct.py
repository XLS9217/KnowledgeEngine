from pydantic import BaseModel


class LoadRequestStruct(BaseModel):
    model_name : str
    model_type: str
    device : str
    extra_params : dict


class TaskRequestStruct(BaseModel):
    task_type: str # rerank? clip? embedding?
    task_name: str
    task_params : dict