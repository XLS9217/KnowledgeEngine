from pydantic import BaseModel


class LoadRequestStruct(BaseModel):
    model_name : str
    model_type: str
    device : str
    extra_params : dict