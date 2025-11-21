
from typing import TypedDict

class SentenceEmbedding(TypedDict):
    text: str
    embedding: list[float]
