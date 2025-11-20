
from typing import TypedDict

class SentenceEmbedding(TypedDict):
    sentence: str
    embedding: list[float]
