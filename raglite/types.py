# raglite/types.py
from typing import Protocol, List
from dataclasses import dataclass

@dataclass
class EmbeddingResult:
    vectors: List[list]  # list of embedding vectors (len == n_texts)

class EmbeddingsClient(Protocol):
    def embed(self, texts: List[str], model: str) -> EmbeddingResult: ...

class ChatClient(Protocol):
    def generate(self, system_prompt: str, user_prompt: str, model: str, temperature: float = 0.0) -> str: ...
