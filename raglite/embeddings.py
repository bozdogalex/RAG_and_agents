# raglite/embeddings.py
from typing import List
from openai import OpenAI
from .types import EmbeddingsClient, EmbeddingResult
from .config import settings

class OpenAIEmbeddings(EmbeddingsClient):
    def __init__(self, base_url: str | None = None, api_key: str | None = None):
        self.client = OpenAI(base_url=base_url or settings.base_url,
                             api_key=api_key or settings.api_key)

    def embed(self, texts: List[str], model: str) -> EmbeddingResult:
        # Batch once; for huge corpora you’d chunk requests here.
        resp = self.client.embeddings.create(model=model, input=texts)
        vectors = [d.embedding for d in resp.data]
        return EmbeddingResult(vectors=vectors)
