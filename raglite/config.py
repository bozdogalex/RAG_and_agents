# raglite/config.py
from pydantic import BaseSettings

class Settings(BaseSettings):
    base_url: str = "https://api.studio.nebius.com/v1/"
    api_key: str
    embed_model: str = "BAAI/bge-en-icl"
    chat_model: str = "meta-llama/Llama-3.2-3B-Instruct"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 5
    class Config:
        env_prefix = "OPENAI_"
        env_file = ".env"

settings = Settings()
