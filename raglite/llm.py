# raglite/llm.py
from openai import OpenAI
from .types import ChatClient
from .config import settings

class OpenAIChat(ChatClient):
    def __init__(self, base_url: str | None = None, api_key: str | None = None):
        self.client = OpenAI(base_url=base_url or settings.base_url,
                             api_key=api_key or settings.api_key)

    def generate(self, system_prompt: str, user_prompt: str, model: str, temperature: float = 0.0) -> str:
        resp = self.client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return resp.choices[0].message.content
