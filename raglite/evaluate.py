# raglite/evaluate.py
from .llm import OpenAIChat

EVAL_SYS = (
    "You are an intelligent evaluation system tasked with assessing the AI assistant's responses. "
    "If the AI assistant's response is very close to the true response, assign a score of 1. "
    "If the response is incorrect or unsatisfactory in relation to the true response, assign a score of 0. "
    "If the response is partially aligned with the true response, assign a score of 0.5."
)

def score(query: str, ai_answer: str, true_answer: str, model: str) -> str:
    evaluation_prompt = (
        f"User Query: {query}\nAI Response:\n{ai_answer}\n"
        f"True Response: {true_answer}\n{EVAL_SYS}"
    )
    chat = OpenAIChat()
    return chat.generate(EVAL_SYS, evaluation_prompt, model, temperature=0.0)
