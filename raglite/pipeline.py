# raglite/pipeline.py
from .config import settings
from .logging import get_logger
from .pdf_utils import extract_text_from_pdf
from .chunking import chunk_text
from .embeddings import OpenAIEmbeddings
from .retrieval import top_k_indices
from .llm import OpenAIChat

LOG = get_logger()

SYSTEM_PROMPT_STRICT = (
    "You are an AI assistant that strictly answers based on the given context. "
    "If the answer cannot be derived directly from the provided context, respond with: "
    "'I do not have enough information to answer that.'"
)

def build_corpus(pdf_path: str):
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
    LOG.info("Corpus: %d chunks", len(chunks))
    return chunks

def embed_corpus(chunks: list[str], embed_model: str):
    emb = OpenAIEmbeddings()
    res = emb.embed(chunks, embed_model)
    return res.vectors

def retrieve(query: str, chunks: list[str], chunk_vecs: list[list[float]], k: int):
    emb = OpenAIEmbeddings()
    qvec = emb.embed([query], settings.embed_model).vectors[0]
    idxs = top_k_indices(qvec, chunk_vecs, k)
    return [chunks[i] for i in idxs]

def answer(query: str, contexts: list[str], chat_model: str, temperature: float = 0.0):
    user_prompt = "\n".join(
        [f"Context {i+1}:\n{c}\n{'='*37}\n" for i, c in enumerate(contexts)]
    ) + f"\nQuestion: {query}"
    chat = OpenAIChat()
    return chat.generate(SYSTEM_PROMPT_STRICT, user_prompt, chat_model, temperature)
