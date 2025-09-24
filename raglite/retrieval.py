# raglite/retrieval.py
import numpy as np
from typing import List, Tuple

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b / (na * nb))

def top_k_indices(query_vec: list[float], doc_vecs: List[list[float]], k: int) -> List[int]:
    q = np.asarray(query_vec, dtype=np.float32)
    sims: List[Tuple[int, float]] = []
    for i, v in enumerate(doc_vecs):
        sims.append((i, cosine_similarity(q, np.asarray(v, dtype=np.float32))))
    sims.sort(key=lambda t: t[1], reverse=True)
    return [i for i, _ in sims[:k]]
