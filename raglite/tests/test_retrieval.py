# tests/test_retrieval.py
import numpy as np
from raglite.retrieval import cosine_similarity, top_k_indices
def test_cosine():
    assert cosine_similarity(np.array([1,0]), np.array([1,0])) == 1.0
def test_topk():
    q = [1,0]
    docs = [[1,0],[0,1],[0.9,0.1]]
    idxs = top_k_indices(q, docs, 2)
    assert idxs[0] == 0
