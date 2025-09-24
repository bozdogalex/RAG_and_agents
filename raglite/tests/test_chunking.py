# raglite/tests/test_chunking.py
from raglite.chunking import chunk_text
def test_chunking_overlap():
    text = "abcd" * 300
    chunks = chunk_text(text, 1000, 200)
    assert chunks
    assert all(len(c) <= 1000 for c in chunks)
