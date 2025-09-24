# raglite/chunking.py
def chunk_text(text: str, n: int, overlap: int) -> list[str]:
    assert n > 0 and 0 <= overlap < n
    step = n - overlap
    return [text[i:i+n] for i in range(0, max(len(text), 1), step)]
