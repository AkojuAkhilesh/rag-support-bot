from typing import List

def simple_chunk(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    """
    Split text into overlapping chunks so retrieval works on long docs.
    For your small FAQ, this will likely produce 1 chunk—totally fine.
    """
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = end - overlap
    return chunks
