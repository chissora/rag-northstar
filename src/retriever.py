"""Wyszukiwanie fragmentów najbliższych pytaniu, z pominięciem dokumentów archiwalnych."""

from src.config import TOP_K
from src.datastore import get_collection
from src.embedder import embed_texts


def retrieve(question: str, top_k: int = TOP_K) -> list[dict]:
    """Zwraca do `top_k` chunków z `id`, `text`, `source`, `section` i `score` (podobieństwo cosinusowe)."""
    collection = get_collection()
    if collection.count() == 0:
        return []

    result = collection.query(
        query_embeddings=embed_texts([question]),
        n_results=top_k,
        where={"archived": False},  # nieaktualne zasady nie mogą konkurować z obowiązującymi
    )
    return [
        {"id": chunk_id, "text": text, "source": meta["source"], "section": meta["section"], "score": 1 - distance}
        for chunk_id, text, meta, distance in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
        )
    ]
