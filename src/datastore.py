"""Trwała kolekcja Chroma przechowująca chunki dokumentów."""

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings
from chromadb.errors import NotFoundError

from src.config import CHROMA_DIR, COLLECTION_NAME


def _client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False))


def get_collection() -> Collection:
    """Otwiera (lub tworzy) kolekcję; embeddingi zawsze przekazujemy jawnie."""
    return _client().get_or_create_collection(
        name=COLLECTION_NAME, configuration={"hnsw": {"space": "cosine"}}, embedding_function=None
    )


def reset_collection() -> bool:
    """Usuwa kolekcję. Zwraca False, jeśli nie istniała."""
    try:
        _client().delete_collection(COLLECTION_NAME)
    except (NotFoundError, ValueError):
        return False
    return True


def upsert_chunks(chunks: list[dict], embeddings: list[list[float]]) -> None:
    """Zapisuje chunki z embeddingami; upsert po ID, więc ponowne dodanie pliku nie tworzy duplikatów."""
    get_collection().upsert(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
        embeddings=embeddings,
    )
