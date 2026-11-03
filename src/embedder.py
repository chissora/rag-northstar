"""Embeddingi liczone lokalnie przez Ollamę (bge-m3)."""

import ollama

from src.config import EMBED_MODEL


def embed_texts(texts: list[str]) -> list[list[float]]:
    return ollama.embed(model=EMBED_MODEL, input=texts).embeddings
