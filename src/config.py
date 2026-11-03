"""Konfiguracja: modele, ścieżki i parametry pipeline'u.

Adres Ollamy biblioteka `ollama` bierze ze zmiennej OLLAMA_HOST (domyślnie http://127.0.0.1:11434).
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = PROJECT_ROOT / "documents"
CHROMA_DIR = PROJECT_ROOT / "chroma_data"
COLLECTION_NAME = "documents"

LLM_MODEL = os.getenv("LLM_MODEL", "qwen3:4b")
EMBED_MODEL = "bge-m3"  # zmiana modelu wymaga `python main.py reset` + `add`

TOP_K = 5

NUM_CTX = 8192
TEMPERATURE = 0.2
SEED = 42
# qwen3:4b (wariant Thinking) zwykle potrzebuje ok. 1300 tokenów na rozumowanie i odpowiedź,
# ale czasem zapętla się w rozumowaniu – limit ucina takie przypadki.
NUM_PREDICT = 3000
