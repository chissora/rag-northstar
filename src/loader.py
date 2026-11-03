"""Wczytywanie dokumentów PDF jako tekstu."""

from pathlib import Path

from pypdf import PdfReader


def load_documents(directory: Path) -> list[tuple[str, str]]:
    """Zwraca pary (nazwa_pliku, tekst) dla plików PDF z katalogu; inne pliki pomija."""
    documents = []
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() != ".pdf":
            print(f"Pomijam {path.name}: obsługiwane są tylko pliki PDF")
            continue
        text = "\n".join(page.extract_text() for page in PdfReader(path).pages).strip()
        if not text:
            print(f"Pomijam {path.name}: brak warstwy tekstowej (np. skan)")
            continue
        documents.append((path.name, text))
    return documents
