"""Chunking według struktury dokumentu: jeden chunk na sekcję, z metadanymi dokumentu."""

import re
from pathlib import Path

# np. "Status: AKTUALNY | Wersja: 3.0 | Obowiązuje od: 2026-02-01"
STATUS_RE = re.compile(r"^Status:\s*([^|]+)")
INTRO_SECTION = "wstęp"


def _is_heading(line: str, previous: str) -> bool:
    """Nagłówek sekcji to krótka linia bez kropki na końcu, zaczynająca się po pełnym zdaniu."""
    return len(line) <= 60 and not line.endswith((".", ",", ":", ";")) and previous.endswith(".")


def chunk_document(source: str, text: str) -> list[dict]:
    """Dzieli dokument na chunki po sekcjach: słowniki z `id`, `text` i `metadata`.

    Tekst każdego chunku zaczyna się od "<tytuł dokumentu> – <sekcja>", więc jest zrozumiały sam w sobie.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    status_idx = next((i for i, line in enumerate(lines) if STATUS_RE.match(line)), None)
    if status_idx is None:  # dokument bez standardowego nagłówka
        title, status, body = Path(source).stem, "", lines
    else:
        title = lines[status_idx - 1] if status_idx > 0 else Path(source).stem
        status = STATUS_RE.match(lines[status_idx]).group(1).strip()
        body = lines[status_idx + 1 :]
    archived = "ARCHIW" in f"{source} {status}".upper()

    sections: list[tuple[str, list[str]]] = [(INTRO_SECTION, [])]
    previous = "."  # nagłówek dokumentu traktujemy jak zakończone zdanie
    for line in body:
        if _is_heading(line, previous):
            sections.append((line, []))
        else:
            sections[-1][1].append(line)
        previous = line

    non_empty = [(section, " ".join(section_lines)) for section, section_lines in sections if section_lines]
    return [
        {
            "id": f"{source}::{i}",
            "text": f"{title} – {section}\n{content}",
            "metadata": {
                "source": source,
                "chunk_index": i,
                "section": section,
                "status": status,
                "archived": archived,
            },
        }
        for i, (section, content) in enumerate(non_empty)
    ]
