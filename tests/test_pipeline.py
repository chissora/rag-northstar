"""Testy deterministycznych części pipeline'u, bez Ollamy. Jakość odpowiedzi LLM mierzy eval/run_eval.py."""

from types import SimpleNamespace

from pypdf import PdfWriter

import src.datastore as datastore
import src.generator as generator
import src.retriever as retriever
from src import config
from src.chunker import chunk_document
from src.loader import load_documents


def test_documents_split_into_sections():
    chunks = [c for source, text in load_documents(config.DOCUMENTS_DIR) for c in chunk_document(source, text)]

    assert len(chunks) == 37
    assert {c["metadata"]["source"] for c in chunks if c["metadata"]["archived"]} == {
        "03_podroze_sluzbowe_v1_ARCHIWUM.pdf"
    }
    remote_work = [c for c in chunks if c["metadata"]["source"] == "01_praca_zdalna_v2.pdf"]
    assert [c["metadata"]["section"] for c in remote_work] == ["Zasady ogólne", "Praca zdalna z zagranicy", "Bezpieczeństwo"]
    assert remote_work[0]["text"].startswith("Polityka pracy zdalnej – Zasady ogólne\n")


def test_document_without_header():
    text = "Wprowadzenie do zasad.\nKrótka sekcja\nTreść sekcji, złamana\nna dwie linie."

    chunks = chunk_document("zasady_ARCHIWUM.pdf", text)

    assert [c["metadata"]["section"] for c in chunks] == ["wstęp", "Krótka sekcja"]
    assert chunks[1]["text"] == "zasady_ARCHIWUM – Krótka sekcja\nTreść sekcji, złamana na dwie linie."
    assert all(c["metadata"]["archived"] for c in chunks)  # z nazwy pliku


def test_loader_reads_only_pdfs_with_text(tmp_path):
    (tmp_path / "notatka.txt").write_text("Treść.", encoding="utf-8")
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)  # jak skan: brak warstwy tekstowej
    writer.write(tmp_path / "skan.pdf")

    assert load_documents(tmp_path) == []


def test_retriever_skips_archived_chunks(tmp_path, monkeypatch):
    monkeypatch.setattr(datastore, "CHROMA_DIR", tmp_path)
    monkeypatch.setattr(retriever, "embed_texts", lambda texts: [[1.0, 0.0]])
    chunks = [
        {"id": f"{name}::0", "text": name,
         "metadata": {"source": name, "chunk_index": 0, "section": "Hotele", "status": "", "archived": archived}}
        for name, archived in [("v3.pdf", False), ("v1.pdf", True)]
    ]
    datastore.upsert_chunks(chunks, embeddings=[[1.0, 0.0], [1.0, 0.0]])

    assert [h["id"] for h in retriever.retrieve("limit hotelu")] == ["v3.pdf::0"]


def _fake_llm(monkeypatch, content: str, done_reason: str = "stop") -> None:
    response = SimpleNamespace(done_reason=done_reason, message=SimpleNamespace(content=content))
    monkeypatch.setattr(generator.ollama, "chat", lambda **kwargs: response)


def test_reasoning_is_removed_from_answer(monkeypatch):
    _fake_llm(monkeypatch, "rozważam kontekst...</think>\n\n800 PLN.\nŹródła: a.pdf (Hotele)")
    assert generator.generate_answer("Jaki limit?", []) == "800 PLN.\nŹródła: a.pdf (Hotele)"


def test_refusal_never_cites_sources(monkeypatch):
    _fake_llm(monkeypatch, f"...</think>{generator.REFUSAL}\nŹródła: a.pdf (Hotele)")
    assert generator.generate_answer("Telefon do prezesa?", []) == generator.REFUSAL


def test_token_limit_gives_clear_message(monkeypatch):
    _fake_llm(monkeypatch, "rozważam, rozważam, rozważam", done_reason="length")
    assert generator.generate_answer("Pytanie?", []) == generator.TRUNCATED_ANSWER
