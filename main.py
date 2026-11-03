"""CLI: python main.py reset | add | query "pytanie"."""

import argparse
import sys
import time
from pathlib import Path

import ollama

from src import config
from src.chunker import chunk_document
from src.datastore import reset_collection, upsert_chunks
from src.embedder import embed_texts
from src.generator import generate_answer
from src.loader import load_documents
from src.retriever import retrieve


def cmd_reset(_: argparse.Namespace) -> None:
    print("Usunięto kolekcję." if reset_collection() else "Kolekcja nie istniała.")


def cmd_add(args: argparse.Namespace) -> None:
    documents = load_documents(Path(args.path))
    total = 0
    for source, text in documents:
        chunks = chunk_document(source, text)
        upsert_chunks(chunks, embed_texts([c["text"] for c in chunks]))
        total += len(chunks)
        archived = " (archiwalny, pomijany przy wyszukiwaniu)" if chunks[0]["metadata"]["archived"] else ""
        print(f"  {source}: {len(chunks)} chunków{archived}")
    print(f"Zapisano {len(documents)} plików, {total} chunków.")


def cmd_query(args: argparse.Namespace) -> None:
    start = time.perf_counter()
    chunks = retrieve(args.question)
    if not chunks:
        print("Baza jest pusta. Najpierw uruchom: python main.py add")
        return
    retrieval_s = time.perf_counter() - start
    answer = generate_answer(args.question, chunks)
    generation_s = time.perf_counter() - start - retrieval_s

    print(answer)
    print(f"\nFragmenty przekazane do modelu (top-{len(chunks)}):")
    for i, c in enumerate(chunks, 1):
        content = c["text"].split("\n", 1)[-1]  # bez linii "tytuł – sekcja"
        print(f"  {i}. [score {c['score']:.3f}] {c['id']} ({c['section']})")
        print(f"     {content[:100]}...")
    print(f"\nCzas: wyszukiwanie {retrieval_s:.1f} s, odpowiedź modelu {generation_s:.1f} s")


def main() -> int:
    parser = argparse.ArgumentParser(description="Lokalny asystent RAG (Ollama + Chroma).")
    sub = parser.add_subparsers(required=True)
    sub.add_parser("reset", help="usuń kolekcję z bazy").set_defaults(func=cmd_reset)
    add = sub.add_parser("add", help="wczytaj, podziel, zaembeduj i zapisz dokumenty")
    add.add_argument("-p", "--path", default=str(config.DOCUMENTS_DIR), help="katalog z dokumentami")
    add.set_defaults(func=cmd_add)
    query = sub.add_parser("query", help="zadaj pytanie")
    query.add_argument("question")
    query.set_defaults(func=cmd_query)
    args = parser.parse_args()

    sys.stdout.reconfigure(errors="replace")  # konsola Windows nie wyświetli każdego znaku z odpowiedzi modelu
    try:
        args.func(args)
    except ConnectionError:
        print("Błąd: brak połączenia z Ollamą. Uruchom aplikację Ollama (albo `ollama serve`).", file=sys.stderr)
        return 1
    except ollama.ResponseError as exc:  # np. brak modelu
        print(f"Błąd Ollamy: {exc.error}. Pobierz modele: ollama pull {config.LLM_MODEL} / {config.EMBED_MODEL}",
              file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(f"Błąd: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
