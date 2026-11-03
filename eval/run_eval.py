"""Ewaluacja całego pipeline'u na eval/questions.json: wyszukiwanie, treść odpowiedzi i cytowane źródła.

Użycie: python -m eval.run_eval
Wymaga indeksu zbudowanego przez `python main.py add`. Zapisuje eval/results.md.
"""

import json
import re
import statistics
import sys
import time
from pathlib import Path

from src.datastore import get_collection
from src.generator import REFUSAL, generate_answer
from src.retriever import retrieve

EVAL_DIR = Path(__file__).resolve().parent
QUESTIONS_FILE = EVAL_DIR / "questions.json"
RESULTS_FILE = EVAL_DIR / "results.md"
REFUSAL_PREFIX = REFUSAL.split(" –")[0]  # "Nie wiem"
TYPE_LABELS = {
    "single": "jeden dokument",
    "archive": "wersja archiwalna w zbiorze",
    "multi": "dwa dokumenty",
    "none": "brak odpowiedzi",
}


def evaluate(q: dict) -> dict:
    """Uruchamia pipeline dla jednego pytania i ocenia wynik.

    - wyszukiwanie: wszystkie oczekiwane pliki są wśród znalezionych chunków (nie dotyczy pytań bez odpowiedzi),
    - odpowiedź: pytanie z odpowiedzią -> są wszystkie wzorce `must`, brak wzorców `must_not`; bez odpowiedzi -> odmowa,
    - źródła: pytanie z odpowiedzią -> cytowane są wszystkie oczekiwane pliki; bez odpowiedzi -> żaden plik.
    """
    start = time.perf_counter()
    chunks = retrieve(q["question"])
    answer = generate_answer(q["question"], chunks)
    retrieved = {c["source"] for c in chunks}
    refused = answer.startswith(REFUSAL_PREFIX)

    if q["type"] == "none":
        retrieval_ok, answer_ok, sources_ok = None, refused, ".pdf" not in answer
    else:
        retrieval_ok = set(q["expected_sources"]) <= retrieved
        answer_ok = (
            not refused
            and all(re.search(p, answer, re.IGNORECASE) for p in q["must"])
            and not any(re.search(p, answer, re.IGNORECASE) for p in q.get("must_not", []))
        )
        sources_ok = all(s in answer for s in q["expected_sources"])
    return {
        **q,
        "answer": answer,
        "retrieval_ok": retrieval_ok,
        "answer_ok": answer_ok,
        "sources_ok": sources_ok,
        "seconds": time.perf_counter() - start,
    }


def _mark(value: bool | None) -> str:
    return "–" if value is None else ("✅" if value else "❌")


def _ratio(rows: list[dict], key: str) -> str:
    values = [r[key] for r in rows if r[key] is not None]
    return f"{sum(values)}/{len(values)}" if values else "–"


def render_report(rows: list[dict]) -> str:
    """Raport Markdown: podsumowanie według typu pytań oraz każde pytanie z odpowiedzią modelu."""
    times = [r["seconds"] for r in rows]
    lines = [
        "# Wyniki ewaluacji",
        "",
        "| typ pytań | liczba | wyszukiwanie | odpowiedź | źródła |",
        "|---|---|---|---|---|",
    ]
    for qtype in ["single", "archive", "multi", "none", None]:
        group = [r for r in rows if qtype is None or r["type"] == qtype]
        if group:
            label = "**razem**" if qtype is None else TYPE_LABELS[qtype]
            lines.append(
                f"| {label} | {len(group)} | {_ratio(group, 'retrieval_ok')} | "
                f"{_ratio(group, 'answer_ok')} | {_ratio(group, 'sources_ok')} |"
            )
    lines += [
        "",
        f"Czas na pytanie: średnio {statistics.mean(times):.0f} s, mediana {statistics.median(times):.0f} s; "
        f"cała ewaluacja: {sum(times) / 60:.0f} min.",
        "",
        "| id | typ | pytanie | wyszukiwanie | odpowiedź | źródła | odpowiedź modelu |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        answer = " / ".join(line.strip() for line in r["answer"].splitlines() if line.strip())
        lines.append(
            f"| {r['id']} | {TYPE_LABELS[r['type']]} | {r['question']} | {_mark(r['retrieval_ok'])} | "
            f"{_mark(r['answer_ok'])} | {_mark(r['sources_ok'])} | {answer.replace('|', '/')} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    sys.stdout.reconfigure(errors="replace")
    if get_collection().count() == 0:  # bez indeksu model dostałby pusty kontekst i zawsze odmawiał
        print("Baza jest pusta. Najpierw uruchom: python main.py add", file=sys.stderr)
        return 1
    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))

    rows = []
    for q in questions:
        row = evaluate(q)
        rows.append(row)
        print(
            f"[{q['id']:>2}] wyszukiwanie {_mark(row['retrieval_ok'])} odpowiedź {_mark(row['answer_ok'])} "
            f"źródła {_mark(row['sources_ok'])} ({row['seconds']:.0f} s) {q['question']}",
            flush=True,
        )

    RESULTS_FILE.write_text(render_report(rows), encoding="utf-8")
    print(f"\nZapisano raport: {RESULTS_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
