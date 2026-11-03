"""Odpowiedź lokalnego LLM na podstawie znalezionych fragmentów."""

import ollama

from src.config import LLM_MODEL, NUM_CTX, NUM_PREDICT, SEED, TEMPERATURE

REFUSAL = "Nie wiem – w dostępnych dokumentach nie ma tej informacji."

SYSTEM_PROMPT = f"""Jesteś asystentem odpowiadającym na pytania pracowników na podstawie dokumentów firmowych.
Zasady:
- Odpowiadaj wyłącznie na podstawie podanego kontekstu. Niczego nie zmyślaj.
- Jeśli w kontekście nie ma odpowiedzi, odpowiedz dokładnie: "{REFUSAL}" i nie podawaj źródeł.
- Odpowiadaj zwięźle, po polsku.
- Na końcu podaj źródła, z których faktycznie korzystałeś, w formacie: Źródła: plik (sekcja), plik (sekcja)"""

TRUNCATED_ANSWER = f"Model przekroczył limit {NUM_PREDICT} tokenów i nie dał odpowiedzi. Spróbuj ponownie."


def build_messages(question: str, chunks: list[dict]) -> list[dict]:
    context = "\n\n".join(f"[źródło: {c['source']}, sekcja: {c['section']}]\n{c['text']}" for c in chunks)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Kontekst:\n{context}\n\nPytanie: {question}"},
    ]


def generate_answer(question: str, chunks: list[dict]) -> str:
    response = ollama.chat(
        model=LLM_MODEL,
        messages=build_messages(question, chunks),
        think=False,
        options={"num_ctx": NUM_CTX, "temperature": TEMPERATURE, "seed": SEED, "num_predict": NUM_PREDICT},
    )
    if response.done_reason == "length":  # ucięte w trakcie rozumowania
        return TRUNCATED_ANSWER
    # qwen3:4b to wariant Thinking: rozumuje mimo think=False, a rozumowanie kończy znacznikiem </think>.
    answer = response.message.content.rsplit("</think>", 1)[-1].strip()
    # Model czasem dopisuje źródła do odmowy; odmowa nie może niczego cytować.
    return REFUSAL if answer.startswith(REFUSAL) else answer
