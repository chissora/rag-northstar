FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py pytest.ini ./
COPY src/ src/
COPY eval/ eval/
COPY tests/ tests/
COPY documents/ documents/

# Testy kodu przy budowaniu obrazu: nie potrzebują Ollamy, a nieudany test przerywa budowanie.
RUN python -m pytest -q

# Ollama działa na hoście (z dostępem do GPU, jeśli jest); kontener łączy się z nią przez sieć.
ENV OLLAMA_HOST=http://host.docker.internal:11434
ENTRYPOINT ["python", "main.py"]
CMD ["--help"]
