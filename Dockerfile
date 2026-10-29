FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FASTEMBED_CACHE_PATH=/app/data/models/fastembed \
    DENSE_EMBEDDING_THREADS=2

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py main.py dense_retrieval.py fast_main.py gemini_main.py production_main.py enterprise_main.py ./
COPY benchmark ./benchmark
COPY tests ./tests

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/data /app/data/models/fastembed \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["uvicorn", "enterprise_main:app", "--host", "0.0.0.0", "--port", "8000"]
