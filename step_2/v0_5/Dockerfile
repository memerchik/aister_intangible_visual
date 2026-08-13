FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MALLOC_ARENA_MAX=1 \
    MALLOC_TRIM_THRESHOLD_=65536 \
    DNNL_PRIMITIVE_CACHE_CAPACITY=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu \
      torch==2.8.0 \
    && pip install --no-cache-dir -r requirements.txt

COPY artifacts ./artifacts
COPY src ./src
COPY static ./static
COPY server.py prediction_worker.py product_contract.json ./

RUN useradd --create-home --uid 10001 aister \
    && mkdir -p /app/contributions/pending \
    && chown -R aister:aister /app

USER aister
EXPOSE 10000

CMD ["python", "server.py"]
