# Lightweight serving image (no torch/faiss — the retriever is precomputed vectors).
FROM python:3.10-slim

# libgomp1 for LightGBM's OpenMP runtime.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY serving/requirements-serving.txt /tmp/req.txt
RUN pip install --no-cache-dir -r /tmp/req.txt

# Bake in code + config. Data and trained artifacts (results/, data/) are mounted at
# runtime via docker-compose — images shouldn't carry datasets or model outputs.
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY serving/ /app/serving/

ENV PYTHONPATH=/app/src RECSYS_DATASET=ml-100k
EXPOSE 8000
CMD ["uvicorn", "app:app", "--app-dir", "serving", "--host", "0.0.0.0", "--port", "8000"]
