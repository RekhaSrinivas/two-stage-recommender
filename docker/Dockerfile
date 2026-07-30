# Airflow image extended with this project's ML dependencies so the DAG tasks can run.
FROM apache/airflow:2.10.5-python3.10

# torch first, from the CPU wheel index (keeps the image small — no CUDA).
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

COPY docker/requirements-airflow.txt /tmp/requirements-airflow.txt
RUN pip install --no-cache-dir -r /tmp/requirements-airflow.txt

# LightGBM (and faiss/sklearn) need the GNU OpenMP runtime, absent from the slim base image.
# Placed last so the expensive torch/pip layers stay cached on rebuild.
USER root
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*
USER airflow
