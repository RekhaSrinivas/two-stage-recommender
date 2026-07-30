"""Airflow DAG orchestrating the two-stage recommender end-to-end.

    ingest -> preprocess -> [train_retrieval, train_graph] -> train_ranker -> evaluate -> export

Each task is an idempotent script invocation (re-running overwrites its own outputs; the
raw download is cached), so the DAG is safe to retry from any point. It runs on **ml-100k**
so a full pipeline run finishes in a few minutes inside the container; point `DATASET` at
ml-1m for the real thing. The `[train_retrieval, train_graph]` fan-out shows two independent
producers feeding one consumer — the retriever vectors and the graph embeddings are both
inputs to the ranker.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT = "/opt/airflow/project"
DATASET = "ml-100k"                       # small enough for a full DAG run in minutes
RUN = f"cd {PROJECT} && python "          # every task runs from the project root

default_args = {
    "owner": "recsys",
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    dag_id="recsys_two_stage_pipeline",
    description="Two-stage recommender: retrieval (two-tower) + graph (LightGCN) -> ranking (LambdaMART)",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule=None,                        # trigger manually (or set '@daily')
    catchup=False,
    tags=["recsys", "two-stage", "portfolio"],
    doc_md=__doc__,
) as dag:
    ingest = BashOperator(
        task_id="ingest",
        bash_command=f"{RUN}scripts/prepare_data.py --step ingest --dataset {DATASET}",
    )
    preprocess = BashOperator(
        task_id="preprocess",
        bash_command=f"{RUN}scripts/prepare_data.py --step preprocess --dataset {DATASET}",
    )
    train_retrieval = BashOperator(
        task_id="train_retrieval",
        bash_command=f"{RUN}scripts/run_phase2.py --dataset {DATASET}",
    )
    train_graph = BashOperator(
        task_id="train_graph",
        bash_command=f"{RUN}scripts/run_phase3.py --dataset {DATASET} --only k3",
    )
    train_ranker = BashOperator(
        task_id="train_ranker",
        bash_command=f"{RUN}scripts/run_phase4.py --dataset {DATASET}",
    )
    evaluate = BashOperator(   # QA gate: fail the run if the metrics table wasn't produced
        task_id="evaluate",
        bash_command=f"cd {PROJECT} && test -s results/phase4_{DATASET}.md "
                     f"&& cat results/phase4_{DATASET}.md",
    )
    export_topn = BashOperator(
        task_id="export_topn",
        bash_command=f"{RUN}scripts/export_topn.py --dataset {DATASET} --topn 10",
    )

    ingest >> preprocess >> [train_retrieval, train_graph] >> train_ranker >> evaluate >> export_topn
