"""FastAPI service exposing the two-stage recommender.

    GET /health                      -> liveness + which dataset is loaded
    GET /recommend/{user_id}?n=10    -> top-n recommendations (retrieval -> ranking)
    GET /users/{user_id}/history     -> a few of the user's past interactions (UI context)

Run locally:  uvicorn app:app --app-dir serving --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recsys.serving import Recommender  # noqa: E402

STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load artifacts + fit the ranker once, at startup.
    STATE["rec"] = Recommender(dataset=os.getenv("RECSYS_DATASET", "ml-100k"), root=ROOT)
    yield
    STATE.clear()


app = FastAPI(title="Two-Stage Recommender", version="1.0", lifespan=lifespan)


def _rec() -> Recommender:
    return STATE["rec"]


@app.get("/health")
def health():
    r = _rec()
    return {"status": "ok", "dataset": r.dataset_name, "n_users": len(r.valid_user_ids)}


@app.get("/recommend/{user_id}")
def recommend(user_id: int, n: int = Query(10, ge=1, le=50)):
    try:
        return {"user_id": user_id, "n": n, "recommendations": _rec().recommend(user_id, n)}
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown user_id {user_id}")


@app.get("/users/{user_id}/history")
def history(user_id: int, k: int = Query(10, ge=1, le=50)):
    try:
        return {"user_id": user_id, "history": _rec().history(user_id, k)}
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown user_id {user_id}")
