# Two-Stage Recommender (Retrieval → Ranking)

A production-style recommender on MovieLens, built to the pattern real systems use:
**candidate generation (retrieval) → ranking**, with a graph-embedding component,
PyTorch models, Airflow orchestration, rigorous offline evaluation, and a simulated
A/B test.

Every number in this repo is **measured on the machine that produced it** — nothing is
hand-waved. Each component is built to be understood and defended, not just to run.

**Demonstrates:** recommender systems (two-stage retrieval→ranking), graph neural networks
(LightGCN, from scratch), learning-to-rank (LambdaMART), ranking evaluation, A/B
experimentation (significance + power), Airflow orchestration, and Dockerized serving —
end-to-end, tested (32 unit tests), reproducible.

## Status  —  all 7 phases complete ✅

| Phase | Component | State |
|---|---|---|
| **1** | Data + leakage-free temporal split + baselines + metric harness | ✅ done |
| **2** | Two-tower retrieval (PyTorch) + FAISS ANN candidate generation | ✅ done |
| **3** | LightGCN graph embeddings (hand-built in PyTorch) + graph ablation | ✅ done |
| **4** | LambdaMART ranking + full retrieval→ranking pipeline + ablations | ✅ done |
| **5** | Simulated A/B test + statistical readout (significance, CI, power) | ✅ done |
| **6** | Airflow DAG (Dockerized) — runs the whole pipeline end-to-end | ✅ done |
| **7** | FastAPI serving + Streamlit demo + Docker | ✅ done |

## Architecture

```
            ┌──────────────────── OFFLINE  (Airflow DAG, Dockerized) ────────────────────┐
MovieLens ──▶ temporal split ──┬──▶ two-tower retrieval (PyTorch) ──▶ item vectors ──┐
                               └──▶ LightGCN graph net (PyTorch)  ──▶ graph embeds  ──┤
                                                                                      ▼
                                                        LambdaMART ranker (LightGBM) ──▶ eval + top-N export
            └────────────────────────────────────────────────────────────────────────┘

            ┌──────────────────────────── ONLINE  (FastAPI + Streamlit) ────────────────┐
request(user_id) ──▶ retrieve top-200 (two-tower) ──▶ rank (LambdaMART + graph feats) ──▶ top-N titles
            └────────────────────────────────────────────────────────────────────────┘
```

Stage 1 (**retrieval**) cheaply narrows 3,525 items → 200 with a dot-product model that can be
ANN-indexed. Stage 2 (**ranking**) spends real modelling effort ordering just those 200 with
richer features — including the LightGCN graph score. A/B testing (Phase 5) decides what ships.

## Quickstart

# Requires Python 3.10 (best wheel support for torch on CPU + Windows).

```bash
py -3.10 -m venv .venv
.venv\Scripts\activate            # Windows;  source .venv/bin/activate on macOS/Linux
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu  # CPU wheel first
pip install -r requirements.txt
pip install -e .

pytest -q                                        # 32 unit tests

# --- modelling (metrics printed + saved to results/) ---
python scripts/run_phase1.py                     # baselines on ml-1m (downloads ~6 MB)
python scripts/run_phase2.py                     # two-tower retrieval  -> saves vectors
python scripts/ann_benchmark.py                  # ANN serving: recall vs latency
python scripts/run_phase3.py --only k0           # LightGCN ablation: K=0 (no graph)
python scripts/run_phase3.py --only k3           # LightGCN: K=3 (graph) -> saves embeddings
python scripts/run_phase3.py --assemble          # collect the Phase 3 table
python scripts/run_phase4.py                     # two-stage: retrieve -> LambdaMART rank
python scripts/run_phase5.py                     # simulated A/B test + statistical readout

# --- orchestration & serving (Docker) ---
docker compose run --rm airflow bash -lc \
  "airflow db migrate && airflow dags test recsys_two_stage_pipeline 2024-01-01"   # Phase 6
docker compose -f docker-compose.serving.yaml up --build   # Phase 7: API :8000, UI :8501
```

> Tip: every `run_phaseN.py` accepts `--dataset ml-100k` for a fast (~seconds) smoke run.

## Phase 1 results — MovieLens-1M

Protocol: implicit feedback (rating ≥ 4 = positive), **per-user temporal leave-last-out**
split (most-recent interaction per user held out for test), full-catalogue ranking with
training items excluded. 6,034 users × 3,525 items, 563k train interactions.

| Model | Recall@10 | NDCG@10 | Recall@20 | NDCG@20 | MAP@20 | Coverage@20 | Novelty@20 |
|---|---|---|---|---|---|---|---|
| MostPopular | 0.0390 | 0.0191 | 0.0673 | 0.0262 | 0.0151 | 0.050 | 8.24 |
| ItemItemCF  | 0.0511 | 0.0254 | 0.0837 | 0.0335 | 0.0199 | 0.130 | 8.57 |
| **ALS (MF)** | **0.0658** | **0.0317** | **0.1114** | **0.0431** | **0.0246** | **0.535** | **9.77** |

Reading this: ALS lifts Recall@20 **+65%** over the popularity baseline and, crucially,
recommends **10× more of the catalogue** (coverage 0.535 vs 0.050) with higher novelty —
it personalises instead of showing everyone the same blockbusters. These are the numbers
the deep models in later phases must beat to justify their complexity.

> Note: with leave-**one**-out (one held-out item per user), Recall@K equals HitRate@K and
> MAP@K equals MRR by construction — each user has exactly one relevant item. This is the
> standard NCF-style protocol; it's expected, not a bug.

## Phase 2 results — two-tower retrieval (PyTorch)

Same test set, same metric harness. The two-tower is trained with an **in-batch sampled
softmax** (logQ popularity correction + accidental-hit masking), item tower fed with genre
features, early-stopped on validation NDCG.

| Model | Recall@20 | NDCG@20 | MAP@20 | Coverage@20 | Novelty@20 |
|---|---|---|---|---|---|
| MostPopular | 0.0673 | 0.0262 | 0.0151 | 0.050 | 8.24 |
| ItemItemCF  | 0.0837 | 0.0335 | 0.0199 | 0.130 | 8.57 |
| ALS (MF)    | 0.1114 | 0.0431 | 0.0246 | 0.535 | 9.77 |
| **TwoTower (PyTorch)** | **0.1318** | **0.0511** | **0.0293** | **0.586** | 9.51 |

The two-tower beats the strong ALS baseline by **+18% Recall@20** and **+19% NDCG@20**,
with the highest catalogue coverage of any model. This is the *retrieval* stage — Phase 4
adds a ranking stage on top of its candidates.

### ANN candidate serving (`ann_benchmark.py`, ml-1m, k=20)

| Method | Recall@20 vs exact | Latency (6,034 users) | Speedup |
|---|---|---|---|
| exact (numpy) | 1.0000 | 193 ms | 1.0× |
| FAISS flat (exact MIPS) | 1.0000 | 14 ms | 13.9× |
| FAISS IVF (nprobe=1) | 0.319 | 20 ms | 9.6× |
| FAISS IVF (nprobe=16) | 0.932 | 25 ms | 7.7× |

`nprobe` is the accuracy/latency dial: probe more Voronoi cells → higher recall, more time.
**Honest scale note:** at 3,525 items exact search is already fast; ANN's real payoff is at
10⁵–10⁸ items. This measures the *mechanism* you'd deploy at scale.

## Phase 3 results — LightGCN graph embeddings (PyTorch, hand-built)

LightGCN treats interactions as a user–item graph and smooths embeddings over neighbours
(`E^{k+1} = Â E^k`, no weight matrix, no nonlinearity), trained with pairwise **BPR** loss.
The key deliverable is an **ablation** — K=0 (no graph propagation = BPR-MF) vs K=3 (graph):

| Model | Recall@20 | NDCG@20 | MAP@20 | Coverage@20 |
|---|---|---|---|---|
| ALS | 0.1114 | 0.0431 | 0.0246 | 0.535 |
| BPR-MF (K=0, **no graph**) | 0.1100 | 0.0405 | 0.0217 | 0.716 |
| **LightGCN (K=3, graph)** | **0.1244** | **0.0482** | **0.0277** | 0.535 |

**Does the graph help? Yes:** K=3 beats the identical model with propagation removed (K=0)
by **+13% Recall@20** and **+19% NDCG@20**. On the smaller ml-100k the lift was tiny — the
graph signal is richer with more data, matching the LightGCN paper's findings.

**Honest comparison:** as a *standalone retriever*, LightGCN (0.1244) does not beat the
Phase 2 two-tower (0.1318) here. That's fine and expected — its value in a two-stage system
is as a **complementary signal**: the saved graph embeddings become ranking features in
Phase 4, where the "with vs without graph features" ablation is the real test.

**Engineering note (a real interview story):** LightGCN first collapsed to popularity. I
diagnosed it methodically — the fix was learning rate (0.001 → 0.01), confirmed by a K=0
ablation proving the BPR training was sound — then found training was CPU-bound on a
`torch.sparse` COO adjacency. Switching to **CSR layout gave a 39× propagation speedup**
(254 ms → 6.5 ms per forward), turning a ~15-minute train into ~5.

## Phase 4 results — two-stage: retrieval → LambdaMART ranking

The full pipeline: the two-tower retrieves top-200 candidates, then **LambdaMART** (LightGBM
learning-to-rank) re-orders them using `tt_score + LightGCN_score + popularity + genre`
features. Trained on genuinely-retrieved val positives, early-stopped, evaluated on test.
Retriever recall@200 (the ranking ceiling): **0.568**.

| Model | Recall@20 | NDCG@20 | MAP@20 | Coverage@20 |
|---|---|---|---|---|
| Retrieval only (two-tower) | 0.1342 | 0.0521 | 0.0299 | 0.588 |
| + Rank (LambdaMART, **no graph**) | 0.1194 | 0.0468 | 0.0273 | 0.692 |
| + Rank (LambdaMART, **with graph**) | 0.1317 | 0.0519 | **0.0304** | 0.678 |

Two honest findings:
1. **The graph embeddings earn their place end-to-end.** Dropping the LightGCN feature costs
   the ranker **−10% Recall@20** (0.132 → 0.119) — the single most valuable non-retriever
   feature, and the payoff of Phase 3 inside the full pipeline.
2. **A strong retriever is hard to beat.** With only IDs + genres, the ranker *matches* the
   two-tower on Recall@20/NDCG and *edges* it on MAP@20 (cleaner precision among hits). The
   headroom is real (recall@200 = 0.57) but needs features MovieLens lacks — session,
   real-time context, cross-features. Reported honestly, not tuned into a false win.

## Phase 5 results — simulated A/B test

Offline metrics say which model *ranks* better; an A/B says whether a change is worth
*shipping*. We simulate one: randomize test users to arms (unit = user), show each arm's
policy, and simulate position-biased clicks (`P(click) = 1/log2(rank+2)` if the relevant
item is shown). Then: two-proportion z-test, bootstrap 95% CI, and a power calculation.

| Experiment | Control CTR | Treatment CTR | Rel. lift | p-value | Verdict |
|---|---|---|---|---|---|
| **1. Personalization vs Popularity** | 0.030 | 0.051 | **+68%** | 4.7e-05 | **SHIP** |
| **2. Graph feature (with vs without)** | 0.0496 | 0.0509 | +2.5% | 0.82 | not significant |

The two experiments are the whole point:
- **A big effect ships cleanly** — personalization crushes popularity, unambiguous.
- **A small-but-real effect can be invisible online.** The graph feature's +10% *offline*
  recall (Phase 4) shows up as a +2.5% CTR lift whose 95% CI **crosses zero** — you'd need
  **~468,000 users per arm** to detect it at 80% power. Offline lift ≠ shippable online, and
  the power math tells you the traffic bill. That reconciliation is the senior insight.

## Phase 6 — Airflow orchestration (Dockerized)

The whole pipeline as an Airflow DAG (`dags/recsys_dag.py`), runnable locally via Docker
Compose. **Verified end-to-end: all 7 tasks green, DagRun `success`.**

```
ingest → preprocess → ┌ train_retrieval ┐→ train_ranker → evaluate → export_topn
                      └ train_graph      ┘
```

- The **fan-out** (`train_retrieval ∥ train_graph → train_ranker`) models two independent
  producers — retriever vectors and graph embeddings — feeding one consumer.
- Every task is an **idempotent** script invocation (re-running overwrites its own outputs;
  the raw download is cached), with `retries=1`, so the DAG is safe to retry from any point.
- `evaluate` is a **QA gate**: it fails the run if the metrics table wasn't produced.
- Runs on ml-100k so a full DAG completes in ~3 min in-container; point `DATASET` at ml-1m
  for the real thing.

```bash
docker compose build
docker compose up          # Airflow UI at http://localhost:8080  (admin/admin)
# or headless, full run:
docker compose run --rm airflow bash -lc \
  "airflow db migrate && airflow dags test recsys_two_stage_pipeline 2024-01-01"
```

The `export_topn` task writes `results/topn_<dataset>.csv` — top-10 recommendations per user
in original MovieLens ids, the artifact a serving layer would consume.

## Phase 7 — serving (FastAPI + Streamlit + Docker)

A REST API runs the full pipeline live per request. **No torch/faiss at serve time** — the
retriever is its precomputed vectors (NumPy) and the ranker is LightGBM, so the serving image
stays small. The ranker is fit once at startup; each request does retrieval → ranking → titles.

```
GET /health                    -> {"status":"ok","dataset":"ml-100k","n_users":938}
GET /recommend/{user_id}?n=10  -> ranked titles + scores
GET /users/{user_id}/history   -> the user's past likes (UI context)
```

Verified live (user 1 liked *Star Wars* / *Fargo* / *Return of the Jedi*):

```json
{"user_id":1,"recommendations":[
  {"rank":1,"item_id":70,"title":"Four Weddings and a Funeral (1994)","score":0.127},
  {"rank":3,"item_id":480,"title":"North by Northwest (1959)","score":0.110},
  {"rank":5,"item_id":511,"title":"Lawrence of Arabia (1962)","score":0.107}]}
```

```bash
# local
uvicorn app:app --app-dir serving --port 8000        # http://localhost:8000/docs
# containerized API + Streamlit UI
docker compose -f docker-compose.serving.yaml up --build   # API :8000, UI :8501
```

`serving/streamlit_app.py` is a small demo UI (pick a user, see their likes and the top-N).

## Design decisions (the "why", for interviews)

- **Implicit feedback, not rating prediction.** We rank items; we never predict a star
  rating. Real signal is clicks/orders, so RMSE is the wrong target — top-N ranking quality
  is what a user experiences.
- **Temporal split, not random.** Holding out each user's *most recent* interactions
  simulates predicting the future from the past. A random split leaks future information and
  inflates every metric.
- **Catalogue = training items.** Val/test interactions on items unseen in training are
  dropped and counted (8 rows on ml-1m) — you can't rank an item the model never saw.
- **ALS from scratch.** The Hu-Koren-Volinsky update is implemented directly
  (`src/recsys/baselines.py`) so the confidence-weighting and alternating-least-squares math
  is transparent, then vectorized into two sparse matmuls + a batched solve.
- **In-batch negatives.** The two-tower doesn't sample explicit negatives — every other item
  in the batch is a negative, which is cheap and scales. The two production fixes (logQ
  popularity correction, accidental-hit masking) are implemented and unit-tested.
- **Retrieval ≠ ranking.** The two-tower is deliberately a *candidate generator*: dot-product
  scoring so items can be pre-indexed for ANN. Rich cross-features (user×item interactions)
  are the ranking stage's job (Phase 4), not retrieval's.
- **LightGCN is "light" on purpose.** No per-layer weight matrix, no nonlinearity — He et al.
  showed those *hurt* when nodes are IDs with no features to transform. The learned parameters
  are only the layer-0 embeddings; everything else is fixed neighbourhood averaging. Built
  from scratch (`src/recsys/lightgcn.py`) so the propagation is fully defensible.
- **Ranking features must vary within a user.** LambdaMART ranks *within* each user "query",
  so a feature constant across a user's candidates (e.g. raw user activity) adds no ranking
  signal and only invites overfitting — every ranker feature is item-level or user×item.
- **Train the ranker on retrieved positives only.** Injecting positives the retriever missed
  teaches the ranker that low retriever-scores can be positive, poisoning its best feature.
  We train on the same distribution we serve: re-ranking what retrieval actually surfaced.

## Repo layout

```
src/recsys/       data.py metrics.py baselines.py two_tower.py torch_data.py ann.py lightgcn.py ranker.py experiment.py
scripts/          run_phase1.py … run_phase5.py  prepare_data.py  export_topn.py  ann_benchmark.py
dags/             recsys_dag.py        (Airflow pipeline)
docker/           Dockerfile  requirements-airflow.txt      (+ docker-compose.yaml at repo root)
serving/          app.py (FastAPI)  streamlit_app.py  Dockerfile  requirements-serving.txt
configs/          default.yaml         (all hyperparameters; nothing hard-coded)
tests/            test_{metrics,split,two_tower,ann,lightgcn,ranker,experiment,serving}.py  (32 tests)
results/          measured metric tables + saved vectors/embeddings (gitignored)
```

## Reproducibility

Python 3.10 venv, pinned `requirements.txt`, deterministic seeds, config-driven runs (nothing
hard-coded in scripts). `numpy` is pinned to 1.x because TensorFlow-era tooling and several
ML wheels still expect that ABI. LightGCN uses **hand-written** PyTorch sparse ops, not
PyTorch Geometric — one fewer fragile dependency, and every line is defensible.
