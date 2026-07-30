"""Phase 5 driver: two simulated A/B tests with statistical readouts.

  Experiment 1 (headline):  two-stage ranker  vs  MostPopular   -> does personalization win?
  Experiment 2 (subtle):    ranker WITH graph vs ranker WITHOUT -> is the +10% offline lift
                                                                    detectable online, and if
                                                                    not, how much traffic?

Reuses saved two-tower vectors + LightGCN embeddings (no retraining of the retrievers).

Usage:  python scripts/run_phase5.py   (after run_phase2.py and run_phase3.py --only k3)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recsys.baselines import MostPopular  # noqa: E402
from recsys.data import load_dataset  # noqa: E402
from recsys.experiment import (bootstrap_diff_ci, required_n_per_arm,  # noqa: E402
                               simulate_clicks, two_proportion_ztest)
from recsys.ranker import (FEATURE_NAMES, GRAPH_FEATURES, RankContext,  # noqa: E402
                           build_training_data, generate_candidates, rerank, train_ranker,
                           union_seen)


def build_policy_recs(ds, cfg, max_k):
    """Return {policy_name: {user: ranked top-max_k}} for MostPopular and the two rankers."""
    vec = ROOT / "results" / f"vectors_{ds.name}"
    emb = ROOT / "results" / f"graph_emb_{ds.name}"
    ctx = RankContext.build(
        np.load(vec / "user_vecs.npy"), np.load(vec / "item_vecs.npy"),
        np.load(emb / "user_graph_emb.npy"), np.load(emb / "item_graph_emb.npy"),
        ds.train_matrix, ds.item_genres,
    )
    N = cfg["ranker"]["n_candidates"]
    val_users = [u for u in ds.val_items_by_user if ds.val_items_by_user[u]]
    test_users = [u for u in ds.test_items_by_user if ds.test_items_by_user[u]]
    cand_train = generate_candidates(ctx.tt_user, ctx.tt_item, val_users, ds.train_items_by_user, N)
    seen_te = union_seen(ds.train_items_by_user, ds.val_items_by_user)
    cand_test = generate_candidates(ctx.tt_user, ctx.tt_item, test_users, seen_te, N)

    X, y, groups = build_training_data(val_users, cand_train, ds.val_items_by_user, ctx)
    all_cols = list(range(len(FEATURE_NAMES)))
    nograph_cols = [i for i, f in enumerate(FEATURE_NAMES) if f not in GRAPH_FEATURES]
    m_full = train_ranker(X, y, groups, all_cols, cfg["ranker"], cfg["ranker"]["seed"])
    m_ng = train_ranker(X, y, groups, nograph_cols, cfg["ranker"], cfg["ranker"]["seed"])

    pop = MostPopular().fit(ds)
    return test_users, {
        "MostPopular": pop.recommend(test_users, max_k),
        "Ranker (no graph)": rerank(cand_test, ctx, lambda f, c: m_ng.predict(f[:, nograph_cols]), max_k),
        "Ranker (with graph)": rerank(cand_test, ctx, lambda f, c: m_full.predict(f[:, all_cols]), max_k),
    }


def run_ab(name, control, treatment, recs, ground_truth, users, ecfg):
    """Randomize users into two arms, simulate clicks, run the stats, return a readout dict."""
    rng = np.random.default_rng(ecfg["seed"])
    users = np.array(users)
    assign = rng.random(len(users)) < 0.5          # True -> control arm
    ctrl_u = users[assign].tolist()
    trt_u = users[~assign].tolist()

    ctrl_clicks = simulate_clicks({u: recs[control][u] for u in ctrl_u}, ground_truth, rng,
                                  ecfg["examine_model"])
    trt_clicks = simulate_clicks({u: recs[treatment][u] for u in trt_u}, ground_truth, rng,
                                 ecfg["examine_model"])
    ca = np.array([ctrl_clicks[u] for u in ctrl_u])
    ta = np.array([trt_clicks[u] for u in trt_u])

    p1, p2, z, pval = two_proportion_ztest(ca.sum(), len(ca), ta.sum(), len(ta))
    lo, hi = bootstrap_diff_ci(ca, ta, rng, ecfg["n_bootstrap"], ecfg["alpha"])
    rel = (p2 - p1) / p1 * 100 if p1 > 0 else float("nan")
    need = required_n_per_arm(p1, abs(p2 - p1)) if p2 != p1 else float("inf")
    return dict(name=name, control=control, treatment=treatment, n_ctrl=len(ca), n_trt=len(ta),
                ctr_ctrl=p1, ctr_trt=p2, abs_lift=p2 - p1, rel_lift=rel, ci=(lo, hi),
                z=z, pval=pval, need_per_arm=need)


def readout(r, alpha):
    sig = "SIGNIFICANT" if r["pval"] < alpha else "not significant"
    lines = [
        f"### {r['name']}",
        f"- Design: randomization unit = user; arms = control ({r['control']}) vs "
        f"treatment ({r['treatment']}); primary metric = CTR (position-biased click sim).",
        f"- Sample: control n={r['n_ctrl']:,}, treatment n={r['n_trt']:,}.",
        f"- Control CTR = {r['ctr_ctrl']:.4f} | Treatment CTR = {r['ctr_trt']:.4f}",
        f"- Absolute lift = {r['abs_lift']:+.4f}  ({r['rel_lift']:+.1f}% relative)",
        f"- 95% bootstrap CI on lift = [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]",
        f"- Two-proportion z = {r['z']:.2f}, p = {r['pval']:.2e}  ->  **{sig}** at alpha={alpha}",
        f"- Power: to detect this lift at 80% power you'd need ~{r['need_per_arm']:,} users/arm.",
        f"- Decision: {'SHIP' if r['pval'] < alpha and r['abs_lift'] > 0 else 'DO NOT ship on this evidence'}.",
    ]
    return "\n".join(lines)


def main():
    cfg = yaml.safe_load(open(ROOT / "configs" / "default.yaml"))
    ks = cfg["eval"]["ks"]
    max_k = max(ks)
    ds = load_dataset(
        dataset=cfg["data"]["dataset"], data_dir=str(ROOT / cfg["data"]["data_dir"]),
        min_rating_positive=cfg["data"]["min_rating_positive"],
        min_user_interactions=cfg["data"]["min_user_interactions"],
        test_holdout=cfg["data"]["test_holdout"], val_holdout=cfg["data"]["val_holdout"],
        seed=cfg["data"]["seed"],
    )
    print(ds.summary())
    if not (ROOT / "results" / f"vectors_{ds.name}" / "item_vecs.npy").exists():
        sys.exit("Missing vectors/embeddings. Run run_phase2.py and run_phase3.py --only k3 first.")

    print("Building policy recommendations ...")
    test_users, recs = build_policy_recs(ds, cfg, max_k)
    ecfg = cfg["experiment"]

    exps = [
        run_ab("Experiment 1 - Personalization vs Popularity", "MostPopular",
               "Ranker (with graph)", recs, ds.test_items_by_user, test_users, ecfg),
        run_ab("Experiment 2 - Graph feature (with vs without)", "Ranker (no graph)",
               "Ranker (with graph)", recs, ds.test_items_by_user, test_users, ecfg),
    ]
    report = "\n\n".join(readout(r, ecfg["alpha"]) for r in exps)
    print("\n=== Phase 5: A/B test readouts (" + ds.name + ") ===\n" + report)

    out = ROOT / "results" / f"phase5_ab_{ds.name}.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"# Phase 5 — simulated A/B tests — {ds.name}\n\n{ds.summary()}\n\n{report}\n")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
