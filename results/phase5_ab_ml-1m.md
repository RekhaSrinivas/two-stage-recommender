# Phase 5 — simulated A/B tests — ml-1m

ml-1m: 6,034 users x 3,525 items | train=563,204 val=6,030 test=6,030 | density=2.6479%

### Experiment 1 — Personalization vs Popularity
- Design: randomization unit = user; arms = control (MostPopular) vs treatment (Ranker (with graph)); primary metric = CTR (position-biased click sim).
- Sample: control n=3,043, treatment n=2,987.
- Control CTR = 0.0302 | Treatment CTR = 0.0509
- Absolute lift = +0.0207  (+68.3% relative)
- 95% bootstrap CI on lift = [+0.0109, +0.0303]
- Two-proportion z = 4.07, p = 4.71e-05  ->  **SIGNIFICANT** at alpha=0.05
- Power: to detect this lift at 80% power you'd need ~1,431 users/arm.
- Decision: SHIP.

### Experiment 2 — Graph feature (with vs without)
- Design: randomization unit = user; arms = control (Ranker (no graph)) vs treatment (Ranker (with graph)); primary metric = CTR (position-biased click sim).
- Sample: control n=3,043, treatment n=2,987.
- Control CTR = 0.0496 | Treatment CTR = 0.0509
- Absolute lift = +0.0013  (+2.5% relative)
- 95% bootstrap CI on lift = [-0.0097, +0.0120]
- Two-proportion z = 0.22, p = 8.22e-01  ->  **not significant** at alpha=0.05
- Power: to detect this lift at 80% power you'd need ~468,139 users/arm.
- Decision: DO NOT ship on this evidence.
