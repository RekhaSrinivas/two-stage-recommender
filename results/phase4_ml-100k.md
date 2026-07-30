# Phase 4 — two-stage retrieval->ranking — ml-100k

ml-100k: 938 users x 1,423 items | train=53,485 val=931 test=921 | density=4.0070%

Retriever recall@200 (ceiling): 0.6895

| Model | recall@20 | ndcg@20 | map@20 | hit@20 | coverage@20 |
|---|---|---|---|---|---|
| Retrieval only (two-tower) | 0.1802 | 0.0711 | 0.0415 | 0.1802 | 0.5643 |
| + Rank (LambdaMART, no graph) | 0.1629 | 0.0582 | 0.0299 | 0.1629 | 0.4989 |
| + Rank (LambdaMART, with graph) | 0.1531 | 0.0588 | 0.0336 | 0.1531 | 0.5116 |
