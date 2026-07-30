# Phase 4 — two-stage retrieval->ranking — ml-1m

ml-1m: 6,034 users x 3,525 items | train=563,204 val=6,030 test=6,030 | density=2.6479%

Retriever recall@200 (ceiling): 0.5677

| Model | recall@20 | ndcg@20 | map@20 | hit@20 | coverage@20 |
|---|---|---|---|---|---|
| Retrieval only (two-tower) | 0.1342 | 0.0521 | 0.0299 | 0.1342 | 0.5881 |
| + Rank (LambdaMART, no graph) | 0.1194 | 0.0468 | 0.0273 | 0.1194 | 0.6922 |
| + Rank (LambdaMART, with graph) | 0.1317 | 0.0519 | 0.0304 | 0.1317 | 0.6780 |
