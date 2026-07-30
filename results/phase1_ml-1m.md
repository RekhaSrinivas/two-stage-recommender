# Phase 1 baselines — ml-1m

ml-1m: 6,034 users x 3,525 items | train=563,204 val=6,030 test=6,030 | density=2.6479%

| Model | recall@20 | ndcg@20 | map@20 | mrr | hit@20 | coverage@20 | novelty@20 |
|---|---|---|---|---|---|---|---|
| MostPopular | 0.0673 | 0.0262 | 0.0151 | 0.0151 | 0.0673 | 0.0499 | 8.2405 |
| ItemItemCF | 0.0837 | 0.0335 | 0.0199 | 0.0199 | 0.0837 | 0.1296 | 8.5672 |
| ALS | 0.1114 | 0.0431 | 0.0246 | 0.0246 | 0.1114 | 0.5350 | 9.7724 |


## Full metric dump

### MostPopular

- coverage@10: 0.0326
- coverage@20: 0.0499
- coverage@5: 0.0224
- diversity@10: 0.6727
- diversity@20: 0.7270
- diversity@5: 0.6207
- hit@10: 0.0390
- hit@20: 0.0673
- hit@5: 0.0207
- map@10: 0.0132
- map@20: 0.0151
- map@5: 0.0109
- mrr: 0.0151
- n_eval_users: 6030.0000
- ndcg@10: 0.0191
- ndcg@20: 0.0262
- ndcg@5: 0.0133
- novelty@10: 8.0703
- novelty@20: 8.2405
- novelty@5: 7.9529
- precision@10: 0.0039
- precision@20: 0.0034
- precision@5: 0.0041
- recall@10: 0.0390
- recall@20: 0.0673
- recall@5: 0.0207

### ItemItemCF

- coverage@10: 0.0962
- coverage@20: 0.1296
- coverage@5: 0.0729
- diversity@10: 0.6875
- diversity@20: 0.7203
- diversity@5: 0.6447
- hit@10: 0.0511
- hit@20: 0.0837
- hit@5: 0.0302
- map@10: 0.0177
- map@20: 0.0199
- map@5: 0.0150
- mrr: 0.0199
- n_eval_users: 6030.0000
- ndcg@10: 0.0254
- ndcg@20: 0.0335
- ndcg@5: 0.0187
- novelty@10: 8.4376
- novelty@20: 8.5672
- novelty@5: 8.3460
- precision@10: 0.0051
- precision@20: 0.0042
- precision@5: 0.0060
- recall@10: 0.0511
- recall@20: 0.0837
- recall@5: 0.0302

### ALS

- coverage@10: 0.4811
- coverage@20: 0.5350
- coverage@5: 0.4244
- diversity@10: 0.6189
- diversity@20: 0.6491
- diversity@5: 0.5907
- hit@10: 0.0658
- hit@20: 0.1114
- hit@5: 0.0353
- map@10: 0.0215
- map@20: 0.0246
- map@5: 0.0174
- mrr: 0.0246
- n_eval_users: 6030.0000
- ndcg@10: 0.0317
- ndcg@20: 0.0431
- ndcg@5: 0.0218
- novelty@10: 9.7885
- novelty@20: 9.7724
- novelty@5: 9.8130
- precision@10: 0.0066
- precision@20: 0.0056
- precision@5: 0.0071
- recall@10: 0.0658
- recall@20: 0.1114
- recall@5: 0.0353

