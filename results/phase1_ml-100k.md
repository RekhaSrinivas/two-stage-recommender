# Phase 1 baselines — ml-100k

ml-100k: 938 users x 1,423 items | train=53,485 val=931 test=921 | density=4.0070%

| Model | recall@20 | ndcg@20 | map@20 | mrr | hit@20 | coverage@20 | novelty@20 |
|---|---|---|---|---|---|---|---|
| MostPopular | 0.0988 | 0.0387 | 0.0223 | 0.0223 | 0.0988 | 0.0675 | 7.6490 |
| ItemItemCF | 0.1303 | 0.0518 | 0.0302 | 0.0302 | 0.1303 | 0.1736 | 8.0214 |
| ALS | 0.1368 | 0.0579 | 0.0364 | 0.0364 | 0.1368 | 0.5292 | 8.9412 |


## Full metric dump

### MostPopular

- coverage@10: 0.0450
- coverage@20: 0.0675
- coverage@5: 0.0267
- diversity@10: 0.6811
- diversity@20: 0.7238
- diversity@5: 0.6636
- hit@10: 0.0597
- hit@20: 0.0988
- hit@5: 0.0315
- map@10: 0.0196
- map@20: 0.0223
- map@5: 0.0158
- mrr: 0.0223
- n_eval_users: 921.0000
- ndcg@10: 0.0288
- ndcg@20: 0.0387
- ndcg@5: 0.0197
- novelty@10: 7.4485
- novelty@20: 7.6490
- novelty@5: 7.2781
- precision@10: 0.0060
- precision@20: 0.0049
- precision@5: 0.0063
- recall@10: 0.0597
- recall@20: 0.0988
- recall@5: 0.0315

### ItemItemCF

- coverage@10: 0.1300
- coverage@20: 0.1736
- coverage@5: 0.1033
- diversity@10: 0.7140
- diversity@20: 0.7326
- diversity@5: 0.7066
- hit@10: 0.0825
- hit@20: 0.1303
- hit@5: 0.0510
- map@10: 0.0269
- map@20: 0.0302
- map@5: 0.0227
- mrr: 0.0302
- n_eval_users: 921.0000
- ndcg@10: 0.0398
- ndcg@20: 0.0518
- ndcg@5: 0.0297
- novelty@10: 7.8758
- novelty@20: 8.0214
- novelty@5: 7.7514
- precision@10: 0.0083
- precision@20: 0.0065
- precision@5: 0.0102
- recall@10: 0.0825
- recall@20: 0.1303
- recall@5: 0.0510

### ALS

- coverage@10: 0.4420
- coverage@20: 0.5292
- coverage@5: 0.3514
- diversity@10: 0.7172
- diversity@20: 0.7274
- diversity@5: 0.7084
- hit@10: 0.0912
- hit@20: 0.1368
- hit@5: 0.0467
- map@10: 0.0331
- map@20: 0.0364
- map@5: 0.0275
- mrr: 0.0364
- n_eval_users: 921.0000
- ndcg@10: 0.0463
- ndcg@20: 0.0579
- ndcg@5: 0.0322
- novelty@10: 8.7878
- novelty@20: 8.9412
- novelty@5: 8.6810
- precision@10: 0.0091
- precision@20: 0.0068
- precision@5: 0.0093
- recall@10: 0.0912
- recall@20: 0.1368
- recall@5: 0.0467

