# s08c - robust prompt recovery (committed 2026-10-26 08:00 UTC, NOT yet submitted: daily budget exhausted)

Why: s08a scored **0.322 public** against a GroupKFold CV of 0.478. Local simulations rule out
incomplete prompts as the cause (s08a-config still gets 0.41 when each prompt has only 4-18 of the models),
so the test either has a different group structure than train or a genuine distribution shift. s08c
makes the group machinery degrade gracefully instead of hurting when the structure is not there:

1. Clustering cuts the average-linkage dendrogram at cosine distance 0.9 (calibrated on train:
   within-prompt distances median 0.80 / 95% 0.91, between-prompt 1% 0.93) instead of forcing N/18 clusters.
   Over-size clusters are still split to <= 18.
2. Size-aware blend: a passage in a cluster of size n gets centred-model weight (n-1)/17 - singletons
   (whose centred features are identically zero) use the uncentred model only.
3. Sinkhorn balancing only runs when >= 80% of rows sit in clusters of size >= 9; Hungarian only inside
   clusters of size 2..18. With no group structure the pipeline reduces to the uncentred LR argmax.
4. Keeps the 24k public pool (s08b).

Local checks: 0.460 on a 539-row full-structure fake test, 0.422 when prompts are incomplete (4-18 models).

Submit tomorrow with: `kaggle competitions submit name-that-llm -k gwey1025/ntl-s08c-cpu-pool-robust-clusters -v 1 -f submission.parquet -m "..."`.
