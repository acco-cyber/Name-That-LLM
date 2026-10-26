# s08a - CPU-only pipeline (submitted 2026-10-26 06:16 UTC, ref 57656259)

No GPU at all. GroupKFold(5) over prompt_id (held-out prompts, like the hidden test):

| step | CV accuracy |
|---|---:|
| char+word TF-IDF + stylometry, LR, argmax | 0.356 |
| + prompt centring (subtract per-prompt mean features) | 0.42 |
| + Sinkhorn balancing + Hungarian inside recovered prompts | **0.478** |

Prompt recovery on held-out folds: average-linkage clustering on content words, capped at 18 per
cluster -> purity 1.000 on all 5 folds. Linear-model choice (LR C in 10..100, LinearSVC) moves CV
by < 0.01; see `cv/cv_variants_results.txt`.

Next (s08b): add the public 24k `nlm_matched_clean` passage pool to training - fold-0 CV jumps
0.456 -> 0.546 with the uncentred model + Hungarian.
