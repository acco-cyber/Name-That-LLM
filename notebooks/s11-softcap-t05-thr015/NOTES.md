# s11 - s10 + soft-cap marginal decoding (tau 0.5) instead of the dense-cluster Hungarian

Idea: the public pipeline's last step is a hard Hungarian assignment inside recovered test clusters of size 12-18
(joint MAP: each model at most once per prompt). Two problems:
1. The metric is per-row (balanced accuracy); for Hamming loss the Bayes decision is the argmax of the *marginal*
   posterior over assignments, not the joint MAP. Sinkhorn iterations approximate those marginals.
2. On the hidden test the Hungarian is fragile to impure clusters: cut 0.04 -> 0.593, 0.08 -> 0.639, 0.15 -> 0.643
   (s08/s09/s07), while on train simulations lower cuts give *better* clusters (cv/cluster_scale.py: ARI 0.975 at 0.08).
   So the test prompts are more similar to each other than train prompts are, and forcing all-distinct labels in a
   merged cluster destroys correct predictions.

Soft-cap: per cluster (size >= 2), P = softmax(tot / tau); repeat 30x { scale down columns whose mass exceeds
cap = max(1, n/18); renormalise rows }; predict argmax P. The cap is an inequality, so small clusters are not
flattened and over-size (impure) clusters may repeat a model.

CPU proxy (cv/soft_assign2.py; TF-IDF LR, 5 folds of 30 held-out prompts, balanced accuracy, argmax 0.342):

| clusters                     | Hungarian (dense) | soft-cap tau 0.5 | tau 1.0 |
|------------------------------|------:|------:|------:|
| true prompts                 | 0.448 | **0.461** | 0.440 |
| two prompts mixed per cluster| 0.340 | **0.377** | 0.368 |
| recovered, cut 0.15          | 0.418 | **0.433** | 0.410 |
| recovered, cut 0.08          | 0.444 | **0.456** | 0.433 |
| recovered, cut 0.04          | 0.448 | **0.461** | 0.439 |
| partial prompts (4-18), 0.15 | 0.378 | **0.390** | 0.365 |

Soft-cap at tau 0.5 beats the hard Hungarian in every scenario, by +0.012..+0.037. Caveat: tau is on the scale of the
pipeline's final log-probs (Sinkhorn-balanced mixture probabilities), which are sharper than the proxy's - hence s13 (tau 1.0).
Everything else identical to s10. Zero extra GPU time.
