# s19 - fork of top public "Who am I?" (romantamrazov) + commit short-circuit

* Kaggle kernel `gwey1025/ntl-s19-whoami-shortcircuit` v1, submitted 2026-10-28 ~10:00 UTC, ref 57732758 (10-28 NTL slot 5/5).
* Upstream is first in the public notebook list sorted by score (above lightningv08 0.652 and the Soft Sinkhorn 0.5 ablation),
  i.e. the highest public score available. Same core stack as our s14/s18 (nlm-v27 v5 Qwen3-14B/8B + Llama-3.1-8B, Nishant
  two-stage Qwen3-14B / Qwen2.5-14B, DeBERTa, 4-view SVC, Gaussian heads, comp stack 0.15, ATLAS 0.10) plus:
  - partial ATLAS (shortest-first, completed rows kept) - same idea as our s17/s18;
  - V38/V39 **gated typography-Bayes correction**: 34 binary typography markers (em-dash, curly quotes, `**`, bullets,
    "However," ...) -> smoothed class-conditional log-likelihood, centred, x2, clipped +-0.35, multiplied by a confidence gate
    clip((0.8 - top2 margin)/0.8, 0, 1), added to the pre-Hungarian logits; final = per-passage argmax (the dense-cluster
    Hungarian result is NOT used for the final file). Upstream labels its no-row-norm variant "candidate_reference_0654".
* Our changes: RERUN short-circuit (commit = input resolution + placeholder, ~3 min, no model load) and ATLAS budget 8.3 h -> 8.0 h.
* Image kept at upstream's BYOD sha256:2757e0... (the one it scored on).
* Why the held slot was used: s17/s18 only test partial ATLAS, which this fork already contains together with the
  typography correction; it is the best-informed candidate available before 00:00 UTC.
