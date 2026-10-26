# s08b - s08a + public passage pool (submitted 2026-10-26 07:16 UTC, ref 57658001)

* Adds the 24,443-row `dhruvtibarewal/nlm-matched-clean` pool (MIT) to training with sample weight 0.5.
* Pool passages have no prompt_id: pseudo-prompts are recovered by the same capped average-linkage
  clustering (2,649 clusters, median size 9) so the pool can be prompt-centred too.
* Blend: 0.5 x uncentred LR + 0.5 x centred LR (C=10), Sinkhorn, per-prompt Hungarian. SVC dropped (no gain, 2x time).
* CV evidence (fold 0 only - the 2-CPU box ran out of memory on fold 1): Hungarian accuracy 0.456 -> 0.546
  (uncentred+pool 0.546, centred+pool 0.535).
* Commit run: 26 min on Kaggle CPU, 42k features, ~5 GB dense centred matrix. No GPU.
* `ntl_s08_template.py` + `mk_nb.py` generate both s08a and s08b notebooks from one source
  (`python mk_nb.py src out USE_EXTRA MAX_CHAR MAX_WORD kernel_id title C_LR W_RAW W_SVC`).

## Ideas queued for the next daily budget (5 submissions / day, shared by the sessions working on this account)
1. Full 5-fold CV of the pool blend on a bigger machine; tune extra_weight (0.3 / 0.5 / 1.0) and w_raw.
2. Pseudo-label the hidden test after Hungarian (high-confidence groups) and refit - transductive, GPU-free.
3. Sentence-embedding features (e5 / bge-small, CPU inference ~2 min for 3k texts) prompt-centred, as a third view.
4. Group-level decoding across the 18-way prompt groups jointly with the GPU pipeline's log-probs
   (combine s07 LoRA log-probs with the CPU views before the Hungarian step).
