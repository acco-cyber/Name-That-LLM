# s07 - NLM v27 pipeline with commit short-circuit (submitted 2026-10-26 05:10 UTC, ref 57654520)

* Same pipeline as s06 (public NLM v27: fold-averaged two-stage LoRA24 on Qwen3-8B/14B-Base,
  Nishant two-stage checkpoints, Qwen2.5-14B LoRA, DeBERTa-v3-large encoder, 4-view n-gram SVM on
  train+pool, shared-prompt Gaussian group head, adaptive centred twins, 10% ATLAS, Sinkhorn + Hungarian).
* Difference: first cell checks `KAGGLE_IS_COMPETITION_RERUN`. The commit run writes a placeholder
  in ~10 s, so the weekly T4 quota is untouched; the full ~8.3 h pipeline runs only in the hidden-test
  scoring rerun.
* s06 (same code, no short-circuit) was left running as an end-to-end validation of the pipeline.
