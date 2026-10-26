# s07 - NLM v27 pipeline with commit short-circuit (submitted 2026-10-26 05:10 UTC, ref 57654520)

* Same pipeline as s06 (public NLM v27: fold-averaged two-stage LoRA24 on Qwen3-8B/14B-Base,
  Nishant two-stage checkpoints, Qwen2.5-14B LoRA, DeBERTa-v3-large encoder, 4-view n-gram SVM on
  train+pool, shared-prompt Gaussian group head, adaptive centred twins, 10% ATLAS, Sinkhorn + Hungarian).
* Difference: first cell checks `KAGGLE_IS_COMPETITION_RERUN`. The commit run writes a placeholder
  in ~10 s, so the weekly T4 quota is untouched; the full ~8.3 h pipeline runs only in the hidden-test
  scoring rerun.
* s06 (same code, no short-circuit) was left running as an end-to-end validation of the pipeline.

## s06 validation result (2026-10-26 08:10 UTC)
s06 (full pipeline on commit, 3-row public test) finished **COMPLETE with no errors in 3h20m on T4 x2**
(comp stack 4,239 s; LoRA members, DeBERTa and ATLAS all loaded; Llama member skipped as the model is gated).
Its three public-row predictions (claude-sonnet-5, gpt-5.6-sol, gemini-3.1-pro-preview) are identical to the
CPU-only s08b predictions. This is the evidence that the s07 scoring rerun should complete; the hidden test is
larger so the rerun will run closer to the 8.3 h ATLAS budget.
