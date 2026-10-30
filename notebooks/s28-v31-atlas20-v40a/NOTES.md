# s28 - s25 (V31 + dual-GPU 8B + ATLAS 20%) + V40A sibling-style residual + global ATLAS deadline

* Base: s25 (omerfarukmerey 0.663 recipe, our 0.662), cells unchanged except:
  1. `sibling_style.py` (romantamrazov "Who am I?" V40A, byte-identical to s26) called after the V31 typography step,
     max_alpha 0.08, OOF-reliability gated, family probability mass preserved; falls back to the s25 prediction on any error.
  2. Global deadline: `ATLAS_BUDGET = min(8.3h - COMP_SEC - 600, 8.15h - notebook_elapsed)`. The commit log shows the
     Llama bf16 copy now takes 314 s and 665 s pass before infer.py starts - the old budget ignored this pre-infer time.
* Why: s26 (V40A on V31 single-GPU + atlas 0.20) failed with "exceeded the allowed runtime"; its only code difference
  from the scored s20 is a cheap CPU post-step, so the run was at the limit; the global deadline bounds total runtime.
* Offline evidence (train.csv, 5-fold GroupKFold(prompt_id), weak TF-IDF LR proxy base 0.3415 bacc):
  V40A 0.08 -> 0.3449 (+0.0034, 2.3% rows changed); 0.16 -> 0.3456; 0.30 -> 0.3456. In-family OOF bacc of the
  style expert: claude 0.42 (chance 0.20), gemini 0.48 (0.25), gpt 0.42 (0.20), glm 0.53 (0.50, reliability 0.18).
  Expected on LB: 0.662-0.666 (gain is within the +-0.009 rerun noise; read together with future repeats).
* Commit v1 COMPLETE (placeholder, 3 rows), 683 s T4. Image BYOD 2757e0 (as s25).
* Submitted 2026-10-30 19:20 UTC, ref 57814920 (10-30 NTL slot 1/5).
