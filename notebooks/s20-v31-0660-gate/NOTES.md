# s20 - mercurydust V31 (public 0.660) with commit-only gate

* Source: alexchilton/name-that-llm-0-654-full-ensemble-what-failed (latest version = mercurydust *Name That LLM-Submission* V31
  run unchanged apart from a `KAGGLE_IS_COMPETITION_RERUN` gate; scored 0.660 on alexchilton's account).
* Our change: Llama-3.1-8B base is license-gated on Kaggle and does not mount on gwey1025 (v1-v3 failed at
  `mdir('/llama-3.1/', '/8b/')`; the `deleted/140855160` asset-cache kernel used by s14-s19 is gone).
  Fallback: public bf16 dataset `ilyass323/llama-3-1-8b-bf16` (shard sizes identical to meta-llama/Llama-3.1-8B,
  generation_config eos 128001 = base, not Instruct). Used only for the ll8v5 member (w 1.0).
* Image pinned to BYOD 2757e0 (same as s19, which scored 0.654 in time).
* GPU: commit = placeholder (~3 min T4); full pipeline only in the scoring rerun.
* Submitted 2026-10-29 06:01 UTC, ref 57760754 (10-29 NTL slot 1/5).
