# Handoff for the next daily session (task fires 22:03 UTC; Kaggle budgets reset 00:00 UTC)

## Account / tooling
* Kaggle user `gwey1025`; CLI 2.2.5 with `KAGGLE_API_TOKEN`. The cloud egress IP is shared and Kaggle rate-limits it
  (HTTP 429, retry-after up to 20 min) - make few API calls, never poll faster than 1/min.
* GitHub: `gh` is blocked in the cloud session; plain `git` works but push needs a preemptive header:
  `git -c http.extraHeader="Authorization: Basic $(printf 'acco-cyber:<PAT>' | base64 -w0)" push origin main`.
* Scoring reruns of the GPU notebooks take ~8 h; results appear in `kaggle competitions submissions name-that-llm`.
* Max 2 concurrent GPU notebook sessions (commits). Commit runs of s10+ take ~2-5 min (short-circuit).

## Submitting a committed notebook (code competition)
`kaggle competitions submit name-that-llm -k gwey1025/<kernel> -v <version> -f submission.parquet -m "<description>"`

## State at 2026-10-27 00:30 UTC
| kernel | version | content | status |
|---|---|---|---|
| ntl-s10-nlmv27-llama-rerun | 2 | public 0.651 recipe + Llama member | **submitted 10-27 00:01, ref 57681742** |
| ntl-s11-softcap-t05-thr015 | 1 | + soft-cap tau 0.5, cut 0.15 | **submitted 10-27 00:07** |
| ntl-s12-softcap-t05-thr008 | 1 | + soft-cap tau 0.5, cut 0.08 | **submitted 10-27 00:16** |
| ntl-s13-softcap-t10-thr015 | 1 | + soft-cap tau 1.0, cut 0.15 | **submitted 10-27 00:25** |
| ntl-s14-softcap-t05-thr004 | 1 | + soft-cap tau 0.5, cut 0.04 (robustness probe) | committed, NOT submitted - 1 slot of 10-27 left in reserve |

The 10-27 budget has ONE slot left. The next session (22:03 UTC) will see the s10-s13 scores: use that slot before
00:00 UTC on the best-informed variant (s14 as committed, or a new one), then the 10-28 budget opens.

## Decision rules once the 10-27 scores land (~08:00-10:00 UTC)
* s10 should be ~0.651 (sanity). If s10 < 0.643 something in the Llama member differs - compare with s07.
* soft-cap gain = s11 - s10. tau direction = sign(s13 - s11). cut direction = s12 vs s11, s14 vs s11.
* If s14 >= s11: clusters can be merged aggressively -> next: cut 0.02 / cap 18 with soft-cap, and let the centred twins
  (adapt_w) use them. If s14 << s11: impurity also hurts the probability members -> keep cut >= 0.08 and work on
  model-aware clustering (block merges whose confident label sets overlap).
* Keep one of the 5 daily slots in reserve until the morning scores are in.

## Beyond post-processing (needed for 0.70+; leaderboard top 0.710 / 0.702 / 0.690)
1. New trained members: fold-averaged LoRA on Qwen3-14B-Base with the 120k-row `nlm_pool_v5` (prompt_key groups) in
   addition to train - the public v5 adapters used a smaller pool. Training = GPU hours (est. 4-6 h T4x2 per member);
   inference is charged to the scoring rerun, not the quota. Start from `lucasboesen/name-that-llm-train-lora24-s7`.
2. Better test-prompt recovery (sentence embeddings + TF-IDF; label-aware merging) - every group-level member benefits.
3. Pseudo-labelling the hidden test inside the rerun (self-training the cheap heads on confident rows) - zero extra GPU.

## Update 2026-10-27 22:55 UTC (hourly run)
* s10-s13 finished with NO public score: they timed out on the newer Kaggle image (17b2fc). The soft-cap
  decoding (s11-s13) is therefore still unmeasured on the real stack.
* The 10-27 slot was used at 19:49 for `ntl-s14-llama-byod` v? (archived in `notebooks/s14-llama-byod`):
  s10 pinned to the nlm-v27 container that scored for s07 + ATLAS budget 8.3h->7.8h. Score pending (~8 h rerun).
  NTL 10-27 budget is fully used; 5 new slots open 00:00 UTC 10-28.
* Plan for 10-28: if s14 scores (~0.651), re-commit s11/s12/s13 with the s14 container pin and time budget
  and submit them one per hourly run. If s14 also times out, cut ATLAS further / drop a LoRA member first.
* Leaderboard 10-27 22:45 UTC: 0.728 / 0.707 / 0.704 / 0.702. Public notebooks still top out at 0.651
  (kirilltym Fresh32 is unscored and depends on private datasets, not forkable).
* Coordination: runs fire hourly at :42 and can overlap. Check `kaggle competitions submissions` before
  every submit and append to RUNLOG.md.

## Update 2026-10-28 00:01 UTC (hourly run)
* s15 = s14 + soft-cap tau 0.5 submitted with 10-28 slot 1 (ref 57716199). s14 (pinned) and s15 now score in parallel,
  so by ~08:00 UTC we know both whether the pinned container fits the time limit and the soft-cap delta (s15 - s14).
* Remaining 10-28 NTL slots: 4. Do NOT spend them on more soft-cap/cut variants until s14/s15 score - if s14 times
  out, s15 does too. Next independent idea worth a slot meanwhile: CPU-side (zero GPU) changes on the pinned image only.
* Quantum: v8 scored 22.009 (best). manwithacat 'The Real 150K' output is byte-identical to v8 (sha be8301da) - no
  public output ahead of us. LB 10-27 23:45: 25.629 / 24.465 / 24.159 / 23.895 / 23.800.

## Update 2026-10-28 02:55 UTC (hourly run)
* s14 scored **0.643** -> the pinned BYOD 9a6bcf image + ATLAS 7.8 h finishes in time. It equals s07 (0.643), so the
  Llama member / public 0.651 recipe did not reproduce its gain on this image; the 0.008 gap is still unexplained
  (candidates: ATLAS budget cut 8.3->7.8 h dropping rows, or Llama asset cache not loading - check the s14 rerun log).
* s16 = s15 + cluster cut 0.08 submitted (slot 2/5, ref 57721428). s15 and s16 score ~08:00-11:00 UTC.
* 3 NTL slots left for 10-28. Next candidates (each one diff on s15, all zero-GPU short-circuit):
  tau 1.0 (s13 idea) / ATLAS budget back to 8.0 h / atlas_w 0.15.

## Update 2026-10-28 03:55 UTC (hourly run)
* Found that ATLAS was all-or-nothing: a projected overrun of `--atlas_budget` raised and dropped the member entirely.
  s17 = s15 + partial ATLAS (stop at hard deadline, mix only on finished rows) submitted, slot 3/5, ref 57722919.
* 2 NTL slots left for 10-28. Read s15 vs s14 (soft-cap), s16 vs s15 (cut), s17 vs s15 (ATLAS coverage) once scored
  (~08:00-12:00 UTC) and combine the winners in s18. If s17 > s15 notably, also try ATLAS budget 8.0 h with partial mode
  (safe now: it can only lose rows, not the member).
