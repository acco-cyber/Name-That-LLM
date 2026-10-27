# s15 - s14 container pin + soft-cap marginal decoding (tau 0.5, min cluster 2)

s11-s13 (soft-cap variants) timed out on the 17b2fc image and never scored, so the soft-cap idea is still unmeasured.
s15 = s14 (pinned nlm-v27 BYOD image 9a6bcf, ATLAS budget 7.8 h) with only the post-processing swapped to the s11
soft-cap decoder (`--post_mode softcap --post_tau 0.5 --post_min 2`). Zero extra GPU: commit run is the short-circuit;
the full pipeline only runs in the scoring rerun. Compare directly with s14 once both score.
