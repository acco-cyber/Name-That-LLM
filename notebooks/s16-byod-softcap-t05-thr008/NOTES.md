# s16 - s15 with test-prompt cluster cut 0.08

s14 (pinned BYOD 9a6bcf, ATLAS 7.8 h, Hungarian, cut 0.15) scored 0.643, so the pinned container finishes inside the
limit. s15 (= s14 + soft-cap tau 0.5) is scoring. s16 changes only `--adapt_thr 0.15 -> 0.08` on top of s15: with the
hard Hungarian the lower cut lost 0.004 (s09 0.639 vs s07 0.643) but soft-cap should tolerate the impurity of the more
complete clusters (train ARI 0.975 vs 0.867; CPU proxy 0.456 vs 0.433). Read: cut effect under soft-cap = s16 - s15.
Zero extra GPU (commit short-circuit).
