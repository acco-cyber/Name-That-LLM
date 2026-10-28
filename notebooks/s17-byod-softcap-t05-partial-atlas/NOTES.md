# s17 - s15 + partial ATLAS (submitted 2026-10-28 03:54 UTC, ref 57722919, NTL slot 3/5 of 10-28)

Finding: in `infer.py` the ATLAS member (weight 0.10, low-margin rows) was all-or-nothing. Every 25 batches it projected
the finish time and, if the projection exceeded `--atlas_budget`, it raised and the whole member was logged as
`ATLAS SKIPPED`. With the budget cut to 7.8 h (s14+) this is a likely cause of s14 = 0.643 (= s07) instead of the public
0.651: the member may simply be missing in the scoring rerun.

Change (only diff vs s15): rows are still scored shortest-first, but the loop stops at the hard deadline
(`time since T0 > budget`) and returns a mask of finished rows; ATLAS is mixed only on those rows. No projection-based
abort any more. Same total time cap as before, so no extra timeout risk.

Read: s17 - s15 = value of keeping ATLAS when it would have been skipped (0 if ATLAS already finished in s15).
Zero extra GPU: commit run is the short-circuit (~4 min), full pipeline only in the scoring rerun.
