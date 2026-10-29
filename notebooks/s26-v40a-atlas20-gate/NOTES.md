# s26 - V40A sibling-style residual (= s21) + ATLAS 20%

* Base: s21 (romantamrazov "Who am I?" V40A: V31 + OOF family-preserving sibling-style residual, max_alpha 0.08) + commit gate + Llama bf16 fallback.
* Only change vs s21: `--atlas_w 0.10 -> 0.20` (omerfarukmerey's A/B: V31 0.660 -> atlas_w 0.20 0.663).
  Built instead of submitting s21 as is, because s25 (atlas20) is the better base and V40A was still on atlas 0.10.
* Commit v1 COMPLETE (placeholder, 3 rows; "competition rerun: False"), ~3 GPU min. Image BYOD 2757e0.
* Submitted 2026-10-29 08:54 UTC, ref 57765425 (10-29 NTL slot 3/5).
* Read-out: s26 - s25 = sibling residual on top of atlas20 (noise between identical runs has been up to 0.009, so treat < 0.005 as a tie).
