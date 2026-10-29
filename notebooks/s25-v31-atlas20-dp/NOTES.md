# s25 - V31 + faster dual-GPU run + ATLAS 20% (public 0.663 recipe)

* Source: omerfarukmerey "0.663 | 0.660 V31 + ATLAS 20%" (byte-for-byte code that scored 0.663 public).
  Its A/B table: V31 0.660 -> + dual-GPU 8B members (same predictions, faster) 0.660 -> + atlas_w 0.10->0.20 **0.663**;
  atlas_w 0.30 also 0.663; margin-only typography gate 0.660/0.661 (no gain).
* Our only change: Llama-3.1-8B from `ilyass323/llama-3-1-8b-bf16` when avazaku's asset-cache kernel does not mount
  (it is deleted / not accessible to gwey1025). Upstream's shard-size assertion is kept and PASSED in our commit.
* Commit: FAST_COMMIT (3 visible rows) -> placeholder, ~6 min T4 (2 min of it the Llama copy). Full pipeline only in the rerun.
* Image BYOD 2757e0 (same as s19/s20).
