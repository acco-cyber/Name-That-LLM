# s12 - s11 with cluster cut 0.08

With the hard Hungarian the lower cut lost 0.004 on the public LB (s09 0.639 vs s07 0.643) although it recovers far more
complete clusters on train (ARI 0.975 vs 0.867, 96% vs 74% of rows in dense clusters). Soft-cap tolerates impure
clusters, so the more complete clusters should now pay off (proxy: 0.456 vs 0.433 at cut 0.15).
