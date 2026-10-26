# s14 - soft-cap tau 0.5 with the aggressive cut 0.04 (robustness probe)

With the hard Hungarian, cut 0.04 scored 0.593 (s08) vs 0.643 at 0.15: the merged clusters were impure and forcing
all-distinct labels destroyed correct predictions. If soft-cap neutralises that (score >= s11), the test clusters can be
merged aggressively and the completeness gain captured; if not, the impurity penalty is in the probabilities themselves
(centred twins / Gaussian heads also use the clusters) and the cut must stay high.
