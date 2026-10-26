# Name That LLM (Kaggle) - notebooks and ideas

Competition: https://www.kaggle.com/competitions/name-that-llm (18-way LLM attribution of 230-1900 char passages,
code competition, hidden test, 5 submissions / day, deadline 2027-01-14). Train: 150 prompts x 18 models.

| folder | what | GPU | status |
|---|---|---|---|
| `ntl-s01..s05*.ipynb` | earlier forks of the public LoRA pipelines (NLM v21 line) | T4 x2 (rerun only) | archive |
| `notebooks/s06` | public NLM v27 pipeline, full run on commit (validation run) | T4 x2 | running |
| `notebooks/s07` | s06 + commit short-circuit (GPU only in the scoring rerun) | T4 x2 (rerun only) | submitted 57654520, pending |
| `notebooks/s08a` | CPU-only: TF-IDF + stylometry, prompt centring, prompt recovery, Sinkhorn + Hungarian | none | **0.322 public** (CV 0.478) |
| `notebooks/s08b` | s08a + 24k public passage pool, raw/centred 0.5/0.5 | none | submitted 57658001, pending |
| `notebooks/s08c` | s08b with threshold clustering, size-aware blend, structure check | none | committed, submit next day |
| `notebooks/s10` | public 0.651 recipe (nlm-v27 + Llama v5 member via avazaku asset cache), commit short-circuit | T4 x2 (rerun only) | committed, submit 2026-10-27 |
| `notebooks/s11-*` | s10 + **soft-cap marginal decoding** (tau 0.5) replacing the dense-cluster Hungarian, cut 0.15 | T4 x2 (rerun only) | committed, submit 2026-10-27 |
| `notebooks/s12-*` | s11 with cluster cut 0.08 | T4 x2 (rerun only) | committed, submit 2026-10-27 |
| `notebooks/s13-*` | s11 with tau 1.0 | T4 x2 (rerun only) | to commit (2 concurrent GPU sessions max) |
| `cv/` | GroupKFold CV scripts, variant results, clustering diagnostics, `cluster_scale.py`, `soft_assign*.py` | - | - |

Daily log: see `notebooks/*/NOTES.md`.

## Scores so far (public LB, 2026-10-26)
s07 (nlm-v27 fork, Llama skipped) **0.643** | s08 GPU cut 0.04: 0.593 | s09 GPU cut 0.08: 0.639 | s08a CPU 0.322 | s08b CPU 0.353.
Leaderboard top: 0.710 / 0.702 / 0.690. Best public notebook: 0.651.

## Where the next gains are
1. Post-processing (CPU, zero GPU): soft-cap marginal decoding (s11-s13) - proxy +0.012..+0.037 over the Hungarian.
2. Test-prompt recovery is the bottleneck of every group-level step: on the real test, lower cuts over-merge
   (prompts are more similar than in train). Better clustering = sentence-embedding + TF-IDF similarity, or
   model-aware clustering (exclude pairs the classifier says come from the same model).
3. Rank 1-3 (0.70+) almost certainly needs new trained members (LoRA on Qwen3-14B with the 120k-row public pool,
   ~4-6 h T4 x2 per member) - this is a GPU-training cost, not an inference cost; the scoring rerun itself is free.
