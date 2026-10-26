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
| `cv/` | GroupKFold CV scripts, variant results, clustering diagnostics | - | - |

Daily log: see `notebooks/*/NOTES.md`.
