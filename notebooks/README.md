# Notebooks

Run them in numeric order. Each notebook states which manuscript artefact it
produces and where its inputs come from.

| Notebook | Produces | Needs | Outputs stored? |
|---|---|---|---|
| `01_baselines.ipynb` | Table 1 (baseline benchmark) | cached `results/baselines_*.csv` | yes |
| `02_cross_validation.ipynb` | Table 3, paired CV comparison | `results/cv_*.csv`, `results/cv_baselines_*.csv` | yes |
| `03_pruning_figures.ipynb` | Figure 4 and Figures S2--S5 | checkpoints, `training_history.json` | yes |
| `04_inference_benchmark.ipynb` | Figure S1, latency tables | checkpoints, baseline CSVs | yes |
| `05_case_study_scorecard.ipynb` | Figure 5 (pencil-and-paper audit) | COMPAS checkpoint | yes |
| `06_ablations.ipynb` | Figure S6, ablation tables | ablation checkpoints | yes |
| `07_tables.ipynb` | every LaTeX table | checkpoints, all `results/*.csv` | yes |

Figures 1–3 (the architecture panels) are drawn by
`paper/neurocomputing/tools/make_architecture_figure.py`, which reads the
released Compact checkpoint; they are not produced by a notebook.

Notebook `07` computes the integer-scorecard fidelity on the **test** partition.
The quantization budget `(Lambda_max, tau_tol)` was chosen with that measure, so
the single-split fidelity in Table 5 is a reported operating point, not a
held-out estimate; the manuscript declares this in §4.1.2 and §4.8.

`training/` contains the notebooks that produced the checkpoints. They are the
provenance record and are not needed to rebuild the tables, because the
checkpoints are committed.

## Two notebooks are generated

`01_baselines.ipynb` and `02_cross_validation.ipynb` are written by
`scripts/build_notebooks.py`. That keeps their prose and code in reviewable
Python rather than in an opaque JSON blob. The generator **does not write
outputs**, so after regenerating them run:

```bash
python scripts/refresh_outputs.py            # 01, 02, 07
```

Both steps only read cached artefacts, so they take seconds and never retrain
anything.

## Guard against malformed notebooks

`scripts/check_notebooks.py` verifies that every notebook is valid nbformat 4,
that each `source` is a list of **lines** (not single characters), that every
line ends with a newline, that each code cell compiles, and that no cell points
at a directory removed during the restructure. Run it before committing:

```bash
python scripts/check_notebooks.py        # exit 1 on failure
```

The same assertions run inside `scripts/build_notebooks.py`, so a broken
generator fails loudly instead of writing an unreadable file.

## Long runs

Cross-validation and the baseline benchmark are better driven from the command
line, where they write one row per fold and resume after an interruption:

```bash
python scripts/run_cv.py --models all --datasets german compas compas_noprot adult synthetic
python scripts/run_baselines.py
```
