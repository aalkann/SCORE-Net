# SCORE-Net — reference implementation

**Self-explaining Convex Operational Rule Engine.** An inherently interpretable
neural network for tabular classification whose hidden units are *localized
convex slabs* (oblique interval rules), together with the exact bias-shifting
step that lets pruned rules be removed without changing predictions, and a
rule-level integer quantization that turns the surviving rules into a
compact integer scorecard (hand-scorable when the rule set is small).

This repository contains the full evidence chain for the manuscript:
the implementation, the trained checkpoints, the cached results, and every
script and notebook that regenerates each table and figure.

---

## Where to start

| I want to… | Go to |
|---|---|
| Understand the method | `utils/model.py`, then `paper/neurocomputing/main.pdf` |
| See a reported number reproduced | `notebooks/07_tables.ipynb` (reads `results/*.csv`) |
| Re-run the baseline benchmark | `notebooks/01_baselines.ipynb` or `scripts/run_baselines.py` |
| Re-run the cross-validation | `notebooks/02_cross_validation.ipynb` or `scripts/run_cv.py` |
| Retrain SCORE-Net | `notebooks/training/<dataset>.ipynb` |
| Regenerate figures | `notebooks/03`–`06`, plus `paper/neurocomputing/tools/make_architecture_figure.py` for Figures 1–3 |
| Check the manuscript against the code | `paper/neurocomputing/README.md` (audit log) |

---

## Repository layout

```
.
├── utils/                     library — the only place with logic
│   ├── rules.py               rule primitive, bias shift, integer scorecard (single source)
│   ├── model.py               SCORENet: bias-free projection + triangular activation
│   ├── train.py               training cycle + iterative magnitude pruning pipeline
│   ├── custom_utils.py        prune_and_compact_model (exact bias shifting)
│   ├── datasets.py            loaders that split and scale
│   ├── frames.py              loaders that return the encoded matrix, unsplit (CV)
│   ├── baselines.py           the 13 baselines, frozen hyperparameters
│   ├── experiments.py         dataset registry, artifact paths, checkpoint I/O
│   ├── statistics.py          stratified bootstrap, McNemar, paired bootstrap
│   ├── cross_validation.py    repeated stratified CV for SCORE-Net
│   ├── analysis.py            feature influence, coverage, rule diagnostics
│   ├── plot_utils.py          plot_exact_actionable_recourse (Figure 9)
│   └── reproducibility.py     set_seed
├── scripts/                   command-line entry points (thin wrappers)
│   ├── run_baselines.py       single-split benchmark  -> results/baselines_*.csv
│   ├── run_cv.py              cross-validation       -> results/cv_*.csv (resumable)
│   ├── run_rule_baselines.py  RuleFit / Skope-rules comparison
│   ├── build_notebooks.py     regenerates notebooks/01 and /02
│   ├── refresh_outputs.py     re-executes 01, 02, 07 to store outputs
│   ├── run_seed_stability.py  five-seed rule stability -> results/seed_stability*.csv
│   ├── run_pruning_depth_sensitivity.py  German depth 14 vs 20 under the same folds
│   ├── check_all.py           every gate in one command (use this)
│   ├── check_notebooks.py     validates every .ipynb (CI-friendly)
│   └── verify_artifacts.py    13 invariant checks over the released artefacts
├── notebooks/
│   ├── 01_baselines.ipynb     Table 3: 13 baselines x 4 datasets (+ no-protected COMPAS)
│   ├── 02_cross_validation.ipynb  5x5 CV, every model, every dataset
│   ├── 03_pruning_figures.ipynb    Figures 4, 6, 8, 9, 10
│   ├── 04_inference_benchmark.ipynb Figure 5 and the latency tables
│   ├── 05_case_study_scorecard.ipynb Figure 7 (pencil-and-paper audit)
│   ├── 06_ablations.ipynb          Figure 11 and the ablation tables
│   ├── 07_tables.ipynb             every LaTeX table
│   └── training/              how each checkpoint was produced
│       ├── compas.ipynb  compas_noprot.ipynb  german.ipynb  adult.ipynb
│       ├── synthetic.ipynb
│       └── ablations/         six ablation runs (capacity, L1, dropout)
├── models/                    checkpoints + training_history.json per dataset
│   ├── compas/  compas_noprot/  german/  adult/  synthetic/  ablations/
├── results/                   cached evidence (version-controlled)
│   ├── baselines_*.csv        per-dataset baseline benchmark
│   ├── benchmark_comparison.csv
│   ├── scorenet_metrics.csv   rule counts, lengths, sparsity, parameters
│   ├── statistical_tests.csv  bootstrap CIs and McNemar tests
│   ├── cv_*.csv, cv_baselines_*.csv, cv_summary.csv
│   ├── ablation_synthetic.csv, theorem1_bias_shifting.csv
│   ├── quantization_fidelity.csv, inference_profile.csv
│   ├── predictions/           per-example predictions for the paired tests
│   └── tables/                generated LaTeX tables
├── legacy/                    retired code, kept for provenance only
└── paper/
    └── neurocomputing/        Elsevier elsarticle submission
```

### Why `utils/` and not a package

Everything that computes is in `utils/`. The notebooks and scripts are thin:
they import, run, and print. That means the manuscript, the tables and the
cross-validation can never disagree about a hyperparameter — there is exactly one
definition, in `utils/baselines.py` and `utils/experiments.py`.

The same rule holds for the model's mathematical primitives. The triangular
activation `a_h(z)`, the exact bias shift of a pruned rule, the per-rule integer
scale search and the continuous-to-scorecard conversion are defined **only** in
`utils/rules.py`. The model, the analysis functions, the pruning code, the plots
and the manuscript figures import them from there. `scripts/verify_artifacts.py`
scans the tree for a second copy and fails the check if one appears.

---

## Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# CUDA 12.4 build of PyTorch:
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

Python 3.12. The versions in `requirements.txt` are the ones used for the
reported experiments. Reported hardware: 6-core / 12-thread x86-64, 16 GB RAM,
NVIDIA GTX 1650 (4 GB), PyTorch 2.6, CUDA 12.4.

---

## Data

| Dataset | Source | Notes |
|---|---|---|
| COMPAS | bundled at `utils/compas-scores-two-years.csv` | ProPublica release; cite Angwin et al. |
| German Credit | OpenML `credit-g` v1 | downloaded on first use |
| UCI Adult | OpenML `adult` v2 | downloaded on first use |
| Synthetic | generated in `utils/datasets.py` | `make_classification`, 20 classes, fixed seed 42 |

If OpenML is unreachable the loaders raise instead of substituting data.
`allow_fallback=True` exists only for smoke tests and must not be used for
reported results.

---

## Protocol

**Single split.** Stratified, seed 42. COMPAS, Adult and Synthetic use 72/8/20;
German Credit uses 80/10/10 because 1,000 examples leave only 80 validation
examples under 72/8/20 (one example = 1.25 percentage points).

**Cross-validation.** 5x5 repeated stratified CV. Outer fold: 80% pool / 20%
test. Inner split of the pool: 80% train / 20% validation. So each fold exposes
`train / validation / test = 640 / 160 / 200` on German Credit. **The same fold
indices are used for SCORE-Net and for every baseline**, which makes the
comparison paired. SCORE-Net early-stops on the validation partition; baselines
have frozen hyperparameters and are fitted on the training partition only.

**No test leakage.** Enforced in code, not by convention:

* `utils/train.py::train_cycle` raises if `val_loader` is missing, so the test
  partition cannot be used for early stopping.
* `utils/datasets.py::_scale_splits` fits the scaler on the training partition
  only; `utils/cross_validation.py` and `scripts/run_cv.py` refit it inside every
  fold.
* `utils/datasets.py::_split_dataset` always splits with `stratify=y`.

**Baseline hyperparameters.** Chosen manually per dataset by comparing
validation performance; the winning configuration is frozen in
`utils/baselines.py`. There is no automated grid search, and nothing was tuned on
test data. The registry reproduces the published single-split numbers exactly
(verified: maximum absolute difference 0.000000).

---

## Reproducing the results

```bash
# 1. Baselines, single split (~20-40 min)
python scripts/run_baselines.py

# 2. Cross-validation, every model, every dataset (hours; resumable)
python scripts/run_cv.py --models all --datasets german compas compas_noprot adult synthetic

# 3. Figures and tables from the cached results
cd notebooks && jupyter lab    # run 03, 05, 06, 07; 04 re-measures latency on an idle host

# 4. Validate the repository before committing
python scripts/check_all.py            # every gate, one command
python scripts/check_all.py --build    # ... and rebuild the manuscript
```

`scripts/check_all.py` is the gate to use after touching anything shared. It
deletes every `__pycache__`, byte-compiles and imports each module, then runs the
notebook, artefact, manuscript-number, figure-typography and (with `--build`)
manuscript-build checks. The reason it exists is that most
defects found during preparation had one shape: a change was made in one file, a
*cached* artefact still looked correct, and the breakage surfaced much later.

Two concrete examples, both now covered by tests:

* `utils/train.py` referenced `WEIGHT_THRESHOLD` without importing it. Every call
  to the training pipeline raised `NameError`, yet all cached checkpoints and
  tables stayed valid, so nothing noticed until a retrain was attempted.
  `verify_artifacts.py::check_training_pipeline_runs` now runs a one-iteration
  pipeline on a tiny problem.
* A rule signature was built from raw weights instead of the thresholded mask,
  which inflated a reported overlap from 0.13 to 0.60. The number was plausible
  and nothing recomputed it. `check_seed_stability_signatures` now pins it.

Rule of thumb: **never trust a cached artefact after a code change**. Re-run the
producer (`scripts/run_*.py`, or the notebook) and then `check_all.py`.

Latency is only meaningful on an idle host. `scripts/run_baselines.py` refuses to
time on a busy machine (it waits, then aborts) and accepts `--skip-timing` when
only accuracy is needed. `notebooks/04_inference_benchmark.ipynb` re-measures
SCORE-Net on CPU and GPU (forward pass on a preloaded tensor, 50 warmup plus 100
batch / 500 single-query runs); it does not enforce the idle-host guard, so run
it on an otherwise idle machine. Its CPU numbers are comparable to the baseline
CSV only when both are measured on the same host and session, with the same
thread budget.

Cross-validation writes one row per fold, so an interrupted run resumes where it
stopped. To retrain SCORE-Net itself, run the notebooks in `notebooks/training/`;
each writes `models/<dataset>/global-best-weights.pth`,
`final-iter-best-weights.pth` and `training_history.json`.

---

## What each cached file supports

| File | Manuscript artefact |
|---|---|
| `results/baselines_*.csv`, `benchmark_comparison.csv` | Table 1; baseline latency in Table~S4 |
| `results/statistical_tests.csv` | Table 2 (bootstrap CIs, validation--test gaps, McNemar) |
| `results/scorenet_metrics.csv` | Table 1 (SCORE-Net rows), Table 4 (rules, lengths, sparsity, parameters) |
| `results/cv_*.csv`, `results/cv_baselines_*.csv` | Table 3; SCORE-Net timing in Table~S3 |
| `results/cv_compas_noprot.csv`, `cv_baselines_compas_noprot.csv` | Table 3 (no-protected rows) and Table~S2 |
| `results/inference_profile.csv` | Table 4 (`tab:computational_efficiency`) and Table~S4 (`tab:cpu_latency_baselines`) |
| `results/quantization_fidelity.csv` | Table 5 (`tab:scorecard_fidelity`) |
| `results/seed_stability.csv` | Table 6 (`tab:seed_stability`) |
| `results/ablation_synthetic.csv` | Table 7 (`tab:ablation_synthetic`) |
| `results/theorem1_bias_shifting.csv` | Table 8 (Proposition 1 validation) |
| `results/pruning_depth_sensitivity.csv` | §4.8 disclosure: German CV at 14 vs 20 pruning iterations |
| `results/tables/*.tex` | every LaTeX table (canonical copy) |
| `results/predictions/*.npz` | McNemar tests (paired per example) |

Table~S3 (`tab:train_cost`) has no cached CSV of its own: notebook 07 derives it
from `results/cv_*.csv` (`elapsed_s`) and `results/baselines_*.csv`
(`Train_Time_Sec`). Table~S1 (`tab:dataset_summary`) has no cached CSV either; it
is computed from the registered splits.

---

## Known limitations of the released evidence

These are stated in the manuscript as well:

* **Single seed** for the single-split numbers. Seed variability is characterized
  by the $5\times5$ cross-validation on all datasets and by the five-seed rule
  stability experiment (Table 11, COMPAS and German Credit). That experiment
  quantifies rule-level stability: feature support is nearly stable for the
  validation-selected checkpoint, while individual rule signatures are not.
* **Small test sets.** German Credit has 100 test examples; one example equals one
  percentage point. Bootstrap intervals, McNemar tests and cross-validation are
  reported together for this reason.
* **Two test-informed operating points.** The German Credit pruning depth
  (14 instead of 20 iterations) and the quantization budget
  `(Lambda_max, tau_tol)` were fixed with the single split available, so the
  single-split figures and the reported fidelity are operating points rather
  than held-out estimates. The manuscript declares both in §4.1.2 and §4.8.
  Repeating the German cross-validation at 20 iterations gives 72.90 ± 2.47
  (Global Best) and 72.94 ± 2.18 (Compact) against 72.72 ± 2.78 and 73.02 ± 2.71
  at 14, i.e. the depth choice moves the CV mean by at most 0.2 points
  (`results/pruning_depth_sensitivity.csv`, reproducible with
  `scripts/run_pruning_depth_sensitivity.py`).
* **Synthetic benchmark** is generated from localized geometry that matches the
  inductive bias of convex slabs; it is a controlled demonstration, not evidence
  of general superiority.
* **No fairness evaluation.** COMPAS is used to demonstrate transparency on a
  high-stakes dataset; fairness is out of scope.

---

## Licence and citation

MIT — see `LICENSE`. Citation metadata in `CITATION.cff`.

Archived release `v1.0.0`: [10.5281/zenodo.24217264](https://doi.org/10.5281/zenodo.24217264)
(all versions: [10.5281/zenodo.24217263](https://doi.org/10.5281/zenodo.24217263)).
