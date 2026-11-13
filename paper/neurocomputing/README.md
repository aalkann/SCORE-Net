# Neurocomputing submission package — SCORE-Net

Elsevier submission version of the SCORE-Net manuscript, prepared for
**Neurocomputing** (ISSN 0925-2312).

This folder is the journal build: Elsevier `elsarticle` class, numbered
references, Elsevier declarations, and tables re-fitted to the Elsevier text
width. It is the only manuscript tree in the repository.

---

## Quick start

```bash
./build.sh                # derive tables, build main.pdf, verify the log
./make_submission.sh      # flatten everything and produce the upload archive
```

Both scripts print a verification summary. A successful run ends with
`CLEAN BUILD` and `abstract word count ... OK`.

---

## Layout

```
neurocomputing/
├── main.tex                     Elsevier front matter + \input of the sections
├── sections/
│   ├── abstract.tex             239 words (journal limit: 250)
│   ├── introduction.tex         (IEEE \IEEEPARstart removed)
│   ├── related_work.tex
│   ├── methodology.tex
│   ├── experiments.tex
│   ├── conclusion.tex
│   └── declarations.tex         CRediT, competing interests, data, GenAI, funding
├── generated/                   12 tables, DERIVED — do not edit by hand
│                                (8 printed in main.pdf, 4 in supplementary.pdf)
├── figures/                     11 vector PDFs
│                                (5 printed in main.pdf, 6 in supplementary.pdf)
├── references.bib
├── prepare_tables.py            derives generated/ from ../../results/tables/
├── audit_figures.py             checks figure type size at printed scale
├── build.sh                     build + verify
├── make_submission.sh           flatten + zip for Editorial Manager
└── submission/
    ├── highlights.txt / highlights.docx   5 bullets, each ≤ 85 characters
    ├── cover_letter.tex/.pdf
    ├── CRediT.md
    ├── data_availability.md
    ├── reviewers.md             how to fill (or skip) the reviewer field
    └── CHECKLIST.md             step-by-step pre-submission checklist
```

---

## Two rules that keep this reproducible
**1. Tables are never edited here.** `generated/*.tex` are derived from the
canonical tables in `results/tables/`, which are produced by
`notebooks/07_tables.ipynb`. If a number changes, change it
in the notebook, re-run it, then run `./build.sh` here. `prepare_tables.py`
normalizes presentation only — it applies `\footnotesize`, removes any
`\resizebox` that would enlarge a narrow table, tightens `\tabcolsep` until the
table fits, and adds a shrink-only `\adjustbox{max width=\linewidth}` wrapper. It
never touches a value.

**2. There is exactly one source per section.** `sections/*.tex` are the only
copy: the Elsevier build is not generated from any other tree, so an edit here
cannot be overwritten by a rebuild.

---

## Requirements and where they come from

| Requirement | Value used here | Source |
|---|---|---|
| Article class | `elsarticle`, option `5p` (two-column, 10 pt) | Elsevier LaTeX instructions (elsevier.com/researcher/author/policies-and-guidelines/latex-instructions); elsarticle v3.5 on CTAN |
| Double-column allowed | yes, for LaTeX submissions | Neurocomputing Guide for Authors |
| Abstract length | ≤ 250 words → **239** | Neurocomputing Guide for Authors |
| Keywords | 1–7, avoid "and"/"of" → **6 single terms** | Neurocomputing Guide for Authors |
| Reference style | numbered, `elsarticle-num` | Neurocomputing Guide for Authors; Elsevier LaTeX instructions |
| DOIs in references | present where known | Neurocomputing Guide for Authors |
| Tables | editable text, no vertical rules, captions + notes | Neurocomputing Guide for Authors |
| Highlights | 3–5 bullets, ≤ 85 chars each, separate file | Neurocomputing Guide for Authors; Elsevier Highlights support page |
| CRediT | required | Neurocomputing Guide for Authors |
| Competing interests | required, via Elsevier Declaration Tool | Neurocomputing Guide for Authors; declarations.elsevier.com |
| Data availability | Option C: deposit + cite, or justify | Neurocomputing Guide for Authors |
| Generative AI declaration | required when AI changes sentence structure | Elsevier GenAI policies for journals |
| No subfolders | all files at one level | Elsevier LaTeX instructions; Editorial Manager LaTeX guide |
| Upload item types | `.tex/.bbl/.bst/.sty/.bib/.cls` → Manuscript; images → Figure | Elsevier LaTeX instructions |

Guide for Authors (canonical):
<https://www.sciencedirect.com/journal/neurocomputing/publish/guide-for-authors>

---

## What changed relative to the IEEE version

| Item | IEEE version | Here |
|---|---|---|
| Class | `IEEEtran`, 10 pt, two-column | `elsarticle`, `5p`, 10 pt, two-column |
| Double-column floats | `dblfloatfix` loaded (IEEE build) | `dblfloatfix` + `\dbltopnumber=3` — see below |
| Abstract | 348 words | 239 words |
| Keywords | `IEEEkeywords` | `elsarticle` `keyword` env, 6 terms |
| Opening | `\IEEEPARstart{T}{abular}` | plain `Tabular` |
| Header | `\markboth{...}` (journal name/volume) | removed; `\journal{Neurocomputing}` |
| Fonts | IEEE default (Times) | `newtxtext`/`newtxmath` (Times, matches Elsevier style) |
| References | `IEEEtran` style | `elsarticle-num`, DOIs as `doi:` |
| Tables | fitted to the IEEE text width (517.9 pt) | 8.03 pt on every table, shrink-only box |
| Declarations | absent | `sections/declarations.tex` |
| Authors | `\thanks{}` footnote | `\author`/`\address`/`\ead`/`\cortext` |
| Orphan figure | `fig_activation_ablation` unused | not copied |

---

## Float placement (why `dblfloatfix` is required)

Elsevier's `elsarticle` does not handle `[b]` on double-column floats
(`table*`, `figure*`), and by default only `\dbltopnumber = 2` such floats may
sit at the top of a page. With 21 `table*`/`figure*` environments the queue backs
up and LaTeX flushes the remainder at `\end{document}` — which puts the figures
and tables **after the reference list**.

`main.tex` therefore loads `dblfloatfix` and raises the limit:

```latex
\usepackage{dblfloatfix}
\setcounter{dbltopnumber}{3}
\renewcommand{\dbltopfraction}{0.9}
\renewcommand{\dblfloatpagefraction}{0.75}
```

Current layout: floats on pages 4–18, references from page 19 (20 pages total). Every float is
cited before or on the page where it is printed (verified for all floats in
`main.pdf`, and separately for the supplementary material). If you remove those four lines, the floats drift behind the
references again — `build.sh` does not catch this automatically, so re-check
page order after any preamble change.

## Verified submissions checks

```bash
./build.sh                      # 0 errors, 0 overfull, 0 undefined, abstract ≤ 250
python audit_figures.py         # figure type size at printed scale (blocking gate)
```

---

## Audit log (most recent full check)

A full manuscript-vs-code audit verified every reported number against
`results/*.csv` and against the checkpoints:

| Claim | Source of truth | Result |
|---|---|---|
| All accuracies, F1, AUC, CIs, McNemar p | `results/benchmark_comparison.csv`, `statistical_tests.csv` | match |
| Rule counts, lengths, sparsity, params | `results/scorenet_metrics.csv` | match |
| Bias-shifting drifts and naive-removal penalty | `results/theorem1_bias_shifting.csv` | match |
| Ablation and capacity sweep | `results/ablation_synthetic.csv` | match |
| Quantization fidelity and integer ranges | `results/quantization_fidelity.csv` | match |
| CV means ($73.02\pm2.71$, $72.72\pm2.78$) | `results/cv_german.csv` | match |
| COMPAS hand audit, no-protected model (+0.43/-0.64, -1.01/+0.93; priors 0/5; age 40/19) | recomputed from checkpoint | match |
| German applicant (margin -0.54, corridor [0.69, 1.87], recourse 0.35/0.42/0.45/0.48) | recomputed from checkpoint | match |
| Jaccard maxima 0.33/0.25/0.22 and the feature pairs | recomputed from checkpoint | match |
| Tier-2 removals: 18 (Synthetic GB), 6 (German Compact), 0 elsewhere | uncompacted vs compacted checkpoints | match |

Corrections made during the audit:

* The synthetic tree-ensemble range was reported as 54.20--65.35 %; the true
  range over all six tree ensembles is 50.85--65.35 % (AdaBoost is 50.85 %).
* "hundreds of trees" on Adult replaced by the measured 100--500 trees.
* "EBM cannot express ..." softened to "represents ... poorly", since EBM with
  pairwise terms is not incapable, only ineffective there.
* Preprocessing paragraph now states explicitly that the one-hot encoder is
  label-free rather than claiming every statistic is training-only.
* The quantization budget is no longer described as "never tuned on the test
  set" (that cannot be guaranteed); it is described as a fixed per-dataset
  operating point, and the limitation is stated in the Limitations section. The
  same applies to the German Credit pruning depth: both are now declared as
  test-informed operating points in §4.1.2 and §4.8.
* Feature-influence coverage is now stated to be computed on the training
  partition, matching the code path (`data_loader=train_loader`).


### Cross-validation audit (2026-10-29)

The German Credit claim was re-examined because a reviewer could reproduce it in
minutes. Running the 13 baselines on **exactly the same 5x5 folds** as SCORE-Net
(`scripts/run_cv.py`) gives:

| Model | mean +- std over 25 folds |
|---|---|
| CatBoost | 75.78 +- 2.68 |
| Gradient Boosting | 75.44 +- 2.91 |
| EBM | 75.32 +- 2.51 |
| Random Forest | 75.30 +- 2.91 |
| SCORE-Net Compact | 73.02 +- 2.71 |
| SCORE-Net Global Best | 72.72 +- 2.78 |

The paired per-fold difference against CatBoost is 2.76 points (Compact) and
3.06 points (Global Best); SCORE-Net wins 2 and 4 of the 25 folds. The single-split
parity (76.00 vs 76.00, p = 1.000) therefore does **not** survive cross-validation.
This is now stated in the abstract, the results section, the limitations and the
conclusion, and Table 3 carries the strongest baseline per dataset on the same folds.

The cause is documented rather than hidden: the German Credit budgets
(H, pruning iterations, dropout, smoothed selection) were chosen on this
1,000-example dataset, whereas the baselines were frozen once and evaluated on
the same folds.

### Other changes in the same pass

* **Proposition, not theorem.** "Exact Bias Shifting Conservation Theorem" was
  renamed to *Exact Bias Shifting for Pruned Rules* and typeset as a
  `proposition`, because it is a direct algebraic identity. Labels changed from
  `thm:bias_shifting` to `prop:bias_shifting`; the table caption is generated.
* **Rule-list learners added.** `scripts/run_rule_baselines.py` runs RuleFit and a
  Skope-rules style miner on the same partitions and writes
  `results/rule_list_comparison.csv`. The Global Best variant exceeds the stronger
  of the two on every dataset, while the Compact variant leads on German Credit
  and UCI Adult but is 0.35 points below RuleFit on COMPAS, which is reported as
  measured. Neither learner supports the 20-class task.
* **Interface text corrected to match the code.** The output layer is trained
  without a bias parameter (not "bias set to zero"); `tau_cov = 0` in every run,
  so the coverage clause never fires.
* **Positioning.** The introduction now frames the contribution as an
  *explainability* result rather than an accuracy frontier, and states that the
  individual components are known; what is examined is whether they compose into
  an end-to-end inspectable model and what that costs.


### Verification pass (2026-10-30)

Three regressions were found and fixed before submission:

1. **SVM AUC was reported as "--" in Table 1.** `SVC(probability=False)` exposes no
   `predict_proba`, so the ROC-AUC column fell back to NaN. `scripts/run_baselines.py`
   now falls back to `decision_function`, and all three SVMs report AUC again
   (71.46 COMPAS, 71.90 German, 88.37 Adult).
2. **Latency values were inflated by 10-75 %** because the benchmark ran while a
   cross-validation job was saturating the CPU. `measure_inference` now waits for
   the load average to settle and aborts rather than record a contaminated number;
   `--skip-timing` is available for accuracy-only runs. The German Credit latency
   table was re-measured on an idle host and reproduces the published values
   exactly.
3. **The rule-list paragraph contradicted itself** ("matches or exceeds on every
   dataset" followed by an admission that the Compact COMPAS model is 0.35 points
   below RuleFit). It now says the Global Best variant leads everywhere while the
   Compact variant leads on German Credit and UCI Adult but trails RuleFit on
   COMPAS.

Also in this pass: `imodels` added to `requirements.txt`, and every
deterministic figure (10 of the 11; all but the latency benchmark, which
re-measures timings) was confirmed to regenerate pixel-identically from the
released artefacts (the active generators are `notebooks/03`–`06` and
`tools/make_architecture_figure.py`);
`scripts/verify_artifacts.py` added to assert that the cross-validation and
single-split encoders agree, that every cached CSV exists, that the LaTeX tables
in `results/tables/` match `generated/` numerically, and that no numeric cell is
NaN.

### Verification pass (2026-11-02)

Three defects were found by comparing the manuscript against re-executed code
rather than against cached artefacts:

1. **The training pipeline could not run.** `utils/train.py` referenced
   `WEIGHT_THRESHOLD` without importing it, so every call to
   `run_iterative_pruning_pipeline` raised `NameError`. Cached checkpoints and
   all tables stayed valid, so nothing surfaced until a retrain was attempted.
   `verify_artifacts.py::check_training_pipeline_runs` now runs a one-iteration
   pipeline on a tiny synthetic problem.
2. **The seed-stability table measured the wrong checkpoint.** It reported the
   validation-selected Global Best model as if it were the Compact one, which
   for COMPAS kept 31.8 of 32 rules — the unpruned model — under the heading
   "compact". Both checkpoints are now measured and reported separately, and the
   rule count is printed next to its initial capacity.
3. **A rule signature ignored the weight threshold.** Building the signature
   from raw weights instead of the thresholded mask inflated the COMPAS overlap
   from 0.13 to 0.60. `check_seed_stability_signatures` pins the property.

The pass also separated three quantities the text had conflated — surviving
connections, rules and distinct features — and documented why the realized
connection count can fall below the geometric schedule of
Equation~\eqref{eq:link_budget} (Eq. 13 in the current build): step-wise
rounding makes the closed form differ from the implemented recurrence by about
one connection, and the two-tier filter then removes empirically dead rules
together with their weights (135 → 124 on the German Credit Compact model).

`scripts/check_all.py` was added so that these checks run as one command: it
purges `__pycache__`, byte-compiles and imports every module, then runs the
notebook, artefact, manuscript-number, figure-typography and (optionally)
manuscript-build gates. Most defects in this project had
the same shape — a change made in one file while a cached artefact still looked
correct — and stale bytecode was twice mistaken for a fixed file.

### Verification pass (2026-11-03): test-informed operating points

The manuscript used to state that "no test example is used to choose an epoch, a
pruning level, a hyperparameter or a checkpoint" and that "none [of the
hyperparameters] is tuned on the test set". Two registered choices made that
claim false:

1. **German Credit pruning depth.** `prune_iters=14` was selected after observing
   the single-split test F1 at 20 iterations. This is a training configuration
   choice, not a checkpoint-selection rule.
2. **Integer-quantization budget.** `(Lambda_max, tau_tol)` was chosen by the
   fidelity/span criterion, and fidelity is measured on the test split
   (`notebooks/07_tables.ipynb` passes `split.loader("test")`). Table 5's
   fidelity is therefore an operating point, not a held-out estimate.

Both are now declared as *test-informed operating points* in §4.1.2 and the
limitations (§4.8), and the protocol sentence was narrowed to what is actually
guaranteed: early stopping, checkpoint selection and the per-iteration pruning
mask use validation only. The depth disclosure is backed by measurement rather
than assertion: `scripts/run_pruning_depth_sensitivity.py` re-runs the same 5x5
folds at 20 iterations and writes `results/pruning_depth_sensitivity.csv`
(72.90 ± 2.47 Global Best, 72.94 ± 2.18 Compact vs 72.72 ± 2.78 and 73.02 ± 2.71
at 14, i.e. at most 0.2 points). `check_manuscript_numbers.py` now verifies both
depths against that CSV, and `verify_artifacts.py` rejects the absolute
phrases if they reappear.

### Verification pass (2026-11-03): figure type size

The six figures listed in the previous version of this README were re-authored
at their printed width. All figures now report 0 % of characters below
6 pt and a minimum of 6.2--7.7 pt (`python audit_figures.py`), and the
typography gate in `scripts/check_all.py` is blocking. Only layout, label
wrapping and font sizes changed; no plotted value was recomputed. The
inference figure was re-laid out from `results/inference_profile.csv` rather
than re-measured, so its latency numbers are the same as in the released
profile.

---

## Figure type size (verified with `python audit_figures.py`)

Elsevier's *Artwork sizing* guidance asks for 7 pt body text and at least 6 pt
for scripts **at the printed size**. All figures (main text and supplement) now satisfy the 6 pt
floor. The six that used to be exported wider than their placement width were
re-authored at the printed width so that no character falls below 6 pt (body
text is 7 pt or more wherever space allows), and LaTeX no longer shrinks them:

| Figure | natural | printed | scale | min pt |
|---|---|---|---|---|
| `fig_ablation_synthetic` | 523.8 pt | 495.9 pt | 0.95 | 7.1 |
| `fig_compas_pencil_paper_scorecard` | 524.8 pt | 490.7 pt | 0.94 | 6.2 |
| `fig_inference_speed_benchmark` | 507.0 pt | 522.0 pt | 1.03 | 7.7 |
| `fig_local_feature_curves` | 391.9 pt | 375.8 pt | 0.96 | 7.7 |
| `fig_pruning_trajectory` | 524.6 pt | 495.9 pt | 0.95 | 7.1 |
| `fig_scorecard_dashboard` | 518.4 pt | 495.9 pt | 0.96 | 6.9 |
| `fig_arch_forward` | 522.0 pt | 522.0 pt | 1.00 | 6.2 |
| `fig_arch_unit` | 287.3 pt | 287.1 pt | 1.00 | 6.2 |
| `fig_arch_scorecard` | 522.0 pt | 522.0 pt | 1.00 | 6.3 |
| `fig_actionable_recourse` | 584.8 pt | 480.2 pt | 0.82 | 7.4 |
| `fig_feature_interaction` | 566.6 pt | 443.7 pt | 0.78 | 7.4 |

The smallest printed type is 6.2 pt (the three architecture panels and the
COMPAS scorecard); the remaining figures are 6.3--7.7 pt. `audit_figures.py` is a
blocking gate in `scripts/check_all.py`, so a future figure exported too wide
fails the check instead of producing only a warning.

Re-authoring notes: simply reducing `figsize` does not fix the problem, because
`bbox_inches="tight"` expands the exported page to include long tick labels.
Each figure was instead laid out at its placement width, long labels were
shortened or wrapped, and mathtext subscripts were removed from legends where
they rendered below the 6 pt floor. The underlying data and every plotted value
are unchanged.

## Pre-submission state

No placeholder remains in the manuscript sources:

```bash
grep -rn "TODO\|University Name\|example.com" --include=*.tex .
# (no output)
```

The affiliation, e-mail addresses, ORCIDs and CRediT statement are final. What a
resubmission still needs is outside the sources: the repository must be public
before the Data availability link is true, and the cover letter needs a date.
