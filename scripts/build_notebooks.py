#!/usr/bin/env python3
"""Build the two reproducibility notebooks (baselines + cross-validation).

They are generated from this file so that their content is version-controlled
text rather than an opaque binary blob.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "notebooks"


_COUNTER = {"n": 0}


def _cid() -> str:
    _COUNTER["n"] += 1
    return f"cell-{_COUNTER['n']:03d}"


def _source(lines: tuple[str, ...]) -> list[str]:
    """nbformat expects a list of *lines*, each keeping its trailing newline.

    ``list("\\n".join(lines))`` would split the text into single characters and
    produce an unreadable notebook, so the join must happen before splitting on
    line boundaries.
    """
    text = "\n".join(lines)
    if not text.endswith("\n"):
        text += "\n"
    return text.splitlines(keepends=True)


def _assert_readable(nb: dict, name: str) -> None:
    """Fail loudly if a cell was split into characters instead of lines."""
    for index, cell in enumerate(nb["cells"]):
        source = cell["source"]
        if not isinstance(source, list):
            raise AssertionError(f"{name}: cell {index} source is {type(source).__name__}")
        if len(source) > 20:
            singles = sum(1 for line in source if len(line) == 1 and line != "\n")
            if singles / len(source) > 0.3:
                raise AssertionError(
                    f"{name}: cell {index} looks character-split "
                    f"({singles}/{len(source)} single-character entries). "
                    "Join the lines before calling list()."
                )
        for line in source[:-1]:
            if not line.endswith("\n"):
                raise AssertionError(
                    f"{name}: cell {index} has a line without a trailing newline; "
                    "the notebook will render as one long line."
                )


def md(*lines: str) -> dict:
    return {"cell_type": "markdown", "id": _cid(), "metadata": {},
            "source": _source(lines)}


def code(*lines: str) -> dict:
    return {
        "cell_type": "code",
        "id": _cid(),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _source(lines),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


BASELINES = notebook([
    md("# 01 — Baseline models (single split)",
       "",
       "This notebook reproduces **Table 3** of the manuscript: the 13 baseline",
       "models on the single train/validation/test split of each dataset. The same",
       "estimators are also run on the no-protected COMPAS variant (7 inputs instead",
       "of 13), so the single-split benchmark covers both feature sets.",
       "",
       "All estimators are instantiated from `utils.baselines`, which is the single",
       "source of truth for their hyperparameters. The values were chosen manually",
       "per dataset by comparing validation accuracy; no automated grid search was",
       "run. Freezing them in one module guarantees that this notebook, the",
       "cross-validation notebook and the manuscript table cannot drift apart.",
       "",
       "**Protocol.** The scaler is fitted on the training partition only. Baselines",
       "have fixed hyperparameters, so they are fitted on the training partition and",
       "scored on the test partition. Nothing is tuned on test data."),
    code("import sys",
         "from pathlib import Path",
         "",
         "import pandas as pd",
         "",
         "cwd = Path.cwd().resolve()",
         "root = next((p for p in [cwd, *cwd.parents] if (p / 'utils' / 'experiments.py').exists()), cwd)",
         "sys.path.insert(0, str(root))",
         "",
         "from utils.baselines import baseline_specs",
         "from utils.experiments import DATASETS, RESULTS_DIR",
         "from scripts.run_baselines import run_dataset, single_split",
         "",
         "pd.set_option('display.width', 200)"),
    md("## 1. Which models are used, and with which hyperparameters",
       "",
       "Every entry is `(name, family, factory)`. Print the registry so a reader can",
       "verify a reported number without reading the notebook code."),
    code("for key in DATASETS:",
         "    print(f'\\n{key} ({len(baseline_specs(key))} baselines)')",
         "    for name, family, build in baseline_specs(key):",
         "        print(f'  {name:24s} [{family}]')"),
    md("## 2. Single-split partitions",
       "",
       "Recomputed from the same loaders the manuscript uses."),
    code("for key in DATASETS:",
         "    Xtr, ytr, Xva, yva, Xte, yte = single_split(key)",
         "    print(f'{key:10s} train={Xtr.shape[0]:6d}  val={Xva.shape[0]:5d}  test={Xte.shape[0]:6d}  features={Xtr.shape[1]}')"),
    md("## 3. Run (or load) the benchmark",
       "",
       "Set `FORCE = True` to refit every model. With `FORCE = False` the cached",
       "`results/baselines_<key>.csv` is reused, which is what the manuscript uses."),
    code("FORCE = False",
         "",
         "frames = []",
         "for key in DATASETS:",
         "    df = run_dataset(key, force=FORCE)",
         "    frames.append(df.assign(Dataset=DATASETS[key].display_name))",
         "",
         "bench = pd.concat(frames, ignore_index=True)",
         "display_cols = ['Dataset', 'Category', 'Model', 'Accuracy', 'F1_Macro', 'ROC_AUC']",
         "bench[display_cols].assign(",
         "    Accuracy=lambda d: (d.Accuracy * 100).round(2),",
         "    F1_Macro=lambda d: (d.F1_Macro * 100).round(2),",
         "    ROC_AUC=lambda d: (d.ROC_AUC * 100).round(2),",
         ").sort_values(['Dataset', 'Accuracy'], ascending=[True, False]).reset_index(drop=True)"),
    md("## 4. Where the numbers come from",
       "",
       "`results/benchmark_comparison.csv` is what `notebooks/07_tables.ipynb` reads",
       "when it writes Table 3. `results/predictions/baselines_<key>_predictions.npz`",
       "holds the per-example predictions used by the McNemar tests."),
    code("print(sorted(p.name for p in RESULTS_DIR.glob('baselines_*.csv')))",
         "print(sorted(p.name for p in (RESULTS_DIR / 'predictions').glob('*.npz')))"),
])

CV = notebook([
    md("# 02 — Repeated stratified cross-validation (all datasets, all models)",
       "",
       "This notebook is the provenance of **Table 6** (cross-validation, all datasets) and of the",
       "paired model comparison used in the results discussion. It runs, or reloads,",
       "a `5 x 5` repeated stratified cross-validation for every dataset and every",
       "model in the paper.",
       "",
       "## Fold construction (identical for every model)",
       "",
       "For repeat `r` and fold `k`:",
       "",
       "```python",
       "outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=1000 + r)",
       "pool, test = outer.split(X, y)                    # 800 / 200 on German Credit",
       "train, val = train_test_split(pool, test_size=0.2,",
       "                              stratify=y[pool],",
       "                              random_state=2000 + 10*r + k)   # 640 / 160",
       "```",
       "",
       "So each fold exposes `640 / 160 / 200` examples for German Credit, i.e. the",
       "inner split keeps a validation partition for early stopping while the outer",
       "20% stays untouched until scoring. **The same indices are used for SCORE-Net",
       "and for every baseline**, which makes the comparison paired.",
       "",
       "SCORE-Net fits on `train` and early-stops on `val`. Baselines have frozen",
       "hyperparameters, so they fit on `train` and are scored on `test`."),
    code("import sys",
         "from pathlib import Path",
         "",
         "import numpy as np",
         "import pandas as pd",
         "",
         "cwd = Path.cwd().resolve()",
         "root = next((p for p in [cwd, *cwd.parents] if (p / 'utils' / 'experiments.py').exists()), cwd)",
         "sys.path.insert(0, str(root))",
         "",
         "from utils.experiments import DATASETS, RESULTS_DIR",
         "from utils.frames import FRAME_LOADERS",
         "from scripts.run_cv import fold_indices, summarise",
         "",
         "pd.set_option('display.width', 200)"),
    md("## 1. Inspect the folds",
       "",
       "Every model sees exactly these partitions."),
    code("for key in DATASETS:",
         "    X, y, _ = FRAME_LOADERS[key]()",
         "    sizes = [(len(tr), len(va), len(te)) for _, _, tr, va, te in fold_indices(y, 5, 5)]",
         "    uniq = sorted(set(sizes))",
         "    print(f'{key:10s} n={len(y):6d}  fold sizes: {uniq}  ({len(sizes)} folds)')"),
    md("## 2. Run or reload the cross-validation",
       "",
       "Full cost on a 6-core CPU + GTX 1650: the baselines take minutes per dataset,",
       "SCORE-Net takes longer because every fold retrains the whole pruning",
       "trajectory. Results are written incrementally, so an interrupted run can be",
       "resumed with the same command."),
    code("# Equivalent CLI (recommended for long runs, resumable):",
         "#   python scripts/run_cv.py --models all --datasets german compas compas_noprot adult synthetic",
         "",
         "from scripts.run_cv import run_baselines, run_scorenet",
         "",
         "RUN = False          # set True to (re)compute instead of loading",
         "DATASETS_TO_RUN = list(DATASETS)",
         "REPEATS, FOLDS = 5, 5",
         "",
         "if RUN:",
         "    for key in DATASETS_TO_RUN:",
         "        print(f'--- {key}: SCORE-Net')",
         "        run_scorenet(key, REPEATS, FOLDS)",
         "        print(f'--- {key}: baselines')",
         "        run_baselines(key, REPEATS, FOLDS)"),
    md("## 3. Mean test accuracy per model",
       "",
       "`std` is the sample standard deviation over the 25 folds."),
    code("summary = summarise()",
         "summary.round(2)"),
    md("## 4. Paired comparison against the best baseline",
       "",
       "Both models are scored on the *same* fold test sets, so the per-fold",
       "difference is paired. The table reports the mean difference and how often",
       "SCORE-Net wins a fold."),
    code("rows = []",
         "for key, spec in DATASETS.items():",
         "    cvs = RESULTS_DIR / f'cv_{key}.csv'",
         "    cvb = RESULTS_DIR / f'cv_baselines_{key}.csv'",
         "    if not (cvs.exists() and cvb.exists()):",
         "        continue",
         "    sn = pd.read_csv(cvs)[['repeat', 'fold', 'test_global', 'test_final']]",
         "    bl = pd.read_csv(cvb)",
         "    # Only folds completed by both are comparable (a run may be in progress).",
         "    bl = bl.merge(sn[['repeat', 'fold']].drop_duplicates(), on=['repeat', 'fold'])",
         "    if bl.empty:",
         "        print(f'{key}: no common folds yet, skipped')",
         "        continue",
         "    best = bl.groupby('model')['test_accuracy'].mean().idxmax()",
         "    bm = bl[bl.model == best][['repeat', 'fold', 'test_accuracy']]",
         "    pair = sn.merge(bm, on=['repeat', 'fold'], how='inner').sort_values(['repeat', 'fold'])",
         "    for col, label in [('test_global', 'Global Best'), ('test_final', 'Compact')]:",
         "        a = pair[col].to_numpy() * 100",
         "        b = pair['test_accuracy'].to_numpy()",
         "        d = a - b",
         "        rows.append({",
         "            'Dataset': spec.display_name, 'Variant': label,",
         "            'Best baseline': best,",
         "            'SCORE-Net': a.mean(), 'Baseline': b.mean(),",
         "            'Delta (pp)': d.mean(), 'Folds won': int((d > 0).sum()), 'Folds': len(d),",
         "        })",
         "pd.DataFrame(rows).round(2)"),
    md("## 5. Files",
       "",
       "* `results/cv_<dataset>.csv` — SCORE-Net, one row per (repeat, fold)",
       "* `results/cv_baselines_<dataset>.csv` — baselines, one row per (repeat, fold, model)",
       "* `results/cv_summary.csv` — the table in section 3"),
    code("for p in sorted(RESULTS_DIR.glob('cv_*.csv')):",
         "    print(f'{p.name:34s} {len(pd.read_csv(p)):4d} rows')"),
])


def main() -> None:
    for name, nb in (
        ("01_baselines.ipynb", BASELINES),
        ("02_cross_validation.ipynb", CV),
    ):
        _assert_readable(nb, name)          # never write a broken notebook
        (HERE / name).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
        print(f"wrote: notebooks/{name} ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
