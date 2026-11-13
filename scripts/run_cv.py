#!/usr/bin/env python3
"""Repeated stratified cross-validation for SCORE-Net and all 13 baselines.

The folds are identical for every model on a dataset: the same outer
``StratifiedKFold(repeats x folds, random_state=1000+repeat)`` and the same
inner ``train_test_split(test_size=0.2, random_state=2000+10*repeat+fold)``.
Each fold therefore exposes ``train / validation / test`` partitions to every
model in exactly the same way, which is what makes the per-fold comparison
paired and fair.

* SCORE-Net fits on ``train`` and early-stops on ``validation``.
* Baselines have frozen hyperparameters (no tuning inside a fold), so they fit on
  ``train`` and are scored on ``test``.

Results are appended per fold, so an interrupted run can be resumed and a
partial dataset still yields usable rows.

Usage
-----
    python scripts/run_cv.py --models baselines --datasets german
    python scripts/run_cv.py --models scorenet  --datasets compas adult
    python scripts/run_cv.py --models all       --datasets german compas compas_noprot adult synthetic

Outputs
-------
``results/cv_<key>.csv``             SCORE-Net, one row per (repeat, fold)
``results/cv_baselines_<key>.csv``   baselines, one row per (repeat, fold, model)
``results/cv_summary.csv``           mean +/- std per dataset, model, variant
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.baselines import baseline_specs  # noqa: E402
from utils.experiments import CV_FOLDS, CV_REPEATS, DATASETS, RESULTS_DIR  # noqa: E402
from utils.frames import FRAME_LOADERS  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def fold_indices(y: np.ndarray, repeats: int, folds: int):
    """Yield ``(repeat, fold, train_index, val_index, test_index)`` for the grid."""
    from sklearn.model_selection import StratifiedKFold, train_test_split

    for repeat in range(repeats):
        outer = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1000 + repeat)
        for fold, (pool_idx, test_idx) in enumerate(outer.split(np.zeros(len(y)), y)):
            train_idx, val_idx = train_test_split(
                pool_idx,
                test_size=0.2,
                stratify=y[pool_idx],
                random_state=2000 + repeat * 10 + fold,
            )
            yield repeat, fold, np.asarray(train_idx), np.asarray(val_idx), np.asarray(test_idx)


def _metrics(y_true, y_pred, n_classes: int):
    from sklearn.metrics import accuracy_score, f1_score

    return (
        accuracy_score(y_true, y_pred) * 100.0,
        f1_score(y_true, y_pred, average="macro") * 100.0,
    )


# ---------------------------------------------------------------------------
# SCORE-Net
# ---------------------------------------------------------------------------
def run_scorenet(key: str, repeats: int, folds: int, force: bool = False) -> pd.DataFrame:
    """Reuse :func:`utils.cross_validation.run_repeated_stratified_cv`.

    Results are flushed to ``results/cv_<key>.csv`` after every fold, so an
    interrupted run resumes instead of starting over.
    """
    from utils.cross_validation import run_repeated_stratified_cv
    from utils.experiments import cv_results_path

    out = cv_results_path(key)
    existing: list[dict] = []
    done: set[tuple[int, int]] = set()
    if out.exists() and not force:
        existing = pd.read_csv(out).to_dict("records")
        done = {(int(r["repeat"]), int(r["fold"])) for r in existing}
        print(f"[scorenet:{key}] {len(done)}/{repeats * folds} folds available, remaining folds will run")

    if len(done) == repeats * folds:
        print(f"[scorenet:{key}] complete -> {out.name}")
        return pd.DataFrame(existing)

    rows: list[dict] = list(existing)

    def on_fold(result: dict, _all_rows: list[dict]) -> None:
        if (int(result["repeat"]), int(result["fold"])) in done:
            return
        rows.append(result)
        out.parent.mkdir(exist_ok=True)
        pd.DataFrame(rows).to_csv(out, index=False)

    frame = run_repeated_stratified_cv(
        key, repeats=repeats, folds=folds, progress=True,
        on_fold=on_fold, skip_folds=done or None,
    )
    merged = pd.DataFrame(rows).sort_values(["repeat", "fold"]).reset_index(drop=True)
    merged.to_csv(out, index=False)
    return merged


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
def run_baselines(key: str, repeats: int, folds: int, force: bool = False) -> pd.DataFrame:
    out = RESULTS_DIR / f"cv_baselines_{key}.csv"
    done: set[tuple[int, int, str]] = set()
    rows: list[dict] = []

    if out.exists() and not force:
        prev = pd.read_csv(out)
        rows = prev.to_dict("records")
        done = {(int(r["repeat"]), int(r["fold"]), str(r["model"])) for r in rows}
        print(f"[baselines:{key}] {len(done)} records available, remaining records will run")

    X, y, _ = FRAME_LOADERS[key]()
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    n_classes = int(len(np.unique(y)))
    specs = baseline_specs(key)

    from sklearn.preprocessing import StandardScaler

    for repeat, fold, tr, va, te in fold_indices(y, repeats, folds):
        pending = [(n, f, b) for n, f, b in specs if (repeat, fold, n) not in done]
        if not pending:
            continue

        # Fit the scaler on the fold's training partition only.
        scaler = StandardScaler().fit(X[tr])
        Xtr = scaler.transform(X[tr]).astype(np.float32)
        Xte = scaler.transform(X[te]).astype(np.float32)

        for name, family, build in pending:
            t0 = time.time()
            model = build().fit(Xtr, y[tr])
            pred = np.asarray(model.predict(Xte)).ravel()
            acc, f1 = _metrics(y[te], pred, n_classes)
            rows.append({
                "dataset": key, "model": name, "family": family,
                "repeat": repeat, "fold": fold,
                "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
                "test_accuracy": acc, "test_f1_macro": f1,
                "elapsed_s": round(time.time() - t0, 2),
            })
            print(f"   [{key} r{repeat}f{fold}] {name:22s} acc={acc:6.2f} ({time.time()-t0:5.1f}s)", flush=True)
            pd.DataFrame(rows).to_csv(out, index=False)

    df = pd.DataFrame(rows)
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    return df


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def summarise() -> pd.DataFrame:
    rows = []
    for key in DATASETS:
        spec = DATASETS[key]
        cv_path = RESULTS_DIR / f"cv_{key}.csv"
        if cv_path.exists():
            d = pd.read_csv(cv_path)
            for col, label in [("test_global", "Global Best"), ("test_final", "Compact")]:
                if col in d:
                    v = d[col].to_numpy() * 100.0
                    rows.append({
                        "dataset": spec.display_name, "model": "SCORE-Net", "variant": label,
                        "n_folds": len(v), "mean": v.mean(), "std": v.std(ddof=1),
                    })
        b_path = RESULTS_DIR / f"cv_baselines_{key}.csv"
        if b_path.exists():
            d = pd.read_csv(b_path)
            for (model, family), g in d.groupby(["model", "family"]):
                v = g["test_accuracy"].to_numpy()
                rows.append({
                    "dataset": spec.display_name, "model": model, "variant": family,
                    "n_folds": len(v), "mean": v.mean(), "std": v.std(ddof=1),
                })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["dataset", "mean"], ascending=[True, False])
        df.to_csv(RESULTS_DIR / "cv_summary.csv", index=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--models", choices=["scorenet", "baselines", "all"], default="all")
    ap.add_argument("--repeats", type=int, default=CV_REPEATS)
    ap.add_argument("--folds", type=int, default=CV_FOLDS)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    for key in args.datasets:
        print(f"\n{'='*78}\n{key}: {args.repeats}x{args.folds} CV\n{'='*78}")
        if args.models in ("scorenet", "all"):
            run_scorenet(key, args.repeats, args.folds, force=args.force)
        if args.models in ("baselines", "all"):
            run_baselines(key, args.repeats, args.folds, force=args.force)

    summary = summarise()
    if not summary.empty:
        print("\n" + summary.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
