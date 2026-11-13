#!/usr/bin/env python3
"""Run the 13 baseline models on the single-split protocol.

This script is the single origin of ``results/baselines_<key>.csv`` and of the
test-set predictions used by the McNemar tests. It instantiates the estimators
from :mod:`utils.baselines`, so the single-split benchmark, the cross validation
and the notebooks can never drift apart.

The output schema matches what the table notebook consumes:

    Model, Category, Accuracy, F1_Macro, F1_Weighted, ROC_AUC,
    Train_Time_Sec, Inference_Time_Sec

``Accuracy``, ``F1_*`` and ``ROC_AUC`` are fractions in [0, 1]; the notebook
multiplies by 100. ``Inference_Time_Sec`` is the mean wall-clock time of one
batch prediction over the whole test set, after warm-up, which feeds the
latency table.

Usage
-----
    python scripts/run_baselines.py                 # all four datasets
    python scripts/run_baselines.py --datasets german compas
    python scripts/run_baselines.py --datasets german --force

Outputs
-------
``results/baselines_<key>.csv``
``results/predictions/baselines_<key>_predictions.npz``
``results/benchmark_comparison.csv``  (all datasets combined, if all are run)
"""

from __future__ import annotations

import argparse
import inspect
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.baselines import baseline_specs  # noqa: E402
from utils.datasets import (  # noqa: E402
    get_preprocessed_synthetic_data,
    load_compas_dataset,
    load_german_credit_dataset,
    load_uci_adult_dataset,
)
from utils.experiments import DATASETS, RESULTS_DIR  # noqa: E402
from utils.reproducibility import set_seed  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

PRED_DIR = RESULTS_DIR / "predictions"

#: The four benchmarks of the main table. Variants such as ``compas_noprot``
#: are reported separately and must not be merged into it.
MAIN_DATASETS = {"compas", "german", "adult", "synthetic"}


def single_split(key: str):
    """Return ``(X_train, y_train, X_val, y_val, X_test, y_test)`` for ``key``.

    The split is taken from the dataset registry, so a variant such as
    ``compas_noprot`` reuses the exact same protocol as its parent dataset.
    """
    spec = DATASETS[key]
    if key == "synthetic":
        return get_preprocessed_synthetic_data(
            return_type="numpy", test_size=spec.test_size, val_size=spec.val_size,
        )
    kwargs = {"return_tensor": False, "test_size": spec.test_size,
              "val_size": spec.val_size}
    if "show_info" in inspect.signature(spec.loader).parameters:
        kwargs["show_info"] = False
    return spec.loader(**kwargs)[0]


def measure_inference(model, X, repeats_floor: int = 3, repeats_ceiling: int = 50):
    """Mean seconds for one batch prediction over ``X`` after warm-up.

    Timing is only meaningful on an otherwise idle machine. A busy host inflates
    every measurement by roughly the same factor, which silently invalidates the
    latency table, so this refuses to run when the load average is high.
    """
    try:
        load1 = os.getloadavg()[0]
        cores = os.cpu_count() or 1
    except (OSError, AttributeError):
        load1, cores = 0.0, 1

    # Some estimators (EBM) fan out over all cores while fitting, so the 1-minute
    # load average stays high for a while afterwards. Wait for it to settle
    # instead of aborting a long benchmark; only fail if the host stays busy.
    limit = max(2.0, 0.25 * cores)
    waited = 0.0
    while load1 > limit and waited < 300.0:
        time.sleep(10)
        waited += 10.0
        load1 = os.getloadavg()[0]
    if load1 > limit:
        raise RuntimeError(
            f"Machine is busy (load average {load1:.2f} on {cores} cores, "
            f"limit {limit:.1f}). Latency must be measured on an idle host; "
            "stop other jobs or pass --skip-timing to benchmark accuracy only."
        )

    model.predict(X)  # warm-up
    t0 = time.perf_counter()
    model.predict(X)
    probe = time.perf_counter() - t0
    repeats = repeats_ceiling if probe < 0.05 else (10 if probe < 0.5 else repeats_floor)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        model.predict(X)
        times.append(time.perf_counter() - t0)
    return float(np.mean(times)), float(np.std(times))


def _scores(model, X):
    """Return ``(proba_or_scores, kind)`` for ROC-AUC.

    Many estimators expose calibrated probabilities, but some do not: a linear
    SVM with ``probability=False`` only provides ``decision_function``. Falling
    back to the decision values keeps the AUC column populated instead of
    silently reporting NaN.
    """
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X), "proba"
        except Exception:
            pass
    if hasattr(model, "decision_function"):
        try:
            return model.decision_function(X), "score"
        except Exception:
            pass
    return None, "none"


def run_dataset(key: str, force: bool = False, skip_timing: bool = False) -> pd.DataFrame:
    spec = DATASETS[key]
    csv_path = RESULTS_DIR / f"baselines_{key}.csv"
    pred_path = PRED_DIR / f"baselines_{key}_predictions.npz"
    if csv_path.exists() and not force:
        print(f"[{key}] found, skipped (use --force to refresh)")
        return pd.read_csv(csv_path)

    # Fix the global RNG before fitting. Some estimators draw a random internal
    # validation split (MLPClassifier with early_stopping=True), so without this
    # their scores would depend on how many models were fitted before them.
    set_seed(42)

    X_train, y_train, X_val, y_val, X_test, y_test = single_split(key)
    y_train = np.asarray(y_train).astype(np.int64)
    y_test = np.asarray(y_test).ravel()
    n_classes = int(len(np.unique(y_train)))

    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    print(f"\n[{key}] train={X_train.shape} test={X_test.shape} classes={n_classes}")
    rows, predictions = [], {}
    for name, family, build in baseline_specs(key):
        t0 = time.time()
        model = build().fit(X_train, y_train)
        train_time = time.time() - t0

        pred = np.asarray(model.predict(X_test)).ravel()
        if skip_timing:
            infer_mean = float("nan")
        else:
            infer_mean, _ = measure_inference(model, X_test)

        acc = accuracy_score(y_test, pred)
        f1_macro = f1_score(y_test, pred, average="macro")
        f1_weighted = f1_score(y_test, pred, average="weighted")
        auc = np.nan
        scores, kind = _scores(model, X_test)
        if scores is not None:
            try:
                if n_classes > 2:
                    auc = roc_auc_score(y_test, scores, multi_class="ovr", average="macro")
                elif kind == "proba":
                    auc = roc_auc_score(y_test, np.asarray(scores)[:, 1])
                else:
                    auc = roc_auc_score(y_test, np.asarray(scores).ravel())
            except Exception:
                auc = np.nan

        rows.append({
            "Model": name, "Category": family,
            "Accuracy": acc, "F1_Macro": f1_macro, "F1_Weighted": f1_weighted,
            "ROC_AUC": auc,
            "Train_Time_Sec": round(train_time, 4),
            "Inference_Time_Sec": round(infer_mean, 6),
        })
        predictions[name] = pred
        print(f"   {name:22s} acc={acc*100:6.2f}  f1={f1_macro*100:6.2f}  "
              f"auc={auc*100 if auc == auc else float('nan'):6.2f}  ({train_time:5.1f}s)",
              flush=True)

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    PRED_DIR.mkdir(exist_ok=True)
    df.to_csv(csv_path, index=False)
    np.savez_compressed(pred_path, y_true=y_test, **predictions)
    print(f"   -> {csv_path.name}, {pred_path.name}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=sorted(MAIN_DATASETS))
    ap.add_argument("--force", action="store_true", help="overwrite existing CSVs")
    ap.add_argument("--skip-timing", action="store_true",
                    help="skip latency measurement (use on a busy machine)")
    args = ap.parse_args()

    frames = []
    for key in args.datasets:
        frames.append(run_dataset(key, force=args.force, skip_timing=args.skip_timing).assign(Key=key))

    combined = pd.concat(frames, ignore_index=True)
    if set(args.datasets) == MAIN_DATASETS:
        out = RESULTS_DIR / "benchmark_comparison.csv"
        combined[["Key", "Category", "Model", "Accuracy", "F1_Macro", "ROC_AUC"]].rename(
            columns={"Category": "Family", "Accuracy": "Acc",
                     "F1_Macro": "F1", "ROC_AUC": "AUC"}
        ).assign(
            Acc=lambda d: d.Acc * 100, F1=lambda d: d.F1 * 100, AUC=lambda d: d.AUC * 100,
        ).to_csv(out, index=False)
        print(f"\n-> {out.name} ({len(combined)} rows)")


if __name__ == "__main__":
    main()
