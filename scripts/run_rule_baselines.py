#!/usr/bin/env python3
"""Compare SCORE-Net against rule-list and rule-set learners.

The main benchmark in the manuscript uses 13 standard baselines. That set does
not contain a *rule-list* learner, which is the closest family conceptually: it
also returns an explicit set of ``if-then`` rules. This script fills that gap
with two established implementations from ``imodels`` so that the claim "the
rules we extract are competitive" can be checked against methods that produce
rules directly.

Models
------
RuleFit
    Friedman & Popescu's rule ensemble: gradient-boosted trees generate rules,
    then a sparse linear model over rule indicators is fitted.
SkopeRules
    Skope-rules style conjunction mining with precision/recall filtering.

Both are fitted on the same training partition and scored on the same test
partition as every other baseline.

Usage
-----
    python scripts/run_rule_baselines.py
    python scripts/run_rule_baselines.py --datasets german compas

Output
------
``results/rule_list_comparison.csv``
    dataset, model, accuracy, f1_macro, n_rules (where available), seconds
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

from utils.datasets import (  # noqa: E402
    get_preprocessed_synthetic_data,
    load_compas_dataset,
    load_german_credit_dataset,
    load_uci_adult_dataset,
)
from utils.experiments import DATASETS, RESULTS_DIR  # noqa: E402

warnings.filterwarnings("ignore")


def split_for(key: str):
    spec = DATASETS[key]
    if key == "compas":
        return load_compas_dataset(return_tensor=False, test_size=spec.test_size,
                                   val_size=spec.val_size, show_info=False)[0]
    if key == "german":
        return load_german_credit_dataset(return_tensor=False, test_size=spec.test_size,
                                          val_size=spec.val_size)[0]
    if key == "adult":
        return load_uci_adult_dataset(return_tensor=False, test_size=spec.test_size,
                                      val_size=spec.val_size)[0]
    if key == "synthetic":
        return get_preprocessed_synthetic_data(return_type="numpy",
                                               test_size=spec.test_size,
                                               val_size=spec.val_size)
    raise KeyError(key)


def n_rules(model) -> float:
    """Best-effort rule count for either implementation."""
    for attr in ("rules_", "rules", "estimators_"):
        obj = getattr(model, attr, None)
        if obj is not None:
            try:
                return float(len(obj))
            except TypeError:
                pass
    return float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    args = ap.parse_args()

    try:
        from imodels import RuleFitClassifier, SkopeRulesClassifier
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "imodels is required: pip install imodels"
        ) from exc

    from sklearn.metrics import accuracy_score, f1_score

    rows = []
    for key in args.datasets:
        X_train, y_train, X_val, y_val, X_test, y_test = split_for(key)
        y_train = np.asarray(y_train).astype(np.int64)
        y_test = np.asarray(y_test).ravel().astype(np.int64)
        print(f"\n[{key}] train={X_train.shape} test={X_test.shape}")

        builders = [
            ("RuleFit", lambda: RuleFitClassifier(random_state=42)),
            ("SkopeRules", lambda: SkopeRulesClassifier(random_state=42)),
        ]
        for name, build in builders:
            t0 = time.time()
            try:
                model = build().fit(X_train, y_train)
                pred = np.asarray(model.predict(X_test)).ravel()
                rows.append({
                    "dataset": key,
                    "model": name,
                    "accuracy": accuracy_score(y_test, pred) * 100.0,
                    "f1_macro": f1_score(y_test, pred, average="macro") * 100.0,
                    "n_rules": n_rules(model),
                    "seconds": round(time.time() - t0, 1),
                })
                print(f"   {name:12s} acc={rows[-1]['accuracy']:6.2f}  "
                      f"f1={rows[-1]['f1_macro']:6.2f}  ({rows[-1]['seconds']:5.1f}s)")
            except Exception as exc:  # noqa: BLE001
                print(f"   {name:12s} skipped: {type(exc).__name__}: {str(exc)[:70]}")

    if rows:
        out = RESULTS_DIR / "rule_list_comparison.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"\n-> {out.name} ({len(rows)} rows)")
    else:
        print("\n[warning] no model could be run; CSV not written.")


if __name__ == "__main__":
    main()
