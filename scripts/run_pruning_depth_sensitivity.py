#!/usr/bin/env python3
"""Quantify how much the German Credit CV mean depends on the pruning depth.

The registered German Credit configuration uses 14 pruning iterations. That
depth was selected on the single split, so the manuscript declares it as a
test-informed operating point (Sections 4.1.2 and 4.8) rather than claiming that
every hyperparameter is test-independent. This script makes the disclosure
checkable: it re-runs the **same** 5x5 folds with 20 iterations and reports the
result next to the registered 14-iteration values.

Nothing here changes a published number. The registered run is read from
``results/cv_german.csv``; only the alternative depth is trained.

Output
------
``results/pruning_depth_sensitivity.csv`` — one row per (variant, depth) with
the mean and standard deviation of cross-validated test accuracy.

Usage
-----
    python scripts/run_pruning_depth_sensitivity.py

Takes roughly the same time as one SCORE-Net cross-validation on German Credit
(about 6 minutes on the reported hardware).
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from utils import experiments as E  # noqa: E402

OUT = E.RESULTS_DIR / "pruning_depth_sensitivity.csv"
REGISTERED = E.DATASETS["german"].prune_iters
ALTERNATIVE = 20


def _stats(frame: pd.DataFrame, depth: int, source: str) -> list[dict]:
    rows = []
    for column, variant in (("test_global", "Global Best"), ("test_final", "Compact")):
        values = frame[column].to_numpy(dtype=float) * 100.0
        rows.append(
            {
                "dataset": "German Credit",
                "variant": variant,
                "prune_iters": depth,
                "n_folds": int(values.size),
                "cv_mean": values.mean(),
                "cv_std": values.std(ddof=1),
                "source": source,
            }
        )
    return rows


def main() -> int:
    registered_path = E.cv_results_path("german")
    if not registered_path.exists():
        print(f"missing {registered_path}; run scripts/run_cv.py first")
        return 1

    rows = _stats(pd.read_csv(registered_path), REGISTERED, str(registered_path.name))

    # Train the alternative depth on exactly the same folds.
    original = E.DATASETS["german"]
    E.DATASETS["german"] = dataclasses.replace(original, prune_iters=ALTERNATIVE)
    try:
        from utils.cross_validation import run_repeated_stratified_cv

        frame = run_repeated_stratified_cv("german", progress=False)
    finally:
        E.DATASETS["german"] = original

    rows += _stats(frame, ALTERNATIVE, "rerun with dataclasses.replace")

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(df.round(3).to_string(index=False))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
