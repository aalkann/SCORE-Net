#!/usr/bin/env python3
"""Measure how stable the extracted rule set is across random seeds.

A rule set that changes completely between runs is not auditable, so the same
model is retrained on the same split with several seeds and the resulting rules
are compared.

Both retained checkpoints are measured, because they depend on the seed in
different ways:

* **Global Best** is the pruning iteration with the highest validation accuracy.
  It can sit early in the pruning chain, where almost no rule has been removed
  yet, so its rule count is not the compact operating point.
* **Compact** is the final pruning iteration. It always runs the full registered
  schedule, so all seeds are compared at the *same* level of sparsification.
  This is the interpretable model the manuscript reports.

Two questions are answered separately:

1. **Do the same features participate?** For every pair of runs we compare the
   *support* of the rule set: the features that appear in at least one surviving
   rule. Reported as a Jaccard index.
2. **Are the same rules recovered?** Each rule is summarised by the set of
   features it uses; two rules match when their feature sets are identical. The
   reported Jaccard index is over rule signatures, so it measures whether the
   same conjunctions reappear, not merely the same variables.

Test-set accuracy is reported for every seed, so a reader can see whether
stability and accuracy move together.

Usage
-----
    python scripts/run_seed_stability.py --datasets compas german
    python scripts/run_seed_stability.py --datasets compas --seeds 0 1 2 3 4

Output
------
``results/seed_stability_runs.csv``  one row per (dataset, seed, checkpoint)
``results/seed_stability.csv``       pairwise agreement per (dataset, checkpoint)
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.experiments import (  # noqa: E402
    DATASETS,
    RESULTS_DIR,
    compact_model,
    load_splits,
)
from utils.train import run_iterative_pruning_pipeline  # noqa: E402
from utils.reproducibility import set_seed  # noqa: E402
from utils.rules import WEIGHT_THRESHOLD  # noqa: E402

#: Datasets of the main benchmark; the protected-attribute variant is excluded
#: because its rule set is already the subject of a separate comparison.
STABILITY_KEYS = ("compas", "german")
DEFAULT_SEEDS = (0, 1, 2, 3, 4)


def rule_signature(model) -> frozenset[str]:
    """Feature set of one rule, used to decide whether two rules are the same."""
    weight = model.feature_linear.weight.detach().cpu().numpy()
    return frozenset(np.flatnonzero(np.abs(weight) > WEIGHT_THRESHOLD).tolist())


def rule_set(model) -> dict:
    """Rule signatures, feature support and connection count of a fitted model.

    ``n_links`` is the number of surviving projection weights. Iterative
    magnitude pruning removes a fixed fraction of the *remaining* weights at each
    step, so this count follows a geometric schedule and is fixed by the
    registered configuration, independently of the seed. What the seed changes is
    how that fixed connection budget is grouped into rules (``n_rules``) and
    which columns it lands on (``n_features_used``). Reporting all three separates
    "the budget is deterministic" from "the grouping is not".
    """
    weight = model.feature_linear.weight.detach().cpu().numpy()
    active = np.abs(weight) > WEIGHT_THRESHOLD
    signatures = [
        frozenset(np.flatnonzero(row_active).tolist())
        for row_active, keep in zip(active, active.any(axis=1))
        if keep
    ]
    support = frozenset().union(*signatures) if signatures else frozenset()
    informative = [s for s in signatures if s]
    return {
        "signatures": informative,
        "support": support,
        "n_rules": len(informative),
        "n_links": int(active.sum()),
    }


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def multiset_jaccard(a, b) -> float:
    """Jaccard over rule signatures; each distinct signature counts once.

    Signatures arrive either as tuples/lists (fresh run) or as lists read back
    from CSV, so both are normalised to tuples before the set operation.
    """
    sa = {tuple(x) for x in a}
    sb = {tuple(x) for x in b}
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


#: The two checkpoints reported in the manuscript.
CHECKPOINTS = ("Global Best", "Compact")


def summarise(compact, split, device) -> dict:
    """Rule-set statistics and test accuracy of one compacted checkpoint."""
    X_test = split.X_test.to(device)
    y_test = split.y_test.cpu().numpy()
    with torch.no_grad():
        pred = compact(X_test).argmax(dim=1).cpu().numpy()
    accuracy = float((pred == y_test).mean() * 100.0)

    info = rule_set(compact)
    return {
        "n_rules": info["n_rules"],
        "n_links": info["n_links"],
        "n_features_used": len(info["support"]),
        "test_accuracy": accuracy,
        "_signatures": sorted([sorted(s) for s in info["signatures"]]),
        "_support": sorted(info["support"]),
    }


def run_one(key: str, seed: int, device: str) -> list[dict]:
    """Train one model with ``seed`` and summarise BOTH retained checkpoints.

    One training run produces the Global Best and the final-iteration model
    together, so both are summarised here and written as two rows that share
    ``(dataset, seed)``. Returning only one of them was the original defect: the
    validation-selected checkpoint was described as the compact model, which for
    COMPAS kept essentially all ``H_init = 32`` rules.
    """
    spec = DATASETS[key]
    split = load_splits(key)

    set_seed(seed)
    from utils.model import SCORENet

    model = SCORENet(
        split.n_features, spec.h_hidden, split.n_classes,
        dropout_rate=spec.dropout, device=device,
    ).to(device)

    H_init = int(model.H)
    started = time.time()
    global_best, final_iter = run_iterative_pruning_pipeline(
        model,
        split.loader("train"),
        split.loader("val"),
        device,
        num_prune_iters=spec.prune_iters,
        prune_amount_feature=spec.prune_amount,
        selection=spec.selection,
        smoothing_window=spec.smoothing_window,
        history_path=None,
        verbose=False,
    )
    elapsed = time.time() - started

    sources = {"Global Best": global_best, "Compact": final_iter}
    rows = []
    for label in CHECKPOINTS:
        compact = compact_model(sources[label], split.loader("train"))
        rows.append({
            "dataset": key,
            "seed": seed,
            "checkpoint": label,
            "H_init": H_init,
            **summarise(compact, split, device),
            "train_seconds": round(elapsed, 1),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=list(STABILITY_KEYS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    ap.add_argument("--device", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    runs_path = RESULTS_DIR / "seed_stability_runs.csv"
    agree_path = RESULTS_DIR / "seed_stability.csv"

    rows: list[dict] = []
    if runs_path.exists() and not args.force:
        existing = pd.read_csv(runs_path)
        if "checkpoint" not in existing.columns:
            # Written before both checkpoints were measured; it cannot be
            # resumed because the rows do not say which model they describe.
            print("Cache uses the old format (no checkpoint column); "
                  "recompute with --force.")
        else:
            rows = existing.to_dict("records")
            for row in rows:
                row["_signatures"] = json.loads(row["_signatures"]) if isinstance(row["_signatures"], str) else row["_signatures"]
                row["_support"] = json.loads(row["_support"]) if isinstance(row["_support"], str) else row["_support"]

    # One training run yields both checkpoints, so a (dataset, seed) pair counts
    # as done only when every checkpoint is present for it.
    done = {(r["dataset"], int(r["seed"]), r["checkpoint"]) for r in rows}

    for key in args.datasets:
        for seed in args.seeds:
            if all((key, int(seed), label) in done for label in CHECKPOINTS):
                print(f"[{key} seed={seed}] found, skipped")
                continue
            print(f"[{key} seed={seed}] training ...", flush=True)
            produced = run_one(key, int(seed), device)
            for result in produced:
                print(
                    f"    {result['checkpoint']:12s} rules={result['n_rules']:4d}/{result['H_init']:<4}"
                    f"  links={result['n_links']:4d}  features={result['n_features_used']:3d}"
                    f"  acc={result['test_accuracy']:6.2f}%  ({result['train_seconds']:.0f}s)",
                    flush=True,
                )
            # Drop any stale rows for this pair before appending the fresh ones.
            rows = [r for r in rows
                    if not (r["dataset"] == key and int(r["seed"]) == seed)]
            rows.extend(produced)
            out = pd.DataFrame(rows)
            out["_signatures"] = out["_signatures"].apply(json.dumps)
            out["_support"] = out["_support"].apply(json.dumps)
            RESULTS_DIR.mkdir(exist_ok=True)
            out.to_csv(runs_path, index=False)

    # ---- pairwise agreement -------------------------------------------------
    agree_rows = []
    frame = pd.DataFrame(rows)
    for key in args.datasets:
        for label in CHECKPOINTS:
            sub = frame[(frame.dataset == key) & (frame.checkpoint == label)].sort_values("seed")
            if len(sub) < 2:
                continue
            signatures = [json.loads(s) if isinstance(s, str) else s
                          for s in sub["_signatures"]]
            supports = [set(json.loads(s) if isinstance(s, str) else s)
                        for s in sub["_support"]]

            rule_scores, feature_scores = [], []
            for i, j in itertools.combinations(range(len(sub)), 2):
                rule_scores.append(multiset_jaccard(signatures[i], signatures[j]))
                feature_scores.append(jaccard(frozenset(supports[i]), frozenset(supports[j])))

            agree_rows.append({
                "dataset": key,
                "checkpoint": label,
                "n_seeds": len(sub),
                "H_init": int(sub["H_init"].iloc[0]),
                "rule_jaccard_mean": float(np.mean(rule_scores)),
                "rule_jaccard_std": float(np.std(rule_scores, ddof=1)) if len(rule_scores) > 1 else 0.0,
                "feature_jaccard_mean": float(np.mean(feature_scores)),
                "feature_jaccard_std": float(np.std(feature_scores, ddof=1)) if len(feature_scores) > 1 else 0.0,
                "rules_mean": float(sub["n_rules"].mean()),
                "rules_std": float(sub["n_rules"].std(ddof=1)) if len(sub) > 1 else 0.0,
                "links_mean": float(sub["n_links"].mean()),
                "links_std": float(sub["n_links"].std(ddof=1)) if len(sub) > 1 else 0.0,
                "links_min": int(sub["n_links"].min()),
                "links_max": int(sub["n_links"].max()),
                "accuracy_mean": float(sub["test_accuracy"].mean()),
                "accuracy_std": float(sub["test_accuracy"].std(ddof=1)) if len(sub) > 1 else 0.0,
                "pairs": len(rule_scores),
            })

    if agree_rows:
        agree = pd.DataFrame(agree_rows)
        agree.to_csv(agree_path, index=False)
        print("\n" + agree.round(3).to_string(index=False))
        print(f"\n-> {agree_path.relative_to(RESULTS_DIR.parent)}")


if __name__ == "__main__":
    main()
