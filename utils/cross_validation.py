"""Repeated stratified K-fold cross-validation.

On small datasets (especially German Credit, $M=1.000$) a single
train/validation/test split has high variance. This module holds out the test
set in the **outer** folds and builds the validation set in the **inner** fold,
independent of the split protocol:

* Outer level: stratified K-fold with ``folds`` folds; each fold leaves 200
  test samples (for German) and a training pool of 800 samples.
* Inner level: a stratified fraction ``inner_val_size`` of the training pool is
  held out for validation. Early stopping and Global Best selection see only
  this set; the outer test set never takes part in selection.

All (repeat, fold) runs use the same training pipeline
(``run_iterative_pruning_pipeline``). Results are written to
``results/cv_<key>.csv``, and the table/figure notebooks read them from there.

Design note
-----------
The full dataset comes from ``DatasetSpec.frame_loader``. This guarantees that
cross-validation shares **the same** preprocessing as the train/test split; no
separate copy of the preprocessing code is kept.
"""

from __future__ import annotations

import argparse
import time

from typing import Callable

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from .experiments import CV_FOLDS, CV_REPEATS, DATASETS, compact_model, cv_results_path
from .model import SCORENet
from .reproducibility import set_seed
from .train import run_iterative_pruning_pipeline


def _accuracy(model: SCORENet, X: torch.Tensor, y: torch.Tensor, device: str) -> float:
    model.eval()
    with torch.no_grad():
        predictions = model(X.to(device)).argmax(dim=1).cpu()
    return float((predictions == y).float().mean().item())


def _run_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    spec,
    seed: int,
    device: str,
) -> dict:
    """Train a single (repeat, fold) run and evaluate on the outer test set."""
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    X_train_t = torch.as_tensor(X_train)
    y_train_t = torch.as_tensor(y_train, dtype=torch.long)
    X_val_t = torch.as_tensor(X_val)
    y_val_t = torch.as_tensor(y_val, dtype=torch.long)
    X_test_t = torch.as_tensor(X_test)
    y_test_t = torch.as_tensor(y_test, dtype=torch.long)

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=spec.batch_size,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(
        TensorDataset(X_val_t, y_val_t), batch_size=spec.batch_size
    )

    # IMPORTANT: the data loader and split operations may consume the global RNG.
    # Re-seed here so model initialization is identical on every run
    # (see README, "Reproducibility").
    set_seed(seed)
    model = SCORENet(
        X_train.shape[1],
        spec.h_hidden,
        int(np.unique(y_train).size),
        dropout_rate=spec.dropout,
        device=device,
    ).to(device)

    started = time.time()
    global_best, final_iter = run_iterative_pruning_pipeline(
        model,
        train_loader,
        val_loader,
        device,
        prune_amount_feature=spec.prune_amount,
        num_prune_iters=spec.prune_iters,
        history_path=None,
        selection=spec.selection,
        smoothing_window=spec.smoothing_window,
        verbose=False,
    )
    elapsed = time.time() - started

    # The models reported in the paper are the compressed (compact) models.
    global_best.eval()
    final_iter.eval()
    global_compact = compact_model(global_best, train_loader)
    final_compact = compact_model(final_iter, train_loader)

    result = {
        "val_global": _accuracy(global_best, X_val_t, y_val_t, device),
        "val_final": _accuracy(final_iter, X_val_t, y_val_t, device),
        "test_global": _accuracy(global_compact, X_test_t, y_test_t, device),
        "test_final": _accuracy(final_compact, X_test_t, y_test_t, device),
        "test_global_raw": _accuracy(global_best, X_test_t, y_test_t, device),
        "test_final_raw": _accuracy(final_iter, X_test_t, y_test_t, device),
        "H_global": int(global_compact.H),
        "H_final": int(final_compact.H),
        "elapsed_s": round(elapsed, 1),
    }

    del model, global_best, final_iter, global_compact, final_compact
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def run_repeated_stratified_cv(
    key: str,
    repeats: int = CV_REPEATS,
    folds: int = CV_FOLDS,
    inner_val_size: float = 0.2,
    seed: int = 42,
    device: str | None = None,
    progress: bool = True,
    on_fold: "Callable[[dict, list[dict]], None] | None" = None,
    skip_folds: "set[tuple[int, int]] | None" = None,
) -> pd.DataFrame:
    """Run repeated stratified K-fold CV for one dataset.

    Args:
        key: Dataset key in ``DATASETS``.
        repeats: Number of outer repeats (each repeat uses a different fold split).
        folds: Number of folds.
        inner_val_size: Fraction of the training pool held out for validation.
        seed: Base seed; a seed is derived for each (repeat, fold).
        device: ``"cuda"`` / ``"cpu"``; chosen automatically when ``None``.
        progress: Print one progress line per fold.
        on_fold: Optional callback. Called with ``(new_row, all_rows)`` right
            after each fold finishes, so the caller can write the result to disk
            and resume an interrupted long run.
        skip_folds: ``(repeat, fold)`` pairs to skip. Used when resuming a
            partial run: these folds are not trained at all.

    Returns:
        A :class:`pandas.DataFrame` with one row per (repeat, fold) run.
    """
    if key not in DATASETS:
        raise KeyError(f"Unknown dataset: {key!r}. Options: {sorted(DATASETS)}")

    spec = DATASETS[key]
    if spec.frame_loader is None:
        raise ValueError(
            f"{spec.display_name} does not support cross-validation: "
            "DatasetSpec.frame_loader is not defined."
        )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    X, y, _ = spec.frame_loader()
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)

    if progress:
        print(
            f"[CV] {spec.display_name}: n={len(y)}, features={X.shape[1]}, "
            f"{repeats}x{folds} fold, selection={spec.selection!r}",
            flush=True,
        )

    rows = []
    for repeat in range(repeats):
        outer = StratifiedKFold(
            n_splits=folds, shuffle=True, random_state=1000 + repeat
        )
        for fold, (train_index, test_index) in enumerate(outer.split(X, y)):
            if skip_folds and (repeat, fold) in skip_folds:
                continue
            X_pool, y_pool = X[train_index], y[train_index]
            X_test, y_test = X[test_index], y[test_index]

            X_train, X_val, y_train, y_val = train_test_split(
                X_pool,
                y_pool,
                test_size=inner_val_size,
                stratify=y_pool,
                random_state=2000 + repeat * 10 + fold,
            )

            fold_seed = 10000 + repeat * 100 + fold
            result = _run_fold(
                X_train,
                y_train,
                X_val,
                y_val,
                X_test,
                y_test,
                spec=spec,
                seed=fold_seed,
                device=device,
            )
            result.update(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "seed": fold_seed,
                    "n_train": int(len(y_train)),
                    "n_val": int(len(y_val)),
                    "n_test": int(len(y_test)),
                }
            )
            rows.append(result)
            if on_fold is not None:
                on_fold(result, rows)

            if progress:
                print(
                    f"  repeat {repeat} fold {fold}: "
                    f"val={result['val_global']:.3f} "
                    f"test={result['test_global']:.3f} "
                    f"H={result['H_global']} ({result['elapsed_s']}s)",
                    flush=True,
                )

    return pd.DataFrame(rows)


def summarize_cv(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize CV rows as mean / std / 95% CI / min / max."""
    metrics = [
        "val_global",
        "val_final",
        "test_global",
        "test_final",
        "test_global_raw",
        "test_final_raw",
    ]
    rows = []
    for metric in metrics:
        values = frame[metric].to_numpy(dtype=float) * 100.0
        n = values.size
        std = values.std(ddof=1) if n > 1 else 0.0
        half_width = 1.96 * std / np.sqrt(n) if n > 1 else 0.0
        rows.append(
            {
                "Metric": metric,
                "Mean": values.mean(),
                "Std": std,
                "CI95": half_width,
                "Min": values.min(),
                "Max": values.max(),
                "N": n,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Repeated stratified K-fold CV")
    parser.add_argument("--key", default="german")
    parser.add_argument("--repeats", type=int, default=CV_REPEATS)
    parser.add_argument("--folds", type=int, default=CV_FOLDS)
    parser.add_argument("--inner-val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    frame = run_repeated_stratified_cv(
        args.key,
        repeats=args.repeats,
        folds=args.folds,
        inner_val_size=args.inner_val_size,
        seed=args.seed,
    )

    output = cv_results_path(args.key) if not args.out else args.out
    frame.to_csv(output, index=False)
    print(f"\n[CV] Results written to: {output}")

    summary = summarize_cv(frame)
    print(summary.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
