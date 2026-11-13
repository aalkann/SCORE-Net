"""Shared experiment registry, checkpoint loading, and artifact I/O helpers.

This module lets the training, analysis, and figure notebooks share

* the same data splits (train / validation / test),
* the same checkpoint-loading rules,
* the same result files (``results/``), and
* the same training-history files (``training_history*.json``).

Retraining therefore requires no manual updates to counts, rule names, or
paths in any notebook.

All paths are resolved relative to the project root, so the notebooks find
the same files regardless of their working directory.
"""

from __future__ import annotations

import contextlib
import inspect
import io
from functools import partial
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from .custom_utils import prune_and_compact_model
from .datasets import (
    get_preprocessed_synthetic_data,
    load_compas_dataset,
    load_german_credit_dataset,
    load_uci_adult_dataset,
)
from .frames import FRAME_LOADERS
from .model import SCORENet
from .rules import WEIGHT_THRESHOLD, QuantizationBudget

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURE_DIR = PROJECT_ROOT / "paper" / "neurocomputing" / "figures"


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DatasetSpec:
    """Defines all artifact paths for one benchmark dataset.

    Args:
        test_size: Fraction of the full data used for the test set.
        val_size: Fraction of the training pool (after removing test) used
            for validation. For example, ``test_size=0.1, val_size=1/9``
            gives exactly an 80/10/10 split.
        frame_loader: Loader returning the full data without splitting
            (``() -> (X, y, feature_names)``). Used for cross-validation;
            ``None`` means CV is not supported for this dataset.
        protocol_note: Human-readable description of the split protocol.
            Table and figure captions use this text; numeric values come
            only from ``test_size`` / ``val_size``.

    Note:
        ``h_hidden``, ``dropout``, ``batch_size``, ``prune_iters``,
        ``prune_amount``, and ``selection`` are the dataset's **actual**
        training hyperparameters. Training notebooks read them from here;
        no second copy is kept in the notebooks. The paper reports the same
        values.
    """

    key: str
    display_name: str
    folder: str
    loader: Any
    history_file: str
    baseline_file: str
    checkpoint_global: str = "global-best-weights.pth"
    checkpoint_final: str = "final-iter-best-weights.pth"
    quant_compact: QuantizationBudget = QuantizationBudget(500, 0.05)
    quant_global: QuantizationBudget = QuantizationBudget(500, 0.05)
    test_size: float = 0.2
    val_size: float = 0.1
    frame_loader: Any = None
    protocol_note: str = "72/8/20"
    batch_size: int = 64
    h_hidden: int = 64
    dropout: float = 0.25
    prune_iters: int = 20
    prune_amount: float = 0.2
    selection: str = "strict"
    smoothing_window: int = 3

    def quantization(self, variant: str) -> QuantizationBudget:
        """Return the budget for a variant name (``Compact`` / ``Global Best``)."""
        if variant == "Compact":
            return self.quant_compact
        if variant == "Global Best":
            return self.quant_global
        raise ValueError(
            f"Unknown variant: {variant!r} (must be 'Compact' or 'Global Best')."
        )

    @property
    def directory(self) -> Path:
        return PROJECT_ROOT / self.folder

    @property
    def history_path(self) -> Path:
        return self.directory / self.history_file

    @property
    def global_checkpoint(self) -> Path:
        return self.directory / self.checkpoint_global

    @property
    def final_checkpoint(self) -> Path:
        return self.directory / self.checkpoint_final


# ---------------------------------------------------------------------------
# Quantization budgets (single source of truth)
# ---------------------------------------------------------------------------
# Rule-based adaptive quantization is controlled by two settings:
#
#   * ``Λ_max`` (``max_scale``)   — largest local multiplier to scan,
#   * ``τ_tol`` (``tolerance``)   — acceptable relative rounding error.
#
# Together they determine the integer score range (cognitive load) and
# fidelity to the float model. Settings are defined **per dataset and
# variant**, because the same relative-error tolerance yields very different
# integer ranges at different weight scales: 1,000-sample German Credit and
# 20-class Synthetic are not comparable under one tolerance. Tables, figures,
# and analysis notebooks read from this dictionary; no second copy of the
# constants exists elsewhere.
#
# Selection criterion: keep sample-level fidelity and accuracy (relative to
# the float model) high while keeping the integer score range --- the human
# cognitive load --- as narrow as possible. The interface (rule count, score
# range, fidelity) is reported in tables and figures; the values here are
# only those operating points.
#
# IMPORTANT: fidelity is measured on the test set (notebook 07,
# ``split.loader("test")``). Because the budget was selected on this metric,
# single-split fidelity values are test-observed operating points; the paper
# states this explicitly in §4.1.2 and §4.8. The budget never enters any
# accuracy number; it only affects the integer scorecard in Table 10.
DATASETS: dict[str, DatasetSpec] = {
    "compas": DatasetSpec(
        key="compas",
        display_name="COMPAS",
        folder="models/compas",
        loader=load_compas_dataset,
        history_file="training_history.json",
        baseline_file="baselines_compas.csv",
        frame_loader=FRAME_LOADERS["compas"],
        quant_compact=QuantizationBudget(max_scale=10, tolerance=0.05),
        quant_global=QuantizationBudget(max_scale=10, tolerance=0.05),
        batch_size=64,
        h_hidden=32,
        dropout=0.1,
        prune_iters=19,
        prune_amount=0.2,
        val_size=0.1,
    ),
    # Protected-attribute ablation. Identical to "compas" except that the sex
    # and race one-hot columns are removed before splitting, so the model never
    # sees a legally protected attribute. Training hyperparameters and split
    # ratios are held fixed so that any difference in the reported accuracy is
    # attributable to the restriction alone.
    #
    # The quantization budget is a *post-hoc* reporting choice: it changes only
    # the integer scorecard, never the float model or the cross-validated
    # accuracy. It is therefore selected per variant with the same criterion as
    # every other dataset (highest fidelity at the narrowest integer span), not
    # inherited from "compas". With (10, 0.03) the three Compact rules reach
    # 99.51% prediction agreement on an integer span of only [-1, 3]; the
    # 25-rule Global Best model needs (30, 0.05) for 99.24% on [-7, 7].
    "compas_noprot": DatasetSpec(
        key="compas_noprot",
        display_name="COMPAS (no protected attributes)",
        folder="models/compas_noprot",
        loader=partial(load_compas_dataset, drop_protected=True),
        history_file="training_history.json",
        baseline_file="baselines_compas_noprot.csv",
        frame_loader=FRAME_LOADERS["compas_noprot"],
        quant_compact=QuantizationBudget(max_scale=10, tolerance=0.03),
        quant_global=QuantizationBudget(max_scale=30, tolerance=0.05),
        batch_size=64,
        h_hidden=32,
        dropout=0.1,
        prune_iters=19,
        prune_amount=0.2,
        val_size=0.1,
    ),
    "german": DatasetSpec(
        key="german",
        display_name="German Credit",
        folder="models/german",
        loader=load_german_credit_dataset,
        history_file="training_history.json",
        baseline_file="baselines_german.csv",
        frame_loader=FRAME_LOADERS["german"],
        quant_compact=QuantizationBudget(max_scale=20, tolerance=0.01),
        quant_global=QuantizationBudget(max_scale=30, tolerance=0.05),
        # German Credit has only 1,000 samples. A 72/8/20 split leaves only 80
        # validation samples (one sample = 1.25 points), so 80/10/10 is used:
        # 800/100/100. On this split the validation–test gap of the selected
        # checkpoints is 3.0 (Global Best) and 1.0 (Compact) points
        # (results/statistical_tests.csv). See README, "Protocol".
        test_size=0.1,
        val_size=1.0 / 9.0,
        protocol_note="80/10/10",
        batch_size=64,
        h_hidden=64,
        dropout=0.25,
        selection="smoothed",
        # 14 pruning iterations is the selected operating point for this
        # dataset, chosen by evaluating on the single split; the paper reports
        # it as a test-observed operating point (§4.1.2, §4.8).
        # With 5x5 CV, 20 iterations gives 72.90±2.47 (Global) and
        # 72.94±2.18 (Compact); 14 iterations differs by at most 0.2 points
        # from their 72.72±2.78 and 73.02±2.71 values
        # (results/pruning_depth_sensitivity.csv).
        prune_iters=14,
        prune_amount=0.2,
    ),
    "adult": DatasetSpec(
        key="adult",
        display_name="UCI Adult",
        folder="models/adult",
        loader=load_uci_adult_dataset,
        history_file="training_history.json",
        baseline_file="baselines_adult.csv",
        frame_loader=FRAME_LOADERS["adult"],
        quant_compact=QuantizationBudget(max_scale=40, tolerance=0.02),
        quant_global=QuantizationBudget(max_scale=40, tolerance=0.06),
        batch_size=256,
        h_hidden=64,
        dropout=0.1,
        prune_iters=15,
        prune_amount=0.25,
        val_size = 0.1,
    ),
    "synthetic": DatasetSpec(
        key="synthetic",
        display_name="Synthetic",
        folder="models/synthetic",
        loader=get_preprocessed_synthetic_data,
        history_file="training_history.json",
        baseline_file="baselines_synthetic.csv",
        frame_loader=FRAME_LOADERS["synthetic"],
        quant_compact=QuantizationBudget(max_scale=150, tolerance=0.03),
        quant_global=QuantizationBudget(max_scale=150, tolerance=0.05),
        batch_size=256,
        h_hidden=1024,
        dropout=0.1,
        prune_iters=20,
        prune_amount=0.1,
        val_size = 0.1,
    ),
}


# ---------------------------------------------------------------------------
# Data splits
# ---------------------------------------------------------------------------
@dataclass
class DatasetSplit:
    """Data container holding exactly the splits used during training."""

    spec: DatasetSpec
    X_train: torch.Tensor
    y_train: torch.Tensor
    X_test: torch.Tensor
    y_test: torch.Tensor
    feature_names: list[str]
    scaler: Any
    X_val: Optional[torch.Tensor] = None
    y_val: Optional[torch.Tensor] = None

    @property
    def n_features(self) -> int:
        return int(self.X_train.shape[1])

    @property
    def n_train(self) -> int:
        return int(self.X_train.shape[0])

    @property
    def n_val(self) -> int:
        return int(self.X_val.shape[0]) if self.X_val is not None else 0

    @property
    def n_test(self) -> int:
        return int(self.X_test.shape[0])

    @property
    def n_classes(self) -> int:
        return int(torch.unique(self.y_train).numel())

    @property
    def has_validation(self) -> bool:
        return self.X_val is not None and self.y_val is not None

    def loader(
        self,
        split: str = "train",
        batch_size: int = 256,
        shuffle: bool = False,
    ) -> DataLoader:
        """Build a ``DataLoader`` for the requested split."""
        if split == "train":
            X, y = self.X_train, self.y_train
        elif split == "val":
            if not self.has_validation:
                raise ValueError(
                    "This split has no validation set. Use load_splits(..., val_size>0)."
                )
            X, y = self.X_val, self.y_val
        elif split == "test":
            X, y = self.X_test, self.y_test
        else:
            raise ValueError(f"Unknown split: {split!r} (train/val/test)")

        return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=shuffle)


def _as_tensors(data: tuple) -> tuple:
    """Convert a 4- or 6-element loader output into tensors."""
    return tuple(
        None if value is None else torch.as_tensor(value, dtype=torch.float32 if index % 2 == 0 else torch.long)
        for index, value in enumerate(data)
    )


def load_splits(
    key: str,
    val_size: Optional[float] = None,
    test_size: Optional[float] = None,
) -> DatasetSplit:
    """Load data using the training protocol (stratified, train-only scaler).

    Split ratios come from the dataset registry (``DatasetSpec``), so every
    notebook sees the same split and the ratios are defined in one place.

    Args:
        key: ``compas``, ``german``, ``adult``, or ``synthetic``.
        val_size: Fraction of the training pool used for validation. ``None``
            (default) uses the dataset's registered protocol.
        test_size: Fraction of the full data used for test. ``None``
            (default) uses the dataset's registered protocol.

    Returns:
        :class:`DatasetSplit`
    """
    if key not in DATASETS:
        raise KeyError(f"Unknown dataset: {key!r}. Options: {sorted(DATASETS)}")

    spec = DATASETS[key]
    val_size = spec.val_size if val_size is None else val_size
    test_size = spec.test_size if test_size is None else test_size

    if key == "synthetic":
        data, scaler = get_preprocessed_synthetic_data(
            show_info=False,
            return_scaler=True,
            val_size=val_size,
            test_size=test_size,
        )
        feature_names = [f"x{i}" for i in range(data[0].shape[1])]
    else:
        # ``show_info`` is only accepted by the COMPAS loader; pass it when the
        # callable exposes it so that every dataset shares one code path.
        kwargs = {"val_size": val_size, "test_size": test_size}
        if "show_info" in inspect.signature(spec.loader).parameters:
            kwargs["show_info"] = False
        data, feature_names, scaler = spec.loader(**kwargs)

    tensors = _as_tensors(data)
    if len(tensors) == 6:
        X_train, y_train, X_val, y_val, X_test, y_test = tensors
    else:
        X_train, y_train, X_test, y_test = tensors
        X_val = y_val = None

    return DatasetSplit(
        spec=spec,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        feature_names=list(feature_names),
        scaler=scaler,
    )


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------
def load_checkpoint(path: str | Path, device: str = "cpu") -> SCORENet:
    """Load a model from a ``.pth`` file, inferring the architecture.

    Layer sizes are read from the checkpoint, so ``H_init``, feature counts,
    and class counts are not hard-coded in notebooks. Dropout is set to 0.0
    because it is inactive in evaluation mode.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            "Run the corresponding training notebook first."
        )

    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    if "feature_linear.weight" not in state or "output_linear.weight" not in state:
        raise KeyError(
            f"{path} is not a valid SCORENet checkpoint "
            "(feature_linear.weight / output_linear.weight missing)."
        )

    H_init, n_features = state["feature_linear.weight"].shape
    n_classes = state["output_linear.weight"].shape[0]
    output_bias = "output_linear.bias" in state

    model = SCORENet(
        int(n_features),
        int(H_init),
        int(n_classes),
        dropout_rate=0.0,
        device=device,
        output_bias=output_bias,
    )
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def compact_model(model: SCORENet, train_loader: Optional[DataLoader] = None, verbose: bool = False) -> SCORENet:
    """Compress the model silently (structural + empirical dead-rule removal)."""
    context = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(io.StringIO())
    with context:
        model = prune_and_compact_model(model, train_loader)
    model.eval()
    return model


def load_compact_model(
    path: str | Path,
    train_loader: Optional[DataLoader] = None,
    device: str = "cpu",
    verbose: bool = False,
) -> SCORENet:
    """Load a checkpoint and return the compacted (final) model."""
    return compact_model(load_checkpoint(path, device=device), train_loader, verbose=verbose)


# ---------------------------------------------------------------------------
# Training history
# ---------------------------------------------------------------------------
def load_history(key: str) -> dict:
    """Read the ``training_history.json`` written by the training notebook."""
    spec = DATASETS[key]
    path = spec.history_path
    if not path.exists():
        raise FileNotFoundError(
            f"Training history not found: {path}\n"
            f"Run the training notebook in '{spec.folder}' first. "
            "Training notebooks write the validation accuracy and "
            "sparsity of each iteration to this file."
        )
    with path.open(encoding="utf-8") as handle:
        history = json.load(handle)

    if "iterations" not in history or not history["iterations"]:
        raise ValueError(f"{path} has no valid 'iterations' record.")
    return history


# ---------------------------------------------------------------------------
# Baseline results
# ---------------------------------------------------------------------------
def baseline_path(key: str) -> Path:
    return RESULTS_DIR / DATASETS[key].baseline_file


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------
# On small datasets such as German Credit, a single train/val/test split has
# high variance: in a 100-sample test set, one sample equals 1 point. The
# single-split result is therefore reported alongside repeated stratified
# K-fold cross-validation. The numbers are defined here; notebooks read them.
CV_REPEATS = 5
CV_FOLDS = 5


def cv_results_path(key: str) -> Path:
    """Return the path to the cross-validation results CSV."""
    return RESULTS_DIR / f"cv_{key}.csv"


def load_baselines(key: str) -> pd.DataFrame:
    """Read the baseline notebook's results and add percentage columns."""
    path = baseline_path(key)
    if not path.exists():
        raise FileNotFoundError(
            f"Baseline results not found: {path}\n"
            "Run 'python scripts/run_baselines.py' first."
        )

    df = pd.read_csv(path)

    if "Accuracy (%)" not in df.columns and "Accuracy" in df.columns:
        df["Accuracy (%)"] = df["Accuracy"] * 100.0
    if "F1_Macro (%)" not in df.columns and "F1_Macro" in df.columns:
        df["F1_Macro (%)"] = df["F1_Macro"] * 100.0
    if "ROC_AUC (%)" not in df.columns and "ROC_AUC" in df.columns:
        df["ROC_AUC (%)"] = df["ROC_AUC"] * 100.0
    if "CPU_Inference_ms" not in df.columns and "Inference_Time_Sec" in df.columns:
        df["CPU_Inference_ms"] = df["Inference_Time_Sec"] * 1000.0

    required = {"Model", "Category", "Accuracy (%)", "CPU_Inference_ms"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing expected columns: {sorted(missing)}")

    return df.sort_values("Accuracy (%)", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Figure saving
# ---------------------------------------------------------------------------
def save_figure(fig, name: str, dpi: int = 300) -> tuple[Path, Path]:
    """Save a figure as PDF + PNG under ``paper/neurocomputing/figures``."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = FIGURE_DIR / f"{name}.pdf"
    png_path = FIGURE_DIR / f"{name}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=dpi)
    print(f"Figure saved: {pdf_path} and {png_path}")
    return pdf_path, png_path


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def active_feature_indices(weight_row: np.ndarray, threshold: float = WEIGHT_THRESHOLD) -> list[int]:
    """Return feature indices whose weight in a rule exceeds the threshold."""
    return [int(j) for j in np.where(np.abs(weight_row) > threshold)[0]]


def rule_bounds(
    weight_row: np.ndarray,
    center: float,
    width: float,
    scaler: Any,
    threshold: float = WEIGHT_THRESHOLD,
) -> dict:
    """Return a rule's bounds in the raw (unstandardized) space.

    Single-feature rules produce ``low_raw`` / ``high_raw`` / ``peak_raw``.
    For multi-feature (oblique) rules these fields are ``None``; the rule is
    defined only in projection space.
    """
    indices = active_feature_indices(weight_row, threshold)
    info = {
        "indices": indices,
        "is_single": len(indices) == 1,
        "center": float(center),
        "width": float(width),
        "low_raw": None,
        "high_raw": None,
        "peak_raw": None,
        "half_width_raw": None,
    }

    if not info["is_single"]:
        return info

    j = indices[0]
    w = float(weight_row[j])
    low_z = (center - width) / w
    high_z = (center + width) / w
    if low_z > high_z:
        low_z, high_z = high_z, low_z

    mu, sigma = float(scaler.mean_[j]), float(scaler.scale_[j])
    info.update(
        {
            "low_raw": mu + sigma * low_z,
            "high_raw": mu + sigma * high_z,
            "peak_raw": mu + sigma * (center / w),
            "half_width_raw": sigma * (width / abs(w)),
        }
    )
    return info


def describe_rule(
    weight_row: np.ndarray,
    center: float,
    width: float,
    feature_names: list[str],
    scaler: Any,
    threshold: float = WEIGHT_THRESHOLD,
) -> dict:
    """Describe a rule with its trigger text, formula, and raw bounds."""
    info = rule_bounds(weight_row, center, width, scaler, threshold)
    indices = info["indices"]
    info["feature_names"] = [feature_names[j] for j in indices]

    if info["is_single"]:
        name = info["feature_names"][0]
        info["trigger"] = (
            f"{name} in [{info['low_raw']:.2f}, {info['high_raw']:.2f}] "
            f"(peak {info['peak_raw']:.2f})"
        )
        info["formula"] = (
            f"max(0, 1 - |{name} - {info['peak_raw']:.2f}| / "
            f"{info['half_width_raw']:.2f})"
        )
        return info

    names = info["feature_names"]
    trigger = f"Oblique slab over {len(names)} features: " + ", ".join(names[:3])
    if len(names) > 3:
        trigger += ", ..."
    info["trigger"] = trigger
    info["formula"] = f"max(0, 1 - |proj - {info['center']:.2f}| / {info['width']:.2f})"
    return info


def practical_rule_text(
    description: dict,
    raw_min: np.ndarray,
    raw_max: np.ndarray,
    decimals: int = 1,
) -> str:
    """Convert a rule into a simplified condition using the observed data range.

    For example, if the lower bound is below the observed minimum, only the
    upper bound is reported. This derives practical scorecard conditions such
    as "Age <= 39.4" from the data.
    """
    if not description["is_single"]:
        return "Oblique slab: " + ", ".join(description["feature_names"])

    j = description["indices"][0]
    name = description["feature_names"][0]
    low = description["low_raw"]
    high = description["high_raw"]
    lo_bound = float(raw_min[j]) - 1e-9
    hi_bound = float(raw_max[j]) + 1e-9

    if low <= lo_bound and high >= hi_bound:
        return f"always active for observed {name}"
    if low <= lo_bound:
        return f"{name} <= {high:.{decimals}f}"
    if high >= hi_bound:
        return f"{name} >= {low:.{decimals}f}"
    return f"{low:.{decimals}f} <= {name} <= {high:.{decimals}f}"


def rule_trigger_text(
    weight_row: np.ndarray,
    center: float,
    width: float,
    feature_names: list[str],
    scaler: Any,
    threshold: float = WEIGHT_THRESHOLD,
) -> tuple[str, str]:
    """Backward-compatibility wrapper: return the output of ``describe_rule``."""
    info = describe_rule(weight_row, center, width, feature_names, scaler, threshold)
    return info["trigger"], info["formula"]
