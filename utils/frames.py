"""Frame loaders: return the fully encoded dataset **without** splitting.

Cross-validation needs the whole encoded matrix so that it can build its own
stratified folds, while the single-split benchmark uses the loaders in
:mod:`utils.datasets` that split and scale. Both paths must see byte-identical
features, so the encoders are shared rather than duplicated: COMPAS, German
Credit and the synthetic generator all call the same encoding function that the
splitting loaders use. ``scripts/verify_artifacts.py`` asserts this, so a change
to one path without the other fails loudly.

The COMPAS loader accepts ``drop_protected`` to build the
protected-attribute ablation described in the manuscript.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .datasets import compas_feature_matrix, load_german_credit_frame


def compas_frame(drop_protected: bool = False) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """COMPAS, one-hot encoded, not split.

    Args:
        drop_protected: Remove the sex and race columns. Used by the
            protected-attribute ablation; the default reproduces the main
            benchmark.
    """
    return compas_feature_matrix(drop_protected=drop_protected)


def german_frame() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """German Credit, one-hot encoded, not split."""
    return load_german_credit_frame()


def adult_frame() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """UCI Adult, one-hot encoded, not split."""
    from sklearn.datasets import fetch_openml

    try:
        data = fetch_openml("adult", version=2, as_frame=True, parser="auto")
        df = data.frame
    except Exception as error:  # noqa: BLE001
        raise RuntimeError(
            "UCI Adult could not be downloaded. Cross-validation and the "
            "single split must use the same data; no silent fallback is produced."
        ) from error

    df = df.dropna().reset_index(drop=True)
    target = "class" if "class" in df.columns else df.columns[-1]
    X_df = df.drop(columns=[target])
    y = df[target].astype(str).str.contains(">50K").astype(int).values

    encoded = pd.get_dummies(X_df, drop_first=True)
    return (
        encoded.values.astype(np.float32),
        y.astype(np.int64),
        list(encoded.columns),
    )


def synthetic_frame() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Synthetic 20-class benchmark, not split (the generator is deterministic)."""
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=20000,
        n_features=30,
        n_informative=20,
        n_redundant=0,
        n_classes=20,
        n_clusters_per_class=2,
        random_state=42,
        shuffle=False,
    )
    names = [f"x{i}" for i in range(X.shape[1])]
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64), names


FRAME_LOADERS = {
    "compas": compas_frame,
    "compas_noprot": lambda: compas_frame(drop_protected=True),
    "german": german_frame,
    "adult": adult_frame,
    "synthetic": synthetic_frame,
}
