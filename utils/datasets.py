"""Data loading and preprocessing helpers.

This module loads the four datasets used in the paper (COMPAS, German Credit,
UCI Adult and Synthetic). All loaders follow the same protocol:

1. Split the data into train/validation/test.
2. Stratify the splits to preserve the class distribution.
3. Fit ``StandardScaler`` on the training split only; validation and test
   splits are only transformed.

This protocol prevents the test set from leaking into preprocessing or model
selection.

Return contract
---------------
With ``val_size=0`` (default) the loader returns a 4-tuple::

    (X_train, y_train, X_test, y_test), feature_names, scaler

With ``val_size>0`` it returns a 6-tuple::

    (X_train, y_train, X_val, y_val, X_test, y_test), feature_names, scaler

``val_size`` is a fraction of the training pool (the part left after the test
split). For example ``test_size=0.2, val_size=0.1`` → 72% train, 8% validation,
20% test.

Split ratios are not fixed in this module: each dataset defines its protocol
in the ``DatasetSpec`` registry in ``utils/experiments.py``, and
``load_splits(key)`` reads it from there. COMPAS, UCI Adult and Synthetic use
72/8/20; German Credit uses 80/10/10 with 1,000 samples.
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.datasets import fetch_openml, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _split_dataset(X, y, test_size, val_size, random_state):
    """Split the data into (train, [val], test) with stratification."""
    X_pool, X_test, y_pool, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )

    if val_size is not None and val_size > 0:
        X_train, X_val, y_train, y_val = train_test_split(
            X_pool, y_pool, test_size=val_size, random_state=random_state,
            stratify=y_pool,
        )
        return X_train, y_train, X_val, y_val, X_test, y_test

    return X_pool, y_pool, None, None, X_test, y_test


def _scale_splits(splits):
    """Fit the scaler on the training split only, then scale all splits."""
    X_train, y_train, X_val, y_val, X_test, y_test = splits

    scaler = StandardScaler()
    scaler.fit(X_train)

    def _transform(X):
        if X is None:
            return None
        return scaler.transform(X).astype(np.float32)

    return (
        _transform(X_train), y_train,
        _transform(X_val), y_val,
        _transform(X_test), y_test,
    ), scaler


def _to_tensors(splits):
    """Convert splits to (float32 X, long y) tensors."""
    converted = []
    for index, array in enumerate(splits):
        if array is None:
            converted.append(None)
        elif index % 2 == 0:
            converted.append(torch.tensor(array, dtype=torch.float32))
        else:
            converted.append(torch.tensor(array, dtype=torch.long))
    return tuple(converted)


def _describe(data, n_features):
    names = ["Train", "Val", "Test"] if len(data) == 6 else ["Train", "Test"]
    for name, X in zip(names, data[0::2]):
        if X is not None:
            print(f"  {name:<5}: {tuple(X.shape)}")
    print(f"  Feature count: {n_features}")


def _raise_download_error(dataset_name, error, allow_fallback):
    if allow_fallback:
        print(
            f"[WARNING] {dataset_name} could not be downloaded ({error}). "
            "Synthetic fallback data is used because allow_fallback=True; "
            "results from it are not comparable to reported results."
        )
        return
    raise RuntimeError(
        f"{dataset_name} could not be downloaded from OpenML: {error}. "
        "The silent synthetic fallback was removed because results produced "
        "with the wrong dataset could be reported unnoticed. Check your "
        "internet connection, use the OpenML cache, or pass "
        "allow_fallback=True only for smoke tests."
    ) from error


# ---------------------------------------------------------------------------
# Synthetic
# ---------------------------------------------------------------------------
def get_preprocessed_synthetic_data(
    show_info=False,
    return_scaler=False,
    return_type="tensor",
    n_samples=20000,
    n_features=30,
    n_classes=20,
    test_size=0.2,
    val_size=0.0,
    random_state=42,
):
    """Generate and preprocess the synthetic multi-class dataset.

    Note: ``n_samples``, ``n_features`` and ``n_classes`` are actually used.
    """
    if n_classes < 2:
        raise ValueError("n_classes must be at least 2.")
    if n_features < 1:
        raise ValueError("n_features must be at least 1.")

    n_informative = min(20, n_features)
    if n_classes * 2 > 2 ** n_informative:
        raise ValueError(
            f"n_classes={n_classes} and n_clusters_per_class=2, "
            f"n_informative={n_informative} is too small; increase n_features."
        )

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=0,
        n_classes=n_classes,
        n_clusters_per_class=2,
        random_state=random_state,
        shuffle=False,
    )

    splits = _split_dataset(X, y, test_size, val_size, random_state)
    splits, scaler = _scale_splits(splits)

    if return_type == "tensor":
        splits = _to_tensors(splits)
    elif return_type != "numpy":
        raise ValueError("return_type must be 'tensor' or 'numpy'.")

    if show_info:
        print("[Synthetic Dataset]")
        _describe(splits, n_features)
        print(f"  Class count: {n_classes}")

    if val_size and val_size > 0:
        data = splits  # (Xtr, ytr, Xval, yval, Xte, yte)
    else:
        X_train, y_train, _, _, X_test, y_test = splits
        data = (X_train, y_train, X_test, y_test)

    if return_scaler:
        return data, scaler
    return data


# ---------------------------------------------------------------------------
# UCI Adult
# ---------------------------------------------------------------------------
def load_uci_adult_dataset(
    test_size=0.2,
    random_state=42,
    return_tensor=True,
    val_size=0.0,
    allow_fallback=False,
):
    """Load the UCI Adult Census Income dataset (binary classification)."""
    print("[Data] Loading UCI Adult Census Income...")
    try:
        data = fetch_openml("adult", version=2, as_frame=True, parser="auto")
        df = data.frame
    except Exception as error:  # noqa: BLE001 - keep the error message
        _raise_download_error("UCI Adult", error, allow_fallback)
        np.random.seed(random_state)
        n_samples = 5000
        df = pd.DataFrame({
            "age": np.random.randint(18, 70, n_samples),
            "workclass": np.random.choice(["Private", "Self-emp", "Gov"], n_samples),
            "education-num": np.random.randint(5, 16, n_samples),
            "capital-gain": np.random.exponential(1000, n_samples),
            "hours-per-week": np.random.randint(20, 60, n_samples),
            "class": np.random.choice(["<=50K", ">50K"], n_samples, p=[0.75, 0.25]),
        })

    df = df.dropna().reset_index(drop=True)
    target_col = "class" if "class" in df.columns else df.columns[-1]

    X = df.drop(columns=[target_col])
    y = df[target_col].astype(str).str.contains(">50K").astype(int).values

    X_encoded = pd.get_dummies(X, drop_first=True)
    feature_names = list(X_encoded.columns)

    splits = _split_dataset(
        X_encoded.values, y, test_size, val_size, random_state)
    splits, scaler = _scale_splits(splits)

    if return_tensor:
        splits = _to_tensors(splits)

    X_train, y_train, X_val, y_val, X_test, y_test = splits
    print(
        f"-> UCI Adult ready: train shape: {tuple(X_train.shape)}, "
        f"Test Shape: {tuple(X_test.shape)}, feature count: {len(feature_names)}"
        + (f", Val Shape: {tuple(X_val.shape)}" if X_val is not None else "")
    )

    if val_size and val_size > 0:
        return (X_train, y_train, X_val, y_val, X_test, y_test), feature_names, scaler
    return (X_train, y_train, X_test, y_test), feature_names, scaler


# ---------------------------------------------------------------------------
# German Credit
# ---------------------------------------------------------------------------
def load_german_credit_frame(allow_fallback=False, random_state=42):
    """Load German Credit **without splitting** and return the encoded full matrix.

    Cross-validation (``utils.cross_validation``) uses this function because it
    works on the whole dataset instead of a single split. It shares the same
    preprocessing as the train/test split path, so the two paths cannot leak or
    produce mismatched columns.

    Returns:
        ``(X, y, feature_names)`` — ``X`` is a float32 matrix, ``y`` a 0/1
        vector (good=1, bad=0), ``feature_names`` the one-hot column names.
    """
    print("[Data] Loading German Credit Risk...")
    try:
        data = fetch_openml("credit-g", version=1, as_frame=True, parser="auto")
        df = data.frame
    except Exception as error:  # noqa: BLE001 - keep the error message
        _raise_download_error("German Credit", error, allow_fallback)
        np.random.seed(random_state)
        n_samples = 1000
        df = pd.DataFrame({
            "duration": np.random.randint(6, 60, n_samples),
            "credit_amount": np.random.exponential(3000, n_samples),
            "age": np.random.randint(19, 75, n_samples),
            "housing": np.random.choice(["own", "rent", "for free"], n_samples),
            "job": np.random.choice(["unskilled", "skilled", "management"], n_samples),
            "class": np.random.choice(["good", "bad"], n_samples, p=[0.7, 0.3]),
        })

    df = df.dropna().reset_index(drop=True)
    target_col = "class" if "class" in df.columns else df.columns[-1]

    X = df.drop(columns=[target_col])
    # good -> 1, bad -> 0
    y = (df[target_col].astype(str).str.lower() == "good").astype(int).values

    X_encoded = pd.get_dummies(X, drop_first=True)
    feature_names = list(X_encoded.columns)
    return X_encoded.values.astype(np.float32), y, feature_names


def load_german_credit_dataset(
    test_size=0.2,
    random_state=42,
    return_tensor=True,
    val_size=0.0,
    allow_fallback=False,
):
    """Load the German Credit Risk dataset (good=1, bad=0)."""
    X, y, feature_names = load_german_credit_frame(
        allow_fallback=allow_fallback, random_state=random_state
    )

    splits = _split_dataset(
        X, y, test_size, val_size, random_state)
    splits, scaler = _scale_splits(splits)

    if return_tensor:
        splits = _to_tensors(splits)

    X_train, y_train, X_val, y_val, X_test, y_test = splits
    print(
        f"-> German Credit ready: train shape: {tuple(X_train.shape)}, "
        f"Test Shape: {tuple(X_test.shape)}, feature count: {len(feature_names)}"
        + (f", Val Shape: {tuple(X_val.shape)}" if X_val is not None else "")
    )

    if val_size and val_size > 0:
        return (X_train, y_train, X_val, y_val, X_test, y_test), feature_names, scaler
    return (X_train, y_train, X_test, y_test), feature_names, scaler


# ---------------------------------------------------------------------------
# COMPAS
# ---------------------------------------------------------------------------
#: One-hot columns of COMPAS that encode a legally protected attribute.
#: ``drop_protected=True`` removes them, which defines the "fairness-restricted"
#: variant used in the protected-attribute ablation.
COMPAS_PROTECTED_PREFIXES = ("sex_", "race_")


def compas_feature_matrix(drop_protected: bool = False):
    """Return the one-hot encoded COMPAS frame without splitting or scaling.

    Args:
        drop_protected: When ``True``, remove the columns whose name starts with
            one of :data:`COMPAS_PROTECTED_PREFIXES`, so the model never sees
            sex or race.

    Returns:
        ``(X, y, feature_names)`` as ``float32`` / ``int64`` / ``list[str]``.
    """
    csv_path = Path(__file__).resolve().parent / "compas-scores-two-years.csv"
    if not csv_path.exists():
        raise RuntimeError(
            f"COMPAS data file not found: {csv_path}. "
            "Make sure utils/compas-scores-two-years.csv exists in the repository."
        )

    cols = [
        "sex", "age", "age_cat", "race", "juv_fel_count", "juv_misd_count",
        "priors_count", "c_charge_degree", "two_year_recid",
    ]
    frame = pd.read_csv(csv_path)[cols].dropna()
    encoded = pd.get_dummies(frame, drop_first=True)

    if drop_protected:
        protected = [
            c for c in encoded.columns
            if c.startswith(COMPAS_PROTECTED_PREFIXES) and c != "two_year_recid"
        ]
        encoded = encoded.drop(columns=protected)

    X = encoded.drop("two_year_recid", axis=1).values.astype(np.float32)
    y = encoded["two_year_recid"].values.astype(np.int64)
    names = encoded.drop("two_year_recid", axis=1).columns.tolist()
    return X, y, names


def load_compas_dataset(
    show_info=False,
    return_tensor=True,
    test_size=0.2,
    val_size=0.0,
    random_state=42,
    drop_protected=False,
):
    """Load the COMPAS two-year recidivism dataset.

    Args:
        drop_protected: Remove the one-hot columns that encode sex and race.
            Used by the protected-attribute ablation; the default reproduces the
            main benchmark exactly.
    """
    X, y, FEATURE_NAMES = compas_feature_matrix(drop_protected=drop_protected)

    splits = _split_dataset(X, y, test_size, val_size, random_state)

    splits, scaler = _scale_splits(splits)

    if return_tensor:
        splits = _to_tensors(splits)

    X_train, y_train, X_val, y_val, X_test, y_test = splits

    if show_info:
        print(f"Train Shape: {tuple(X_train.shape)}")
        print(f"Test Shape: {tuple(X_test.shape)}")
        if X_val is not None:
            print(f"Val Shape: {tuple(X_val.shape)}")
        print(f"Number of Feature: {X_train.shape[1]}")
        print(f"One Sample Label: {y_train[0]} (0: No Recidivism, 1: Recidivism)")

    if val_size and val_size > 0:
        return (X_train, y_train, X_val, y_val, X_test, y_test), FEATURE_NAMES, scaler
    return (X_train, y_train, X_test, y_test), FEATURE_NAMES, scaler
