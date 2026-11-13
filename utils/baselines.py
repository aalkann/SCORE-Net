"""Single source of truth for the baseline models.

Every hyperparameter below was selected by the authors through manual,
dataset-by-dataset search on the validation split, and the configuration kept is
the one with the best validation performance. There is no automated grid search;
the values are frozen here so that the single-split benchmark, the cross
validation and the notebooks all instantiate *exactly* the same estimator.

The registry is intentionally explicit rather than clever: a reader can compare
this file with Table 3 of the manuscript and check every number.
"""

from __future__ import annotations

from typing import Callable

FAMILIES = ("Interpretable", "SOTA Ensemble", "Geometric & DL")


def _sklearn_models() -> dict:
    """Import scikit-learn estimators lazily (keeps import time low)."""
    from sklearn.ensemble import (
        AdaBoostClassifier,
        GradientBoostingClassifier,
        RandomForestClassifier,
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier

    return {
        "LogisticRegression": LogisticRegression,
        "DecisionTreeClassifier": DecisionTreeClassifier,
        "GaussianNB": GaussianNB,
        "RandomForestClassifier": RandomForestClassifier,
        "GradientBoostingClassifier": GradientBoostingClassifier,
        "AdaBoostClassifier": AdaBoostClassifier,
        "SVC": SVC,
        "KNeighborsClassifier": KNeighborsClassifier,
        "MLPClassifier": MLPClassifier,
    }


# ---------------------------------------------------------------------------
# Per-dataset configurations
# ---------------------------------------------------------------------------
# Each entry is ``(display_name, family, builder)``. ``builder`` returns a fresh,
# unfitted estimator; it must not share state between folds.
# ---------------------------------------------------------------------------
def baseline_specs(key: str) -> list[tuple[str, str, Callable[[], object]]]:
    """Return the 13 baselines used for ``key`` in the manuscript."""
    sk = _sklearn_models()

    import lightgbm as lgb
    import xgboost as xgb
    from catboost import CatBoostClassifier
    from interpret.glassbox import ExplainableBoostingClassifier

    # `compas_noprot` is the same task with the protected columns removed, so it
    # reuses the COMPAS baseline configuration unchanged.
    if key in ("compas", "compas_noprot"):
        return [
            ("Logistic Regression", "Interpretable",
             lambda: sk["LogisticRegression"](penalty="l1", C=0.01, max_iter=1000, solver="saga", random_state=42)),
            ("Decision Tree", "Interpretable",
             lambda: sk["DecisionTreeClassifier"](max_depth=5, min_samples_split=3, criterion="entropy", random_state=42)),
            ("EBM (InterpretML)", "Interpretable",
             lambda: ExplainableBoostingClassifier(max_bins=32, learning_rate=0.01, max_leaves=4, random_state=42)),
            ("Naive Bayes", "Interpretable", lambda: sk["GaussianNB"]()),
            ("XGBoost", "SOTA Ensemble",
             lambda: xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, eval_metric="logloss")),
            ("LightGBM", "SOTA Ensemble",
             lambda: lgb.LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbose=-1)),
            ("CatBoost", "SOTA Ensemble",
             lambda: CatBoostClassifier(iterations=100, depth=6, l2_leaf_reg=5, learning_rate=0.01, random_state=42, verbose=0)),
            ("Random Forest", "SOTA Ensemble",
             lambda: sk["RandomForestClassifier"](n_estimators=200, max_depth=10, min_samples_leaf=2, min_samples_split=5, max_features="sqrt", n_jobs=-1, random_state=42)),
            ("Gradient Boosting", "SOTA Ensemble",
             lambda: sk["GradientBoostingClassifier"](n_estimators=10, learning_rate=0.05, max_depth=5, subsample=0.8, random_state=42)),
            ("AdaBoost", "SOTA Ensemble",
             lambda: sk["AdaBoostClassifier"](estimator=sk["DecisionTreeClassifier"](max_depth=3), n_estimators=50, learning_rate=0.7, random_state=42)),
            ("SVM (RBF)", "Geometric & DL",
             lambda: sk["SVC"](kernel="rbf", C=1.0, gamma="scale", random_state=42)),
            ("KNN", "Geometric & DL",
             lambda: sk["KNeighborsClassifier"](n_neighbors=55, weights="uniform", metric="euclidean", n_jobs=-1)),
            ("MLP (sklearn)", "Geometric & DL",
             lambda: sk["MLPClassifier"](hidden_layer_sizes=(32, 32, 16), activation="relu", max_iter=1000, early_stopping=True, random_state=42)),
        ]

    if key == "german":
        return [
            ("Logistic Regression", "Interpretable",
             lambda: sk["LogisticRegression"](penalty="l1", C=0.1, max_iter=1000, solver="saga", random_state=42)),
            ("Decision Tree", "Interpretable",
             lambda: sk["DecisionTreeClassifier"](max_depth=3, min_samples_split=2, min_samples_leaf=1, criterion="entropy", random_state=42)),
            ("EBM (InterpretML)", "Interpretable",
             lambda: ExplainableBoostingClassifier(max_bins=64, learning_rate=0.1, max_leaves=4, random_state=42)),
            ("Naive Bayes", "Interpretable", lambda: sk["GaussianNB"](var_smoothing=1e-5)),
            ("XGBoost", "SOTA Ensemble",
             lambda: xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42, eval_metric="logloss")),
            ("LightGBM", "SOTA Ensemble",
             lambda: lgb.LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbose=-1)),
            ("CatBoost", "SOTA Ensemble",
             lambda: CatBoostClassifier(iterations=200, depth=8, learning_rate=0.03, l2_leaf_reg=1, random_state=42, verbose=0)),
            ("Random Forest", "SOTA Ensemble",
             lambda: sk["RandomForestClassifier"](n_estimators=200, max_depth=10, min_samples_leaf=2, min_samples_split=5, max_features="sqrt", n_jobs=-1, random_state=42)),
            ("Gradient Boosting", "SOTA Ensemble",
             lambda: sk["GradientBoostingClassifier"](n_estimators=100, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=42)),
            ("AdaBoost", "SOTA Ensemble",
             lambda: sk["AdaBoostClassifier"](estimator=sk["DecisionTreeClassifier"](max_depth=3), n_estimators=300, learning_rate=0.1, random_state=42)),
            ("SVM (RBF)", "Geometric & DL",
             lambda: sk["SVC"](kernel="rbf", C=1, random_state=42)),
            ("KNN", "Geometric & DL",
             lambda: sk["KNeighborsClassifier"](n_neighbors=33, weights="uniform", metric="euclidean", n_jobs=-1)),
            ("MLP (sklearn)", "Geometric & DL",
             lambda: sk["MLPClassifier"](hidden_layer_sizes=(32, 16, 8), activation="relu", max_iter=500, early_stopping=True, random_state=42)),
        ]

    if key == "adult":
        return [
            ("Logistic Regression", "Interpretable",
             lambda: sk["LogisticRegression"](penalty="l1", C=1.0, max_iter=1000, solver="saga", random_state=42)),
            ("Decision Tree", "Interpretable",
             lambda: sk["DecisionTreeClassifier"](max_depth=5, min_samples_split=3, criterion="entropy", random_state=42)),
            ("EBM (InterpretML)", "Interpretable",
             lambda: ExplainableBoostingClassifier(max_bins=64, interactions=0, random_state=42)),
            ("Naive Bayes", "Interpretable", lambda: sk["GaussianNB"]()),
            ("XGBoost", "SOTA Ensemble",
             lambda: xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.2, random_state=42, eval_metric="logloss")),
            ("LightGBM", "SOTA Ensemble",
             lambda: lgb.LGBMClassifier(n_estimators=400, max_depth=3, learning_rate=0.1, random_state=42, verbose=-1)),
            ("CatBoost", "SOTA Ensemble",
             lambda: CatBoostClassifier(iterations=300, depth=5, learning_rate=0.1, random_state=42, verbose=0)),
            ("Random Forest", "SOTA Ensemble",
             lambda: sk["RandomForestClassifier"](n_estimators=500, max_depth=20, min_samples_leaf=2, max_features="sqrt", n_jobs=-1, random_state=42)),
            ("Gradient Boosting", "SOTA Ensemble",
             lambda: sk["GradientBoostingClassifier"](n_estimators=200, learning_rate=0.1, max_depth=5, subsample=1, random_state=42)),
            ("AdaBoost", "SOTA Ensemble",
             lambda: sk["AdaBoostClassifier"](estimator=sk["DecisionTreeClassifier"](max_depth=3), n_estimators=300, learning_rate=0.5, random_state=42)),
            ("SVM (RBF)", "Geometric & DL",
             lambda: sk["SVC"](kernel="rbf", C=10.0, probability=False, random_state=42)),
            ("KNN", "Geometric & DL",
             lambda: sk["KNeighborsClassifier"](n_neighbors=15, weights="uniform", n_jobs=-1)),
            ("MLP (sklearn)", "Geometric & DL",
             lambda: sk["MLPClassifier"](hidden_layer_sizes=(64, 128, 64), activation="relu", max_iter=500, early_stopping=True, random_state=42)),
        ]

    if key == "synthetic":
        return [
            ("Logistic Regression", "Interpretable",
             lambda: sk["LogisticRegression"](penalty="l1", C=0.5, max_iter=1000, solver="saga", random_state=42)),
            ("Decision Tree", "Interpretable",
             lambda: sk["DecisionTreeClassifier"](max_depth=10, min_samples_split=2, criterion="entropy", random_state=42)),
            ("EBM (InterpretML)", "Interpretable",
             lambda: ExplainableBoostingClassifier(max_bins=64, interactions=2, random_state=42)),
            ("Naive Bayes", "Interpretable", lambda: sk["GaussianNB"]()),
            ("XGBoost", "SOTA Ensemble",
             lambda: xgb.XGBClassifier(n_estimators=500, max_depth=7, learning_rate=0.1, random_state=42, eval_metric="logloss")),
            ("LightGBM", "SOTA Ensemble",
             lambda: lgb.LGBMClassifier(n_estimators=700, max_depth=7, learning_rate=0.1, random_state=42, verbose=-1)),
            ("CatBoost", "SOTA Ensemble",
             lambda: CatBoostClassifier(iterations=700, depth=7, learning_rate=0.1, random_state=42, verbose=0)),
            ("Random Forest", "SOTA Ensemble",
             lambda: sk["RandomForestClassifier"](n_estimators=500, max_depth=20, min_samples_leaf=2, max_features="sqrt", n_jobs=-1, random_state=42)),
            ("Gradient Boosting", "SOTA Ensemble",
             lambda: sk["GradientBoostingClassifier"](n_estimators=300, learning_rate=0.1, max_depth=5, subsample=0.8, random_state=42)),
            ("AdaBoost", "SOTA Ensemble",
             lambda: sk["AdaBoostClassifier"](estimator=sk["DecisionTreeClassifier"](max_depth=7), n_estimators=300, learning_rate=0.5, random_state=42)),
            ("SVM (RBF)", "Geometric & DL",
             lambda: sk["SVC"](kernel="rbf", C=10.0, probability=True, random_state=42)),
            ("KNN", "Geometric & DL",
             lambda: sk["KNeighborsClassifier"](n_neighbors=51, weights="distance", n_jobs=-1)),
            ("MLP (sklearn)", "Geometric & DL",
             lambda: sk["MLPClassifier"](hidden_layer_sizes=(1024, 512, 1024), activation="relu", max_iter=500, early_stopping=True, random_state=42)),
        ]

    raise KeyError(f"Unknown dataset: {key!r}")
