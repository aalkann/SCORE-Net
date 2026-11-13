"""Statistical helpers for reliable comparisons on small test sets.

On small benchmarks such as COMPAS (1,443 test samples) and especially German
Credit (100 test samples), a single accuracy percentage hides sampling noise:
on German Credit **one sample equals 1.0 percentage point**. This module
provides three tools to measure the uncertainty of the reported numbers:

1. :func:`bootstrap_ci` — stratified bootstrap confidence interval for a
   metric (accuracy, macro-F1, ROC-AUC).
2. :func:`mcnemar_test` — paired test (McNemar) comparing the predictions of
   two models on the **same** test samples.
3. :func:`paired_bootstrap_diff` — paired bootstrap distribution of the
   accuracy difference between two models, with a two-sided p-value.

All functions are deterministic (``seed`` parameter) and never modify the test
set; they only resample the existing predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    labels = np.unique(y_true)
    scores = []
    for label in labels:
        tp = float(((y_pred == label) & (y_true == label)).sum())
        fp = float(((y_pred == label) & (y_true != label)).sum())
        fn = float(((y_pred != label) & (y_true == label)).sum())
        denom = 2.0 * tp + fp + fn
        scores.append(0.0 if denom == 0 else 2.0 * tp / denom)
    return float(np.mean(scores))


METRICS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "accuracy": _accuracy,
    "macro_f1": _macro_f1,
}


@dataclass(frozen=True)
class BootstrapResult:
    """Point estimate and bootstrap confidence interval for a metric."""

    metric: str
    n_samples: int
    point: float
    ci_low: float
    ci_high: float
    n_boot: int
    alpha: float

    @property
    def ci_width(self) -> float:
        return self.ci_high - self.ci_low


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str = "accuracy",
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> BootstrapResult:
    """Compute a confidence interval for a metric with stratified bootstrap.

    Args:
        y_true: Ground-truth labels, ``(M,)``.
        y_pred: Model predictions, ``(M,)``.
        metric: ``"accuracy"`` or ``"macro_f1"``.
        n_boot: Number of bootstrap repetitions.
        alpha: Significance level (0.05 → 95% CI).
        seed: Random seed.

    Returns:
        :class:`BootstrapResult`

    Note:
        Resampling is done per class (stratified). This keeps the bootstrap
        distribution stable on sets with a minority class and preserves class
        ratios.
    """
    if metric not in METRICS:
        raise KeyError(f"Unknown metric: {metric!r}. Options: {sorted(METRICS)}")

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true {y_true.shape} and y_pred {y_pred.shape} must have the same size."
        )

    score = METRICS[metric]
    rng = np.random.default_rng(seed)

    # Per-class indices: each bootstrap sample preserves the class ratio.
    class_indices = [np.where(y_true == label)[0] for label in np.unique(y_true)]

    boot = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        sample = np.concatenate(
            [rng.choice(idx, size=len(idx), replace=True) for idx in class_indices]
        )
        boot[b] = score(y_true[sample], y_pred[sample])

    low, high = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return BootstrapResult(
        metric=metric,
        n_samples=int(y_true.size),
        point=score(y_true, y_pred),
        ci_low=float(low),
        ci_high=float(high),
        n_boot=int(n_boot),
        alpha=float(alpha),
    )


@dataclass(frozen=True)
class McNemarResult:
    """Paired error structure of two models (McNemar test)."""

    n_samples: int
    model_a_errors: int
    model_b_errors: int
    a_wrong_b_right: int
    a_right_b_wrong: int
    statistic: float
    p_value: float
    test: str


def mcnemar_test(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
) -> McNemarResult:
    """Apply the McNemar test between two models.

    The test uses only the samples where the two models **disagree**. For small
    disagreement counts (``n < 25``) it uses the exact binomial test, otherwise
    the chi-square approximation with continuity correction.

    Args:
        y_true: Ground-truth labels.
        pred_a: Predictions of the first model.
        pred_b: Predictions of the second model.

    Returns:
        :class:`McNemarResult`
    """
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    if not (y_true.shape == pred_a.shape == pred_b.shape):
        raise ValueError("y_true, pred_a, and pred_b must have the same size.")

    a_right = pred_a == y_true
    b_right = pred_b == y_true

    b_only = int((~a_right & b_right).sum())  # A wrong, B right
    a_only = int((a_right & ~b_right).sum())  # A right, B wrong

    n_discordant = a_only + b_only
    if n_discordant == 0:
        return McNemarResult(
            n_samples=int(y_true.size),
            model_a_errors=int((~a_right).sum()),
            model_b_errors=int((~b_right).sum()),
            a_wrong_b_right=b_only,
            a_right_b_wrong=a_only,
            statistic=0.0,
            p_value=1.0,
            test="identical-predictions",
        )

    if n_discordant < 25:
        # Exact binomial test (two-sided).
        from scipy.stats import binomtest

        p_value = float(
            binomtest(min(a_only, b_only), n_discordant, 0.5, alternative="two-sided").pvalue
        )
        return McNemarResult(
            n_samples=int(y_true.size),
            model_a_errors=int((~a_right).sum()),
            model_b_errors=int((~b_right).sum()),
            a_wrong_b_right=b_only,
            a_right_b_wrong=a_only,
            statistic=float(min(a_only, b_only)),
            p_value=p_value,
            test="exact-binomial",
        )

    # Chi-square approximation with continuity correction.
    statistic = (abs(a_only - b_only) - 1.0) ** 2 / n_discordant
    from scipy.stats import chi2

    p_value = float(chi2.sf(statistic, df=1))
    return McNemarResult(
        n_samples=int(y_true.size),
        model_a_errors=int((~a_right).sum()),
        model_b_errors=int((~b_right).sum()),
        a_wrong_b_right=b_only,
        a_right_b_wrong=a_only,
        statistic=float(max(statistic, 0.0)),
        p_value=p_value,
        test="chi2-continuity-corrected",
    )


@dataclass(frozen=True)
class PairedDiffResult:
    """Bootstrap distribution of the accuracy difference between two models."""

    diff_point: float
    diff_ci_low: float
    diff_ci_high: float
    p_value: float
    n_boot: int


def paired_bootstrap_diff(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> PairedDiffResult:
    """Paired bootstrap CI and p-value for the accuracy difference of two models.

    The difference is defined as ``accuracy(A) - accuracy(B)``. The two-sided
    p-value is computed without regard to which side the bootstrap distribution
    crosses zero (``2 * min(P(diff<=0), P(diff>=0))``).
    """
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)

    correct_a = (pred_a == y_true).astype(np.float64)
    correct_b = (pred_b == y_true).astype(np.float64)
    diff_point = float(correct_a.mean() - correct_b.mean())

    rng = np.random.default_rng(seed)
    n = y_true.size
    boot = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[b] = correct_a[idx].mean() - correct_b[idx].mean()

    low, high = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p_low = float((boot <= 0).mean())
    p_high = float((boot >= 0).mean())
    p_value = min(1.0, 2.0 * min(p_low, p_high))

    return PairedDiffResult(
        diff_point=diff_point,
        diff_ci_low=float(low),
        diff_ci_high=float(high),
        p_value=p_value,
        n_boot=int(n_boot),
    )


def format_ci(result: BootstrapResult, decimals: int = 2, scale: float = 100.0) -> str:
    """Return readable text in the form ``72.00 [65.50, 78.00]``."""
    return (
        f"{result.point * scale:.{decimals}f} "
        f"[{result.ci_low * scale:.{decimals}f}, {result.ci_high * scale:.{decimals}f}]"
    )


def minimum_detectable_difference(n_samples: int, alpha: float = 0.05) -> float:
    """Roughly smallest detectable accuracy difference (%) for a given test size.

    The McNemar test depends on the number of discordant samples; here the 95%
    CI half-width of two independent proportions is used as a practical upper
    bound. The goal is to show the reader the smallest difference that can be
    reported as "significant" on small test sets.
    """
    # Worst-case variance, for p=0.5.
    se = np.sqrt(0.5 * 0.5 / max(n_samples, 1))
    z = 1.959963984540054
    return float(2 * z * se * 100.0)
