"""Core SCORE-Net rule primitives and integer scorecards.

This module is the single source of truth for:

* the triangular activation ``a_h(z)`` and its baseline value ``a_h(0)``;
* exact bias transfer when a rule is pruned (Proposition 1);
* per-rule integer scale search (``per_rule_scales``);
* conversion of a continuous rule into an integer scorecard
  (``scorecard_rule``);
* the quantization budget per dataset/variant (``QuantizationBudget``).

Why one module?
---------------
Duplicating these formulas creates two versions that can silently diverge.
The model forward pass, analysis, pruning, plots, and paper figures all import
from here. ``scripts/verify_artifacts.py`` checks for duplicate formulas.

Quantization scope
------------------
Only the input projection ``W_f`` is rounded to integers. The rule center
``c_h`` and width ``w_h`` are scaled by the same ``λ_h`` but are not rounded,
so the firing interval remains identical to the float model. The output layer
``W_o`` and bias ``b_o`` are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------
EPS = 1e-8
"""Small constant used to avoid division by zero."""

WEIGHT_THRESHOLD = 1e-4
"""Magnitude below which a weight is treated as zero. Defined once for the repo."""


# ---------------------------------------------------------------------------
# Quantization budget
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class QuantizationBudget:
    """Per-variant settings for rule-wise integer quantization.

    Args:
        max_scale: Largest local multiplier to search, ``Λ_max``.
        tolerance: Acceptable relative L1 rounding error, ``τ_tol``.

    Note:
        These values control the integer score range (cognitive load) and
        fidelity to the float model. ``Λ_max`` is only a search ceiling; the
        selected multiplier always satisfies ``λ_h ≤ Λ_max``. Keep the ceiling
        high enough not to restrict any rule, but not so high that the search
        produces coefficients too large for manual scoring.
    """

    max_scale: int
    tolerance: float


# ---------------------------------------------------------------------------
# Rule primitive: triangular activation and bias shifting
# ---------------------------------------------------------------------------
def triangular_activation(z, center, width, eps: float = EPS):
    """Triangular activation ``a_h(z) = [|w_h| - |c_h - z|]^+ / (|w_h| + eps)``.

    Support (Definition 1): the activation is strictly positive on
    ``c_h - w_h < z < c_h + w_h`` and exactly zero at the boundary and outside.
    The closed interval ``c_h - w_h <= z <= c_h + w_h`` is the closure of the
    strictly positive region and is called the "closed support" in the paper.
    Boundary points belong to the slab but have zero activation.

    Args:
        z: Projection score; shape ``(B, H)``, ``(H,)``, or scalar.
        center: Rule centers ``c_h``; ``(H,)``.
        width: Rule radii ``w_h``; ``(H,)``. The sign is ignored.
        eps: Numerical stability term.

    Returns:
        Activation in ``[0, 1]`` with the broadcast shape of ``z``.
    """
    width = torch.abs(width)
    return torch.relu(width - torch.abs(center - z)) / (width + eps)


def base_activation(center, width, eps: float = EPS):
    """Activation at ``z = 0``, ``a_h(0)``.

    This is the constant value a fully zeroed rule contributes for every
    input. Pruning transfers this term to the output bias.
    """
    return triangular_activation(torch.zeros_like(center), center, width, eps)


def bias_shift(W_o, center, width, keep=None, eps: float = EPS):
    """Exact contribution of pruned rules to the output bias (Proposition 1).

    ``Δb_o = W_o[:, S] · a_S(0)``, where ``S`` is the set of removed rules.
    Adding this term leaves model outputs unchanged because the removed rule's
    contribution no longer depends on the input.

    Args:
        W_o: Output weights ``(C, K)`` for the removed rule columns only.
        center: Centers of removed rules ``(K,)``.
        width: Widths of removed rules ``(K,)``.
        keep: Optional ``(K,)`` boolean mask. Only ``True`` rules contribute to
            the bias (e.g., empirically dead rules are dropped without bias).

    Returns:
        Bias contribution of shape ``(C,)``.
    """
    activation = base_activation(center, width, eps)
    if keep is not None:
        activation = activation * torch.as_tensor(
            keep, dtype=activation.dtype, device=activation.device
        )
    return W_o @ activation


# ---------------------------------------------------------------------------
# Rule-wise integer quantization
# ---------------------------------------------------------------------------
def per_rule_scales(W_f, max_scale, tolerance, weight_threshold: float = WEIGHT_THRESHOLD):
    """Choose the smallest acceptable integer multiplier for each rule.

    This implements the discrete search in Equation ``eq:adaptive_scale``.
    All integer multipliers in ``[1, max_scale]`` are scanned in a vectorized
    way, and the smallest multiplier whose relative L1 rounding error is at
    most ``tolerance`` is selected. If no multiplier meets the tolerance, the
    one with the smallest error is used.

    Args:
        W_f: Input projection ``(H, N)`` (NumPy array or tensor).
        max_scale: ``Λ_max``, the largest multiplier to scan.
        tolerance: ``τ_tol``, acceptable relative rounding error.
        weight_threshold: Magnitude below which a weight is treated as zero.

    Returns:
        ``(scales, errors)``: ``(H,)`` float32 scales and the relative
        rounding error at the selected scale for each rule.
    """
    W = np.asarray(W_f, dtype=np.float32)
    H = W.shape[0]
    scales = np.ones(H, dtype=np.float32)
    errors: list[float] = []

    # Integer candidates 1..max_scale as a column: (M, 1)
    candidates = np.arange(1, max_scale + 1, dtype=np.float32)[:, np.newaxis]

    for h in range(H):
        rule = W[h, :]
        active = np.abs(rule) > weight_threshold
        if not np.any(active):
            errors.append(0.0)
            continue

        W_act = rule[active][np.newaxis, :]                          # (1, K)
        norm = np.sum(np.abs(W_act)) + EPS
        reconstructed = np.round(candidates * W_act) / candidates    # (M, K)
        rel_errors = np.sum(np.abs(W_act - reconstructed), axis=1) / norm

        acceptable = np.where(rel_errors <= tolerance)[0]
        best = acceptable[0] if len(acceptable) else int(np.argmin(rel_errors))
        scales[h] = float(candidates[best, 0])
        errors.append(float(rel_errors[best]))

    return scales, errors


def integer_weights(W_f, scales, weight_threshold: float = WEIGHT_THRESHOLD):
    """Return the integer matrix ``round(λ_h · W_f[h, j])``.

    Entries below the threshold (pruned weights) are set exactly to zero so
    that rounding noise does not enter the integer scorecard.

    Args:
        W_f: Continuous projection ``(H, N)``.
        scales: Per-rule scales ``(H,)``.

    Returns:
        ``(H, N)`` int64 weight matrix.
    """
    W = np.asarray(W_f, dtype=np.float32)
    lam = np.asarray(scales, dtype=np.float32).reshape(-1, 1)
    ints = np.round(W * lam)
    ints[np.abs(W) <= weight_threshold] = 0.0
    return ints.astype(np.int64)


def quantize_projection(W_f, scales):
    """Quantized projection used by the model: ``round(λ_h W_f) / λ_h``.

    Args:
        W_f: Torch tensor ``(H, N)``.
        scales: Per-rule scales ``(H,)``.

    Returns:
        Quantized projection with the same device and dtype as ``W_f``.
    """
    lam = torch.as_tensor(scales, dtype=W_f.dtype, device=W_f.device).reshape(-1, 1)
    return torch.round(W_f * lam) / lam


def quantize_projection_uniform(W_f, scale):
    """Quantize with one global multiplier: ``round(scale · W_f) / scale``."""
    return torch.round(W_f * scale) / scale


# ---------------------------------------------------------------------------
# Integer scorecard
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ScorecardRule:
    """Continuous and integer scorecard representation of one rule.

    Attributes:
        index: Rule index in the model (0-based).
        scale: Selected multiplier ``λ_h``.
        rounding_error: Relative L1 rounding error at this scale.
        continuous: Active feature -> continuous weight.
        integer: Active feature -> integer weight.
        center: Continuous center ``c_h``.
        width: Continuous radius ``w_h``.
        center_scaled: ``λ_h c_h`` (not rounded).
        width_scaled: ``λ_h w_h`` (not rounded).
        lower: Lower firing bound ``λ_h(c_h - w_h)``.
        upper: Upper firing bound ``λ_h(c_h + w_h)``.
        firing_integers: Integer scores inside the firing interval.
    """

    index: int
    scale: float
    rounding_error: float
    continuous: Mapping[str, float]
    integer: Mapping[str, int]
    center: float
    width: float
    center_scaled: float
    width_scaled: float
    lower: float
    upper: float
    firing_integers: tuple[int, ...]

    @property
    def is_univariate(self) -> bool:
        """Whether the rule depends on a single feature."""
        return len(self.integer) == 1

    @property
    def feature(self) -> str:
        """Name of the active feature with the largest absolute weight."""
        return max(self.continuous, key=lambda name: abs(self.continuous[name]))

    @property
    def integer_weight(self) -> int:
        """Integer coefficient of the feature with the largest absolute weight."""
        return self.integer[self.feature]


def scorecard_rule(
    W_f_row,
    center: float,
    width: float,
    scale: float,
    *,
    index: int = 0,
    rounding_error: float = 0.0,
    feature_names: Sequence[str] | None = None,
    weight_threshold: float = WEIGHT_THRESHOLD,
) -> ScorecardRule:
    """Convert one continuous rule into an integer scorecard entry.

    This is the single link between Figure 1(c) and Table
    ``tab:scorecard_fidelity``; both read the same numbers from here.

    Args:
        W_f_row: Continuous projection row ``(N,)``.
        center: Rule center ``c_h``.
        width: Rule radius ``w_h``.
        scale: Selected multiplier ``λ_h``.
        index: Rule index (0-based).
        rounding_error: Error returned by ``per_rule_scales``.
        feature_names: Feature names; indices are used if omitted.
        weight_threshold: Threshold for active features.

    Returns:
        A populated :class:`ScorecardRule`.
    """
    W = np.asarray(W_f_row, dtype=np.float32).reshape(-1)
    active = np.flatnonzero(np.abs(W) > weight_threshold)
    names = list(feature_names) if feature_names is not None else list(range(W.size))

    continuous = {str(names[j]): float(W[j]) for j in active}
    integer = {str(names[j]): int(round(scale * float(W[j]))) for j in active}

    center_scaled = scale * float(center)
    width_scaled = scale * abs(float(width))
    lower = center_scaled - width_scaled
    upper = center_scaled + width_scaled

    # Integer scores inside the interval; the 1e-9 tolerance keeps a score
    # exactly on the boundary from being excluded by floating-point error.
    first = int(np.ceil(lower - 1e-9))
    last = int(np.floor(upper + 1e-9))

    return ScorecardRule(
        index=index,
        scale=float(scale),
        rounding_error=float(rounding_error),
        continuous=continuous,
        integer=integer,
        center=float(center),
        width=float(width),
        center_scaled=center_scaled,
        width_scaled=width_scaled,
        lower=lower,
        upper=upper,
        firing_integers=tuple(range(first, last + 1)),
    )


def scorecard_rules(
    W_f,
    centers,
    widths,
    scales,
    errors=None,
    feature_names: Sequence[str] | None = None,
    weight_threshold: float = WEIGHT_THRESHOLD,
) -> list[ScorecardRule]:
    """Apply ``scorecard_rule`` to every rule."""
    W = np.asarray(W_f)
    centers = np.asarray(centers).reshape(-1)
    widths = np.asarray(widths).reshape(-1)
    scales = np.asarray(scales).reshape(-1)
    errors = [0.0] * W.shape[0] if errors is None else list(errors)
    return [
        scorecard_rule(
            W[h], centers[h], widths[h], scales[h],
            index=h, rounding_error=errors[h], feature_names=feature_names,
            weight_threshold=weight_threshold,
        )
        for h in range(W.shape[0])
    ]


# ---------------------------------------------------------------------------
# Single-feature response curve: exact breakpoints
# ---------------------------------------------------------------------------
# Notation (Equation ``eq:breakpoints``): the superscript denotes the slab
# edge in rule space ``z``, not its position on the ``x`` axis.
#
#     v_h^{-} = (c_h - w_h - b) / W_f,hj      (left edge, z = c_h - w_h)
#     v_h^{0} = (c_h      - b) / W_f,hj      (peak)
#     v_h^{+} = (c_h + w_h - b) / W_f,hj      (right edge, z = c_h + w_h)
#
# If W_f,hj < 0 the division flips the inequality, so v_h^{-} > v_h^{+}
# is possible. This is not an error; it is why the caller sorts the values.
def breakpoints(
    W_f_column,
    center,
    width,
    offset,
    weight_threshold: float = WEIGHT_THRESHOLD,
):
    """Exact breakpoints of a rule along one feature.

    Args:
        W_f_column: Projection column of the feature, ``(H,)``.
        center: Rule centers ``c_h``, ``(H,)``.
        width: Rule radii ``w_h``, ``(H,)``.
        offset: Constant term from the other features,
            ``b_h^{(j)} = sum_{k != j} W_f,hk x_k``, ``(H,)``.
        weight_threshold: Weights below this threshold are treated as zero.

    Returns:
        Unsorted tensor of shape ``(3K,)`` with ``v_h^{-}``, ``v_h^{0}``, and
        ``v_h^{+}`` for every active rule. The caller applies ``torch.sort``.
    """
    column = torch.as_tensor(W_f_column)
    center = torch.as_tensor(center, dtype=column.dtype, device=column.device)
    width = torch.abs(torch.as_tensor(width, dtype=column.dtype, device=column.device))
    offset = torch.as_tensor(offset, dtype=column.dtype, device=column.device)

    active = column.abs() > weight_threshold
    if not bool(active.any()):
        return column.new_zeros(0)

    step = column[active]
    c, w, b = center[active], width[active], offset[active]
    return torch.cat([(c - w - b) / step, (c - b) / step, (c + w - b) / step])


def response_breakpoints(
    W_f_column,
    center,
    width,
    offset,
    lower: float,
    upper: float,
    extra=(),
    decimals: int = 10,
    include_bounds: bool = True,
    weight_threshold: float = WEIGHT_THRESHOLD,
):
    """Evaluation points needed to plot a one-feature response curve.

    Returned points lie in ``[lower, upper]`` and are sorted. The model's logit
    response is exactly affine between consecutive points, so evaluating only
    at these points gives zero interpolation error; a dense grid is
    unnecessary.

    Args:
        W_f_column, center, width, offset: Same as :func:`breakpoints`.
        lower, upper: Valid feature range ``[f_min, f_max]``.
        extra: Additional values to include (e.g., the applicant's own value,
            which is marked on the curve).
        decimals: Rounding precision used to merge floating-point duplicates.
        include_bounds: If ``False``, ``lower``/``upper`` are not added; only
            the exact breakpoints are returned. Recourse search uses this
            option because the example value is already supplied via ``extra``.
        weight_threshold: Same as :func:`breakpoints`.

    Returns:
        Sorted, unique 1-D tensor.
    """
    points = breakpoints(W_f_column, center, width, offset, weight_threshold)
    keep = [points]
    if include_bounds:
        keep.append(torch.as_tensor([float(lower), float(upper)], dtype=points.dtype))
    if len(extra):
        keep.append(torch.as_tensor([float(v) for v in extra], dtype=points.dtype))

    merged = torch.cat(keep)
    merged = merged[(merged >= float(lower)) & (merged <= float(upper))]
    merged = torch.round(merged * (10 ** decimals)) / (10 ** decimals)
    return torch.unique(merged).sort().values
