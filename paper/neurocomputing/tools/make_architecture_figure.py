#!/usr/bin/env python3
"""Draw the three SCORE-Net architecture figures (Figures 1--3).

Every panel is written as its **own** file, in both vector PDF (for LaTeX) and
PNG (for quick inspection), at exactly the width at which the manuscript prints
it.  Because the figure is never rescaled by LaTeX, each font keeps its nominal
size and the Elsevier 7 pt / 6 pt rule is satisfied by construction.

Everything is drawn with Matplotlib primitives: there is no embedded raster
image, so every box, colour, label and number can be edited in this file and
re-rendered.  All numbers are read from the released COMPAS Compact checkpoint
through ``utils.experiments``; the rule primitive, the integer scales and the
scorecard entries come from ``utils.rules``, i.e. the same functions that
produce the scorecard tables, so the figure cannot drift from the results.

    fig_arch_forward.pdf/.png    (a) the forward pass and the three learned rules
    fig_arch_unit.pdf/.png       (b) the triangular activation of one hidden unit
    fig_arch_scorecard.pdf/.png  (c) all three rules after integer quantization

Layout control
--------------
Colours and font sizes are the module-level constants ``INK``, ``ACCENT``,
``WARM`` and ``FS_*``.  Panel (a) is driven by the ``stages`` list and the
rule-table offsets inside :func:`figure_forward`; panel (c) by the ``LAYOUT``
block at the top of :func:`figure_scorecard`, where every vertical position is
one named constant.  Panel (b) exposes its axes rectangle and the ``lo``/``hi``
window as plain numbers.  Changing any of them and re-running this file is the
only step needed to update the artwork.

Usage
-----
    python paper/neurocomputing/tools/make_architecture_figure.py
    python paper/neurocomputing/tools/make_architecture_figure.py --panel b
    python paper/neurocomputing/tools/make_architecture_figure.py --no-checkpoint
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

REPO = Path(__file__).resolve().parents[3]
FIG_DIR = REPO / "paper" / "neurocomputing" / "figures"

# ---------------------------------------------------------------------------
# Canvas sizes (inches).  Each equals the width at which LaTeX includes the
# figure, so the print scale is 1.00 and no font shrinks.
# ---------------------------------------------------------------------------
WIDTH_FULL = 7.25          # \textwidth  (522 pt)
WIDTH_UNIT = 3.99          # 0.55\textwidth

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
INK = "#1f2937"
ACCENT = "#1d4ed8"
NEG = "#b91c1c"
MUTED = "#6b7280"
FAINT = "#f3f4f6"
WARM = "#fef3c7"
WARM_EDGE = "#b45309"

FS_TITLE = 9.4
FS_BODY = 7.6
FS_SMALL = 7.0
FS_TINY = 6.3

SUBSCRIPT = {"0": "\u2080", "1": "\u2081", "2": "\u2082",
             "3": "\u2083", "4": "\u2084", "5": "\u2085"}


def sub(i: int) -> str:
    return "".join(SUBSCRIPT[c] for c in str(i))


def num(x: float, digits: int = 2, signed: bool = False) -> str:
    """Format a number with a typographic minus sign (and optional +)."""
    text = f"{x:+.{digits}f}" if signed else f"{x:.{digits}f}"
    return text.replace("-", "\u2212")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_data() -> dict:
    """Read the released COMPAS Compact checkpoint and its integer scales."""
    try:
        sys.path.insert(0, str(REPO))
        from utils.experiments import DATASETS, load_compact_model, load_splits
        from utils.rules import per_rule_scales, scorecard_rules

        spec = DATASETS["compas_noprot"]
        split = load_splits("compas_noprot")
        model = load_compact_model(spec.final_checkpoint, split.loader("train"))
        budget = spec.quantization("Compact")

        Wf = model.feature_linear.weight.detach().cpu().numpy()
        Wo = model.output_linear.weight.detach().cpu().numpy()
        bias = model.output_linear.bias.detach().cpu().numpy()
        center = model.base_feature_center.detach().cpu().numpy()
        width = np.abs(model.base_feature_width.detach().cpu().numpy())
        names = list(split.feature_names)
        scales, errors = per_rule_scales(Wf, budget.max_scale, budget.tolerance)
        meta = dict(budget=budget, real=True)
    except Exception:  # noqa: BLE001 - the figure must render in a fresh clone
        from utils.rules import per_rule_scales, scorecard_rules

        Wf = np.array([[-0.342, 0, 0, 0, 0, 0, 0],
                       [0, 0.518, 0, 0, 0, 0, 0],
                       [0, 0, 0.297, 0, 0, 0, 0]], dtype=float)
        Wo = np.array([[0.594, 0.227, -0.828], [-0.727, -0.275, 0.797]])
        bias = np.array([-0.227, 0.171])
        center = np.array([0.240, 0.101, -0.406])
        width = np.array([0.298, 0.289, 0.175])
        names = ["age", "juv_fel_count", "juv_misd_count", "priors_count",
                 "age_cat_Greater than 45", "age_cat_Less than 25",
                 "c_charge_degree_M"]
        scales, errors = per_rule_scales(Wf, 10, 0.05)
        meta = dict(budget=None, real=False)

    # Per-rule summary used by panels (a) and (c).  The conversion lives in
    # utils.rules so the figure shows exactly what the tables report.
    rules = scorecard_rules(Wf, center, width, scales, errors,
                            feature_names=names)

    h_show = int(np.argmax(np.abs(Wf).max(axis=1)))
    return dict(Wf=Wf, Wo=Wo, bias=bias, names=names, rules=rules,
                show=rules[h_show], **meta)


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------
def arrow(ax, p0, p1, color=INK, lw=1.0, style="-|>", scale=8):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle=style, mutation_scale=scale, linewidth=lw,
        color=color, zorder=6, shrinkA=0, shrinkB=0,
    ))


def stage_box(ax, x, y, w, h, title, subtitle, face="#ffffff"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012",
        facecolor=face, edgecolor=INK, linewidth=0.9, zorder=3,
    ))
    ax.text(x + w / 2, y + h * 0.63, title, ha="center", va="center",
            fontsize=FS_BODY + 1.2, color=INK, zorder=4)
    ax.text(x + w / 2, y + h * 0.26, subtitle, ha="center", va="center",
            fontsize=FS_TINY, color=MUTED, zorder=4)


def save(fig, out_dir: Path, stem: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in (".pdf", ".png"):
        path = out_dir / f"{stem}{suffix}"
        fig.savefig(path, dpi=400)
        paths.append(path)
    plt.close(fig)
    return paths


# ---------------------------------------------------------------------------
# (a) forward pass
# ---------------------------------------------------------------------------
def figure_forward(d: dict, out_dir: Path) -> list[Path]:
    Wf = d["Wf"]
    H, N = Wf.shape
    rules = d["rules"]

    fig = plt.figure(figsize=(WIDTH_FULL, 3.10))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.012, 0.945, "(a) Forward pass of the pruned COMPAS model",
            fontsize=FS_TITLE, fontweight="bold", color=INK, va="center")

    # --- pipeline ----------------------------------------------------------
    y0, h = 0.605, 0.185
    stages = [
        ("input  $x$", f"{N} features", 0.135),
        (r"$z = W_f\,x$", f"{H} rules, no bias", 0.175),
        (r"$a_h$", "interval activation", 0.165),
        (r"$s = W_o\,a + b_o$", "2 class scores", 0.180),
        (r"$\hat{y}$", "argmax", 0.125),
    ]
    gap = (0.976 - sum(w for _, _, w in stages)) / (len(stages) - 1)
    x = 0.012
    shapes = [f"N = {N}", f"H = {H}", f"H = {H}", "C = 2"]
    for k, (title, subtitle, w) in enumerate(stages):
        stage_box(ax, x, y0, w, h, title, subtitle,
                  face=WARM if k == 2 else "#ffffff")
        if k < len(stages) - 1:
            ax.text(x + w + gap / 2, y0 + h + 0.030, shapes[k], ha="center",
                    va="bottom", fontsize=FS_TINY, color=MUTED)
            arrow(ax, (x + w + 0.004, y0 + h / 2),
                  (x + w + gap - 0.004, y0 + h / 2))
        x += w + gap

    # --- the three rules that survive pruning ------------------------------
    ax.text(0.012, 0.515, "The three rules that survive pruning",
            fontsize=FS_BODY + 0.4, fontweight="bold", color=INK, va="center")

    row_y = [0.415, 0.325, 0.235]
    for k, rule in enumerate(rules):
        y = row_y[k]
        if k % 2 == 0:
            ax.add_patch(Rectangle((0.010, y - 0.036), 0.978, 0.072,
                                   facecolor=FAINT, edgecolor="none", zorder=1))
        ax.text(0.032, y, f"R{sub(k + 1)}", fontsize=FS_BODY, fontweight="bold",
                color=ACCENT, va="center")
        ax.text(0.085, y,
                f"z{sub(k + 1)} = {num(rule.continuous[rule.feature], 3, True)}"
                f" \u00b7 {rule.feature}",
                fontsize=FS_BODY, color=INK, va="center")
        ax.text(0.545, y,
                f"fires while {num(rule.center - rule.width)}"
                f" \u2264 z{sub(k + 1)}"
                f" \u2264 {num(rule.center + rule.width)}",
                fontsize=FS_SMALL, color=MUTED, va="center")

    # --- footnote ----------------------------------------------------------
    ax.text(0.012, 0.130,
            f"After pruning, {int((np.abs(Wf) > 1e-4).sum())} of the "
            f"{H * N} projection weights survive, exactly one feature per rule; "
            "the output layer stays dense.",
            fontsize=FS_TINY + 0.2, color=MUTED, va="center")
    ax.text(0.012, 0.060,
            "Outside its interval a rule returns exactly zero, so a prediction "
            "is a sum of at most three interval checks.",
            fontsize=FS_TINY + 0.2, color=ACCENT, va="center")

    return save(fig, out_dir, "fig_arch_forward")


# ---------------------------------------------------------------------------
# (b) one hidden unit
# ---------------------------------------------------------------------------
def figure_unit(d: dict, out_dir: Path) -> list[Path]:
    from utils.rules import triangular_activation

    rule = d["show"]
    center, width = rule.center, rule.width

    fig = plt.figure(figsize=(WIDTH_UNIT, 2.95))
    ax = fig.add_axes([0.135, 0.150, 0.845, 0.625])

    fig.text(0.135, 0.955, "(b) Inside one hidden unit",
             fontsize=FS_TITLE, fontweight="bold", color=INK, va="center")
    fig.text(0.135, 0.893,
             f"rule R{sub(rule.index + 1)}:  z{sub(rule.index + 1)} = "
             f"{num(rule.continuous[rule.feature], 3, True)} \u00b7 {rule.feature}",
             fontsize=FS_BODY + 0.6, color=MUTED, va="center")
    fig.text(0.135, 0.845,
             "$a_h = [\\,w_h - |c_h - z|\\,]^+ \\,/\\, (w_h + \\epsilon)$",
             fontsize=FS_BODY + 1.2, color=INK, va="center")

    lo, hi = center - 2.6 * width, center + 2.6 * width
    z = torch.linspace(lo, hi, 600)
    a = triangular_activation(
        z, torch.tensor(center), torch.tensor(width)
    ).numpy()
    ax.plot(z, a, color=ACCENT, lw=1.5, zorder=4)
    ax.fill_between(z, 0, a, color=ACCENT, alpha=0.13, zorder=3)

    for edge in (center - width, center + width):
        ax.axvline(edge, color=MUTED, ls=":", lw=0.8, zorder=2)
    ax.axvline(center, color=ACCENT, lw=0.7, alpha=0.55, zorder=2)

    ax.annotate("", xy=(center - width, 0.26), xytext=(center + width, 0.26),
                arrowprops=dict(arrowstyle="<|-|>", lw=0.8, color=MUTED,
                                shrinkA=0, shrinkB=0))
    ax.text(center, 0.30, "$2w_h$", ha="center", va="bottom",
            fontsize=FS_SMALL + 0.8, color=MUTED)

    ax.set_xlim(lo, hi)
    ax.set_xlabel("projected scalar  $z = W_{f,h:}\\,x$", fontsize=FS_BODY + 1.4,
                  labelpad=1)
    ax.set_ylabel("$a_h$", fontsize=FS_BODY + 1.4, labelpad=0)
    ax.set_ylim(-0.10, 1.16)
    ax.set_yticks([0, 1])
    ax.tick_params(labelsize=FS_TINY + 1.1, length=2, pad=1)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.7)

    ax.annotate("$c_h$", xy=(center, 1.0), xytext=(center + 0.30 * width, 1.06),
                fontsize=FS_SMALL + 0.8, color=ACCENT, va="center",
                arrowprops=dict(arrowstyle="-", lw=0.6, color=ACCENT))

    return save(fig, out_dir, "fig_arch_unit")


# ---------------------------------------------------------------------------
# (c) all three rules as integer scorecards
# ---------------------------------------------------------------------------
def _interval_strip(ax, x, y, w, h, rule):
    """Draw the firing interval on an integer axis inside one card."""
    c, wd = rule.center_scaled, rule.width_scaled
    lo, hi = rule.lower, rule.upper
    span = 2.35 * wd
    left, right = c - span, c + span

    def px(v):
        return x + (v - left) / (right - left) * w

    ax.add_patch(Rectangle((px(lo), y), px(hi) - px(lo), h,
                           facecolor=ACCENT, alpha=0.20, edgecolor=ACCENT,
                           linewidth=0.6, zorder=2))
    ax.plot([x, x + w], [y + h / 2, y + h / 2], color=MUTED, lw=0.5, zorder=1)
    for k in range(int(np.ceil(left)), int(np.floor(right)) + 1):
        pxk = px(k)
        if not (x - 1e-6 <= pxk <= x + w + 1e-6):
            continue
        ax.plot([pxk, pxk], [y + h / 2 - 0.010, y + h / 2 + 0.010],
                color=MUTED, lw=0.5, zorder=3)
        ax.text(pxk, y - 0.012, str(k).replace("-", "\u2212"), ha="center",
                va="top", fontsize=FS_TINY, color=MUTED)


def figure_scorecard(d: dict, out_dir: Path) -> list[Path]:
    """Three cards, one per COMPAS rule, continuous -> scaled integer weight.

    All vertical positions are expressed relative to ``CARD_TOP`` so the card
    can be re-spaced by editing a single block of constants.
    """
    rules = d["rules"]
    budget = d["budget"]

    # --- layout constants (edit these to re-space the cards) ---------------
    FIG_H = 3.05
    CARD_TOP, CARD_BOTTOM = 0.850, 0.230
    HEADER_H = 0.075
    COLHEAD_DY = 0.115
    ROW0_DY, ROW_STEP = 0.170, 0.058
    SCALE_DY = 0.345
    SUM_DY = 0.410
    STRIP_DY, STRIP_H = 0.475, 0.036
    FIRES_DY = 0.550
    CARD_W, GAP = 0.305, 0.022
    X0 = 0.012

    fig = plt.figure(figsize=(WIDTH_FULL, FIG_H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(X0, 0.955, "(c) All three COMPAS rules as integer scorecards",
            fontsize=FS_TITLE, fontweight="bold", color=INK, va="center")
    if budget is not None:
        ax.text(X0, 0.900,
                "one integer scale per rule, chosen by the adaptive search "
                f"(\u039bmax = {budget.max_scale}, "
                f"tolerance {100 * budget.tolerance:g}%)",
                fontsize=FS_SMALL, color=MUTED, va="center")

    x = X0
    for k, rule in enumerate(rules):
        ax.add_patch(FancyBboxPatch(
            (x, CARD_BOTTOM), CARD_W, CARD_TOP - CARD_BOTTOM,
            boxstyle="round,pad=0.006,rounding_size=0.016",
            facecolor="#ffffff", edgecolor=INK, linewidth=0.8, zorder=2,
        ))
        # header bar
        ax.add_patch(FancyBboxPatch(
            (x + 0.005, CARD_TOP - HEADER_H), CARD_W - 0.010, HEADER_H,
            boxstyle="round,pad=0.002,rounding_size=0.012",
            facecolor="#0f172a", edgecolor="none", zorder=3,
        ))
        ax.text(x + 0.020, CARD_TOP - HEADER_H / 2, f"R{sub(k + 1)}",
                fontsize=FS_BODY, fontweight="bold", color="white",
                va="center", zorder=4)
        ax.text(x + 0.062, CARD_TOP - HEADER_H / 2, rule.feature,
                fontsize=FS_SMALL, color="white", va="center", zorder=4)

        # column headers
        y = CARD_TOP - COLHEAD_DY
        ax.text(x + 0.022, y, "parameter", fontsize=FS_TINY, color=MUTED,
                va="center")
        ax.text(x + 0.150, y, "continuous", fontsize=FS_TINY, color=MUTED,
                va="center", ha="center")
        ax.text(x + 0.255, y, "after \u00d7\u03bb", fontsize=FS_TINY, color=MUTED,
                va="center", ha="center")

        rows = [
            ("weight  W", num(rule.continuous[rule.feature], 3, True),
             str(rule.integer_weight).replace("-", "\u2212"), True),
            ("center  c", num(rule.center, 3, True),
             num(rule.center_scaled), False),
            ("width  w", num(rule.width, 3), num(rule.width_scaled), False),
        ]
        y = CARD_TOP - ROW0_DY
        for label, before, after, is_int in rows:
            ax.text(x + 0.022, y, label, fontsize=FS_SMALL, color=INK,
                    va="center")
            ax.text(x + 0.150, y, before, fontsize=FS_SMALL, color=INK,
                    va="center", ha="center")
            ax.text(x + 0.198, y, "\u2192", fontsize=FS_TINY, color=MUTED,
                    va="center", ha="center")
            if is_int:
                ax.add_patch(Rectangle(
                    (x + 0.222, y - 0.019), 0.066, 0.038,
                    facecolor=WARM, edgecolor=WARM_EDGE, linewidth=0.6,
                    zorder=3,
                ))
                ax.text(x + 0.255, y, after, fontsize=FS_SMALL,
                        fontweight="bold", color=WARM_EDGE, va="center",
                        ha="center", zorder=4)
            else:
                ax.text(x + 0.255, y, after, fontsize=FS_SMALL, color=INK,
                        va="center", ha="center")
            y -= ROW_STEP

        # the scale actually chosen for this rule, and its rounding error
        ax.text(x + 0.022, CARD_TOP - SCALE_DY,
                f"\u00d7 \u03bb = {rule.scale:g}   "
                f"(rounding error {100 * rule.rounding_error:.1f}%)",
                fontsize=FS_TINY, color=MUTED, va="center")

        # the integer point sum of this rule
        ax.text(x + 0.022, CARD_TOP - SUM_DY,
                f"h{sub(k + 1)} = {rule.integer_weight:+d} \u00b7 "
                f"{rule.feature}".replace("-", "\u2212"),
                fontsize=FS_SMALL, color=INK, va="center")

        _interval_strip(ax, x + 0.035, CARD_TOP - STRIP_DY,
                        CARD_W - 0.070, STRIP_H, rule)

        ints = rule.firing_integers
        if len(ints) == 1:
            fires = f"fires for h{sub(k + 1)} = {ints[0]}"
        elif len(ints) <= 4:
            fires = (f"fires for h{sub(k + 1)} \u2208 "
                     f"{{{', '.join(str(v) for v in ints)}}}")
        else:
            fires = (f"fires for h{sub(k + 1)} \u2208 "
                     f"{{{ints[0]}, \u2026, {ints[-1]}}}")
        ax.text(x + CARD_W / 2, CARD_TOP - FIRES_DY,
                fires.replace("-", "\u2212"),
                fontsize=FS_SMALL, color=ACCENT, va="center", ha="center")

        x += CARD_W + GAP

    ax.text(X0, 0.150,
            "Only the weights are rounded to integers. Center and width are "
            "rescaled by \u03bb but not rounded, so the firing interval stays exact.",
            fontsize=FS_TINY + 0.2, color=MUTED, va="center")
    ax.text(X0, 0.080,
            "The output weights and biases stay continuous, which is why the "
            "integer rules still reproduce the float predictions.",
            fontsize=FS_TINY + 0.2, color=MUTED, va="center")

    return save(fig, out_dir, "fig_arch_scorecard")


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", choices=["a", "b", "c", "all"], default="all")
    ap.add_argument("--no-checkpoint", action="store_true",
                    help="use the built-in fallback values instead of the checkpoint")
    ap.add_argument("--out-dir", default=str(FIG_DIR))
    args = ap.parse_args()

    d = load_data()

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": FS_BODY,
    })

    out_dir = Path(args.out_dir)
    builders = {"a": figure_forward, "b": figure_unit, "c": figure_scorecard}
    chosen = ["a", "b", "c"] if args.panel == "all" else [args.panel]
    for key in chosen:
        for path in builders[key](d, out_dir):
            print(f"wrote {path}")
    if not d["real"]:
        print("note: checkpoint unavailable, built-in fallback values used")
    return 0


if __name__ == "__main__":
    sys.exit(main())
