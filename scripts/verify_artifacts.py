#!/usr/bin/env python3
"""Health check for the released artefacts.

A reviewer should be able to trust that the repository is internally consistent.
This script checks the invariants that the pipeline relies on, so that a drift
between two copies of the same information fails loudly instead of silently
changing a published number.

Checks
------
1. ``utils.frames`` and ``utils.datasets`` encode each dataset identically.
   Cross-validation builds its own folds from the former while the single-split
   benchmark uses the latter; if the matrices differ, the two protocols are no
   longer comparable.
2. Every ``results/*.csv`` that the notebooks read exists and is non-empty.
3. Every LaTeX table in ``results/tables/`` exists and is non-empty; this is the
   single canonical location that the manuscript reads from.
4. The manuscript references exactly the tables that exist, and no table is
   generated but never referenced.
5. No table value is ``NaN`` or ``--`` where a number is expected.
6. The rule primitive (triangular activation, exact bias shift) and the integer
   quantizer are defined only in ``utils/rules.py``. A second copy anywhere in
   ``utils/``, ``scripts/``, ``paper/neurocomputing/tools/`` or the notebooks is
   reported, because two copies of a formula drift apart silently.
7. ``quantization_fidelity.csv`` and ``scorenet_metrics.csv`` agree bit-for-bit on
   the float accuracy of the same model, and ``Float + Delta`` reproduces the
   scorecard accuracy. Without this the two tables can print the same model as
   95.78 and 95.77.
8. The two latency tables describe the same measurement: the amortized
   per-instance cost is exactly ``batch / M_test``, the declared ``M_test`` matches
   the profile, and the amortized value never exceeds the single-query latency.

Usage
-----
    python scripts/verify_artifacts.py

Exit code 1 on failure, so it can gate a commit.
"""

from __future__ import annotations

import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from decimal import Decimal, InvalidOperation

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

warnings.filterwarnings("ignore")

RESULTS = REPO / "results"
TABLES = RESULTS / "tables"
NC = REPO / "paper" / "neurocomputing"
NC_GEN = NC / "generated"
NC_SECTIONS = NC / "sections"

REQUIRED_CSV = [
    "baselines_compas.csv", "baselines_german.csv", "baselines_adult.csv",
    "baselines_synthetic.csv", "benchmark_comparison.csv",
    "scorenet_metrics.csv", "statistical_tests.csv",
    "cv_compas.csv", "cv_german.csv",
    "ablation_synthetic.csv", "theorem1_bias_shifting.csv",
    "quantization_fidelity.csv", "inference_profile.csv",
]

# ---------------------------------------------------------------------------
# Single-source-of-truth check
# ---------------------------------------------------------------------------
# The rule primitive (triangular activation, bias shift) and the integer
# quantizer are defined only in ``utils/rules.py``. The patterns below flag a
# duplicate, so a second "correct" copy is caught before it silently drifts.
SINGLE_SOURCE = REPO / "utils" / "rules.py"
SCAN_DIRS = ("utils", "scripts", "paper/neurocomputing/tools")
FORBIDDEN = [
    (r"torch\.relu\(\s*\w+\s*-\s*torch\.abs\(", "triangular activation"),
    (r"torch\.round\(\s*[\w\.\[\]]+\s*\*\s*(?:scale|scales|lam|lambda)",
     "integer quantization"),
    (r"np\.round\(\s*[\w\.\[\]]+\s*\*\s*(?:scale|scales|lam|lambda)",
     "integer quantization"),
]


def check_encodings() -> list[str]:
    """The two encoding paths must produce identical matrices.

    The splitting loaders standardise after encoding, so the comparison undoes
    that transform with the returned scaler and then compares the rows as a
    multiset (the split shuffles row order).
    """
    from utils.frames import FRAME_LOADERS
    from utils.datasets import (
        get_preprocessed_synthetic_data,
        load_compas_dataset,
        load_german_credit_dataset,
        load_uci_adult_dataset,
    )

    problems: list[str] = []

    def raw_matrix(splits, scaler):
        X_train, _, X_val, _, X_test, _ = splits
        parts = [X_train] + ([X_val] if X_val is not None else []) + [X_test]
        scaled = np.vstack([np.asarray(p, dtype=np.float64) for p in parts])
        return scaler.inverse_transform(scaled)

    loaders = {
        "compas": lambda: load_compas_dataset(
            return_tensor=False, val_size=0.1, show_info=False),
        "german": lambda: load_german_credit_dataset(
            return_tensor=False, test_size=0.1, val_size=1.0 / 9.0),
        "adult": lambda: load_uci_adult_dataset(
            return_tensor=False, val_size=0.1),
        "synthetic": lambda: get_preprocessed_synthetic_data(
            return_type="numpy", val_size=0.1, return_scaler=True),
    }

    for key, load in loaders.items():
        frame_X, frame_y, frame_names = FRAME_LOADERS[key]()
        out = load()
        if key == "synthetic":
            splits, scaler = out
        else:
            splits, _, scaler = out
        raw = raw_matrix(splits, scaler)

        if frame_X.shape != raw.shape:
            problems.append(f"{key}: frame {frame_X.shape} != split {raw.shape}")
            continue

        a = np.sort(frame_X.astype(np.float64).round(6), axis=0)
        b = np.sort(raw.astype(np.float64).round(6), axis=0)
        if not np.allclose(a, b, atol=1e-5):
            problems.append(f"{key}: encoded columns differ between the two paths")

        if len(frame_names) != frame_X.shape[1]:
            problems.append(
                f"{key}: {len(frame_names)} names for {frame_X.shape[1]} columns")

    return problems


def check_csvs() -> list[str]:
    problems = []
    for name in REQUIRED_CSV:
        path = RESULTS / name
        if not path.exists():
            problems.append(f"missing results/{name}")
        elif len(pd.read_csv(path)) == 0:
            problems.append(f"empty results/{name}")
    for name in ("cv_compas.csv", "cv_german.csv"):
        path = RESULTS / name
        if path.exists():
            n = len(pd.read_csv(path))
            if n != 25:
                problems.append(f"results/{name} has {n} folds, expected 25")
    return problems


def check_tables() -> list[str]:
    problems = []
    if not TABLES.exists():
        return ["missing results/tables/"]

    produced = {p.stem for p in TABLES.glob("*.tex")}
    for name in sorted(produced):
        if not (TABLES / f"{name}.tex").read_text().strip():
            problems.append(f"results/tables/{name}.tex is empty")

    # The neurocomputing build must ship a derived copy of every table.
    shipped = {p.stem for p in NC_GEN.glob("*.tex")} if NC_GEN.exists() else set()
    for name in sorted(produced - shipped):
        problems.append(f"paper/neurocomputing/generated/{name}.tex missing")

    # Every table must be cited by the manuscript or by the supplementary
    # material. Some tables (dataset summary, protected-attribute ablation,
    # training cost, baseline latency) are printed only in the supplement to
    # keep the main text focused; they still have to be referenced there.
    text = "".join(p.read_text() for p in NC_SECTIONS.glob("*.tex"))
    supplementary = NC / "supplementary.tex"
    if supplementary.exists():
        text += supplementary.read_text()
    for name in sorted(produced):
        label = f"tab:{name.removeprefix('tab_')}"
        if f"\\ref{{{label}}}" not in text and f"\\input{{generated/{name}.tex}}" not in text:
            problems.append(f"{name}.tex is generated but never referenced")

    return problems


def check_no_nan() -> list[str]:
    problems = []
    for name in ("benchmark_comparison.csv", "statistical_tests.csv",
                 "quantization_fidelity.csv"):
        path = RESULTS / name
        if not path.exists():
            continue
        df = pd.read_csv(path)
        numeric = df.select_dtypes(include=[np.number])
        nan_cols = [c for c in numeric.columns if numeric[c].isna().any()]
        if nan_cols:
            problems.append(f"results/{name} has NaN in {nan_cols}")
    return problems


def check_float_accuracy_consistency() -> list[str]:
    """The same float model must report one accuracy, not two.

    ``scorenet_metrics.csv`` feeds the main benchmark table
    (``tab:benchmark_comparison``); ``quantization_fidelity.csv`` feeds the
    scorecard-fidelity table (``tab:scorecard_fidelity``) and recomputes the float
    accuracy it compares against. A one-ULP difference is invisible
    scientifically but visible to a reviewer reading the two tables side by side,
    so the invariant is enforced here.

    Checked per ``(Key, Variant)`` pair present in both files:
      * ``Float_Acc_%`` equals ``scorenet_metrics.Acc`` exactly;
      * ``Float_Acc_% + Delta_Acc_% == Scorecard_Acc_%`` (no silent drift in the
        derived column).
    """
    problems: list[str] = []
    metrics_path = RESULTS / "scorenet_metrics.csv"
    fidelity_path = RESULTS / "quantization_fidelity.csv"
    if not metrics_path.exists() or not fidelity_path.exists():
        return problems

    metrics = pd.read_csv(metrics_path)
    fidelity = pd.read_csv(fidelity_path)

    for _, row in fidelity.iterrows():
        match = metrics[
            (metrics["Dataset"] == row["Dataset"]) & (metrics["Variant"] == row["Variant"])
        ]
        if match.empty:
            # compas_noprot is reported in tab:scorecard_fidelity only; nothing to compare.
            continue
        reference = float(match.iloc[0]["Acc"])
        stored = float(row["Float_Acc_%"])
        if stored != reference:
            problems.append(
                f"results/quantization_fidelity.csv {row['Key']}/{row['Variant']}: "
                f"Float_Acc_%={stored!r} but scorenet_metrics.csv Acc={reference!r}"
            )

        derived = stored + float(row["Delta_Acc_%"])
        if abs(derived - float(row["Scorecard_Acc_%"])) > 1e-9:
            problems.append(
                f"results/quantization_fidelity.csv {row['Key']}/{row['Variant']}: "
                f"Float + Delta = {derived!r} but Scorecard = "
                f"{float(row['Scorecard_Acc_%'])!r}"
            )

    # The same invariant must survive rounding in the printed table, otherwise a
    # reader who adds the two displayed columns obtains a third number.
    table = TABLES / "tab_scorecard_fidelity.tex"
    if table.exists():
        for line in table.read_text().splitlines():
            cells = [c.strip().replace(r"\\", "") for c in line.split("&")]
            if len(cells) < 7 or "textbf" in line or "midrule" in line:
                continue
            try:
                shown_float = Decimal(cells[3])
                shown_score = Decimal(cells[4])
                shown_delta = Decimal(cells[5])
            except InvalidOperation:
                continue
            if shown_float + shown_delta != shown_score:
                problems.append(
                    f"results/tables/tab_scorecard_fidelity.tex: "
                    f"{shown_float} {shown_delta} != {shown_score} in row "
                    f"'{cells[1]}'; the displayed row does not add up"
                )

    return problems


def check_latency_definitions() -> list[str]:
    """The two latency tables must describe the same measurement.

    ``tab_computational_efficiency`` reports the batch time and the single-query
    latency; ``tab_cpu_latency_baselines`` reports the same batch time and the
    amortized per-instance cost ``batch / M_test``. These are different
    quantities, so both relations are checked explicitly:

      * the batch time is the same in both tables for the same model;
      * ``amortized_per_sample * M_test`` reproduces that batch time;
      * the amortized value is never larger than the single-query latency.
    """
    problems: list[str] = []
    profile_path = RESULTS / "inference_profile.csv"
    german_path = TABLES / "tab_cpu_latency_baselines.tex"
    if not profile_path.exists() or not german_path.exists():
        return problems

    profile = pd.read_csv(profile_path)
    german = profile[profile["Dataset"] == "German Credit"]
    if german.empty:
        return problems
    n_test = float(german["Test_N"].iloc[0])

    # Announced in the table text; a reader divides batch by it to obtain the
    # amortized column, so it has to be the real test-set size.
    declared = re.search(r"M_\{\\text\{test\}\}\s*=\s*([\d,]+)", german_path.read_text())
    if declared and float(declared.group(1).replace(",", "")) != n_test:
        problems.append(
            f"tab_cpu_latency_baselines.tex declares M_test={declared.group(1)} "
            f"but inference_profile.csv has Test_N={n_test:.0f}"
        )

    for _, row in german.iterrows():
        expected = float(row["CPU_Batch_ms"]) / n_test * 1000.0
        if abs(expected - float(row["CPU_Per_Sample_us"])) > 1e-6:
            problems.append(
                f"inference_profile.csv {row['Variant']}: CPU_Per_Sample_us="
                f"{float(row['CPU_Per_Sample_us'])!r} != batch/M_test={expected!r}"
            )
        if float(row["CPU_Per_Sample_us"]) > float(row["CPU_Single_us"]):
            problems.append(
                f"inference_profile.csv {row['Variant']}: amortized per-sample "
                f"latency exceeds the single-query latency, which is impossible"
            )

    return problems


def check_rule_geometry() -> list[str]:
    """The activation must have the support stated in Definition 1.

    The manuscript defines ``S_h`` as the *closed support* of ``a_h``: the
    activation is strictly positive on the open slab and exactly zero on its
    boundary and outside. The property is asserted here, because an activation
    that leaks a positive value outside the slab would break the geometry
    chapter without changing any number in the tables.
    """
    from utils.rules import triangular_activation

    problems: list[str] = []
    center = torch.tensor([0.0, 0.3, -1.2])
    width = torch.tensor([0.5, 0.2, 0.75])
    margin = 1e-5

    # Strictly outside the slab: |c - z| > w. Must be exactly zero.
    outer = center + torch.sign(torch.tensor([1.0, -1.0, 1.0])) * (width + margin)
    outside = triangular_activation(outer, center, width)
    if not torch.all(outside == 0):
        problems.append(
            "triangular_activation is not exactly zero outside the closed slab: "
            f"got {outside.tolist()}"
        )

    # Strictly inside the slab: |c - z| < w. Must be strictly positive and <= 1.
    inner = center + 0.5 * width
    inside = triangular_activation(inner, center, width)
    if not torch.all(inside > 0) or not torch.all(inside <= 1.0 + 1e-6):
        problems.append(
            "triangular_activation is not strictly positive in (0, 1] inside the "
            f"slab: got {inside.tolist()}"
        )

    # The peak is 1 up to the epsilon regulariser.
    peak = triangular_activation(center, center, width)
    if not torch.allclose(peak, torch.ones_like(peak), atol=1e-6):
        problems.append(f"triangular_activation peak is not 1: got {peak.tolist()}")

    return problems


def check_response_curve_breakpoints() -> list[str]:
    """The response-curve figure must be evaluated at the exact breakpoints.

    Section 3.5 and the caption of the local-response figure state that the
    curves are evaluated only at the algebraic breakpoints, so that they carry
    no interpolation error. A fixed ``linspace`` grid would be accurate to
    ~1e-6 but contradict the text and tie the forward-pass count to a grid
    resolution. This check enforces the claimed property:

      * ``utils.rules.breakpoints`` returns ``c_h \\mp w_h`` and ``c_h`` mapped
        through the affine relation, for a negative weight as well;
      * ``response_breakpoints`` keeps those points, sorts them, and adds the
        domain endpoints;
      * no notebook draws the response curve from a ``linspace`` grid.
    """
    from utils.rules import breakpoints, response_breakpoints

    problems: list[str] = []

    column = torch.tensor([2.0, -3.0, 0.0])
    center = torch.tensor([1.0, -1.0, 0.5])
    width = torch.tensor([0.5, 0.25, 0.1])
    offset = torch.tensor([0.0, 0.5, 0.0])

    # Rule 0 (positive weight):  v- = 0.25, v0 = 0.5, v+ = 0.75.
    # Rule 1 (negative weight):  v- = 7/12 > v+ = 5/12, with v0 = 0.5 between
    # them. The second rule is the regression case: its "minus" breakpoint lies
    # to the right of its "plus" breakpoint, which is exactly what the sort
    # absorbs and what the paper now states explicitly.
    points = breakpoints(column, center, width, offset).sort().values
    expected = torch.tensor(
        [0.25, 5.0 / 12.0, 0.5, 0.5, 7.0 / 12.0, 0.75]
    ).sort().values
    if not torch.allclose(points, expected, atol=1e-6):
        problems.append(
            f"breakpoints() returned {points.tolist()}, expected {expected.tolist()}"
        )

    curve = response_breakpoints(
        column, center, width, offset, lower=-2.0, upper=2.0, extra=[0.1]
    )
    for required in (0.1, -2.0, 2.0):
        if not torch.any(torch.isclose(curve, torch.tensor(required), atol=1e-9)):
            problems.append(f"response_breakpoints() dropped the point {required}")
    if not torch.all(curve[1:] > curve[:-1]):
        problems.append("response_breakpoints() is not strictly increasing")

    # No notebook may draw the response curve from a dense grid.
    for path in sorted((REPO / "notebooks").rglob("*.ipynb")):
        try:
            notebook = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        for index, cell in enumerate(notebook.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            if "response_breakpoints" in source or "fig_local_feature_curves" not in source:
                continue
            problems.append(
                f"{path.relative_to(REPO)}[cell {index}]: the response-curve figure "
                "does not use utils.rules.response_breakpoints"
            )

    return problems


def check_training_pipeline_runs() -> list[str]:
    """The training entry point must actually be importable and runnable.

    ``utils/train.py`` is the only place that produces checkpoints, so a missing
    import or a renamed constant there is invisible until someone retrains. This
    check runs a one-iteration pipeline on a tiny synthetic problem, so such a
    failure surfaces here instead of on a six-minute seed sweep.
    """
    problems: list[str] = []
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        from utils.model import SCORENet
        from utils.train import run_iterative_pruning_pipeline

        torch.manual_seed(0)
        n, d, c = 64, 5, 2
        X = torch.randn(n, d)
        y = (X[:, 0] + 0.5 * X[:, 1] > 0).long()
        data = TensorDataset(X, y)
        train_loader = DataLoader(data, batch_size=16, shuffle=True)
        val_loader = DataLoader(data, batch_size=16)

        model = SCORENet(d, 4, c, dropout_rate=0.0, device="cpu")
        global_best, final_iter = run_iterative_pruning_pipeline(
            model,
            train_loader,
            val_loader,
            "cpu",
            base_epochs=1,
            prune_epochs=1,
            num_prune_iters=1,
            prune_amount_feature=0.2,
            verbose=False,
        )
        if global_best is None or final_iter is None:
            problems.append("run_iterative_pruning_pipeline returned None")
    except Exception as exc:  # noqa: BLE001 - any failure here is the finding
        problems.append(
            f"utils/train.py cannot run: {type(exc).__name__}: {exc}"
        )
    return problems


def check_seed_stability_signatures() -> list[str]:
    """Rule signatures must respect the weight threshold.

    ``scripts/run_seed_stability.py`` summarises a rule by the set of features it
    uses. That set must come from the *thresholded* mask: the raw weight would
    include every column, because float weights are never exactly zero, and the
    signature Jaccard would silently jump. This check pins the property with a
    synthetic model whose weights are known.
    """
    import importlib.util

    problems: list[str] = []
    script = REPO / "scripts" / "run_seed_stability.py"
    if not script.exists():
        return problems

    spec = importlib.util.spec_from_file_location("_seed_stability", script)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001
        return [f"scripts/run_seed_stability.py cannot be imported: {exc}"]

    class _Stub:
        """Minimal stand-in exposing only what ``rule_set`` reads."""

        class feature_linear:
            pass

    weight = torch.zeros(2, 4)
    weight[0, 1] = 0.5          # rule 0 uses feature 1 only
    weight[1, 2] = 1e-6         # below threshold: this unit must not count as a rule
    weight[1, 3] = 0.25         # rule 1 uses feature 3 only
    _Stub.feature_linear.weight = type("W", (), {"detach": lambda self: type(
        "D", (), {"cpu": lambda self: type("C", (), {"numpy": lambda self: weight})()}
    )()})()

    info = module.rule_set(_Stub)
    if info["n_rules"] != 2:
        problems.append(f"rule_set counted {info['n_rules']} rules, expected 2")
    if info["support"] != frozenset({1, 3}):
        problems.append(
            f"rule_set support is {sorted(info['support'])}, expected [1, 3] "
            "(a sub-threshold weight must not leak into the signature)"
        )
    if info["n_links"] != 2:
        problems.append(f"rule_set counted {info['n_links']} links, expected 2")

    return problems


def check_link_budget() -> list[str]:
    """The connection budget quoted in the paper must match the code's schedule.

    Section 3.3.2 states that iterative magnitude pruning removes a fixed
    fraction of the *remaining* weights, so the count follows
    ``R_{k+1} = R_k - round(p R_k)``. The recurrence is recomputed here and
    compared with the largest count observed across seeds, so a changed
    ``prune_amount`` or ``prune_iters`` cannot silently stale the quoted numbers.
    """
    problems: list[str] = []
    from utils.experiments import DATASETS

    seed_path = RESULTS / "seed_stability.csv"
    if not seed_path.exists():
        return problems
    observed = pd.read_csv(seed_path)
    observed = observed[observed["checkpoint"] == "Compact"]

    for _, row in observed.iterrows():
        key = row["dataset"]
        spec = DATASETS.get(key)
        if spec is None:
            continue
        remaining = spec.h_hidden * (13 if key == "compas" else 48)
        for _ in range(spec.prune_iters):
            remaining -= round(spec.prune_amount * remaining)
        if int(row["links_max"]) != remaining:
            problems.append(
                f"{key}: seed_stability.csv reports links_max={int(row['links_max'])} "
                f"but the schedule of Eq. (10) gives {remaining} for "
                f"p={spec.prune_amount}, K={spec.prune_iters}"
            )
        if int(row["links_min"]) > remaining:
            problems.append(
                f"{key}: links_min={int(row['links_min'])} exceeds the schedule "
                f"bound {remaining}"
            )
    return problems


def check_single_source() -> list[str]:
    """The rule primitive and the quantizer must live in one module only.

    A second copy of ``a_h(z)`` or of ``round(lambda * W)`` is the kind of
    drift that silently changes a published number, so any occurrence outside
    ``utils/rules.py`` is reported.
    """
    problems: list[str] = []

    def scan(text: str, label: str) -> None:
        for pattern, feature in FORBIDDEN:
            for match in re.finditer(pattern, text):
                line = text.count("\n", 0, match.start()) + 1
                problems.append(
                    f"{label}:{line} re-implements the {feature}; "
                    "import it from utils/rules.py"
                )

    targets = []
    for folder in SCAN_DIRS:
        base = REPO / folder
        if base.exists():
            targets.extend(base.rglob("*.py"))
    for path in targets:
        if path.resolve() == SINGLE_SOURCE.resolve():
            continue
        scan(path.read_text(errors="ignore"), str(path.relative_to(REPO)))

    for path in sorted((REPO / "notebooks").rglob("*.ipynb")):
        try:
            notebook = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        for index, cell in enumerate(notebook.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            scan(source, f"{path.relative_to(REPO)}[cell {index}]")

    return problems


# Absolute statements that are false because two registered choices are
# test-informed: the German Credit pruning depth and the quantization budget.
# The manuscript declares both in Sections 4.1.2 and 4.8; this guard stops the
# blanket claim from being reintroduced.
FORBIDDEN_TEST_CLAIMS = (
    "No test example is used to choose",
    "none is tuned on the test set",
    "the test partition is never used for model selection",
    "with an untouched test set",
)


def check_no_absolute_test_claims() -> list[str]:
    """Reject blanket claims that no configuration choice saw test data."""
    problems: list[str] = []
    sources = sorted(NC_SECTIONS.glob("*.tex")) + [NC / "main.tex"]
    sources += [NC / "submission" / "cover_letter.tex"]
    for path in sources:
        if not path.exists():
            continue
        text = path.read_text()
        for phrase in FORBIDDEN_TEST_CLAIMS:
            if phrase.lower() in text.lower():
                problems.append(
                    f"{path.relative_to(REPO)} still claims {phrase!r}; "
                    "declare the two test-informed operating points instead "
                    "(Sections 4.1.2 and 4.8)"
                )
    return problems


def main() -> int:
    groups = [
        ("encodings", check_encodings),
        ("csv artefacts", check_csvs),
        ("latex tables", check_tables),
        ("missing numbers", check_no_nan),
        ("single source of truth", check_single_source),
        ("float accuracy consistency", check_float_accuracy_consistency),
        ("latency definitions", check_latency_definitions),
        ("rule geometry", check_rule_geometry),
        ("training pipeline runs", check_training_pipeline_runs),
        ("seed-stability signatures", check_seed_stability_signatures),
        ("link budget", check_link_budget),
        ("response-curve breakpoints", check_response_curve_breakpoints),
        ("no absolute test claims", check_no_absolute_test_claims),
    ]
    failures = 0
    for label, fn in groups:
        try:
            problems = fn()
        except Exception as exc:  # noqa: BLE001
            problems = [f"crashed: {type(exc).__name__}: {exc}"]
        if problems:
            failures += len(problems)
            print(f"FAIL  {label}")
            for p in problems:
                print(f"        {p}")
        else:
            print(f"OK    {label}")

    print()
    if failures:
        print(f"{failures} problem(s) found.")
        return 1
    print("All artefact invariants hold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
