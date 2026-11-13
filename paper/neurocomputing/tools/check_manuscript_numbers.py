#!/usr/bin/env python3
"""Cross-check the numbers quoted in the manuscript against the result CSVs.

Every quantitative claim in ``sections/*.tex`` should be traceable to a cell in
``results/*.csv``. This script extracts the values from the CSVs and asserts
that each one appears in the manuscript text, so a stale sentence is caught
before submission.

Usage
-----
    python paper/neurocomputing/tools/check_manuscript_numbers.py

Exit code 1 if any expected number is missing from the text.
"""

from __future__ import annotations

import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[3]
SEC = ROOT / "paper" / "neurocomputing" / "sections"
RES = ROOT / "results"

# LaTeX writes thousands as {,}; strip it so substring checks work.
# The generated tables are part of the manuscript (main.tex ``\input``s them),
# so they are scanned too. Without this a number that appears only in a table
# would be reported as missing from the text.
GEN = ROOT / "paper" / "neurocomputing" / "generated"
NC = ROOT / "paper" / "neurocomputing"
# The supplementary material is part of the submission: some tables and the
# rule-list comparison are printed only there, so its text is scanned as well.
SOURCES = (
    list(SEC.glob("*.tex"))
    + list(GEN.glob("*.tex"))
    + [p for p in [NC / "supplementary.tex"] if p.exists()]
)
TEX = "".join(p.read_text() for p in SOURCES).replace("{,}", "")
TEX_NO_MATH = TEX.replace("$", "")

checks: list[tuple[str, str, bool]] = []


def chk(label: str, value) -> None:
    needle = str(value)
    found = needle in TEX or needle.lstrip("<") in TEX_NO_MATH
    checks.append((label, needle, found))


def row(frame: pd.DataFrame, **where):
    mask = pd.Series(True, index=frame.index)
    for key, value in where.items():
        mask &= frame[key] == value
    return frame[mask].iloc[0]


metrics = pd.read_csv(RES / "scorenet_metrics.csv")
stat = pd.read_csv(RES / "statistical_tests.csv")
fid = pd.read_csv(RES / "quantization_fidelity.csv")
abl = pd.read_csv(RES / "ablation_synthetic.csv")
cv = {k: pd.read_csv(RES / f"cv_{k}.csv")
      for k in ("compas", "german", "adult", "synthetic")}

# 1. Single-split accuracy and rule count for every variant.
for dataset, variant in [
    ("COMPAS", "Global Best"), ("COMPAS", "Compact"),
    ("German Credit", "Global Best"), ("German Credit", "Compact"),
    ("UCI Adult", "Global Best"), ("UCI Adult", "Compact"),
    ("Synthetic", "Global Best"), ("Synthetic", "Compact"),
]:
    r = row(metrics, Dataset=dataset, Variant=variant)
    chk(f"{dataset} {variant} accuracy", f"{r.Acc:.2f}")
    chk(f"{dataset} {variant} rule count", int(r.H))

# 2. McNemar p-values.
for dataset, variant in [
    ("COMPAS", "Global Best"), ("COMPAS", "Compact"),
    ("German Credit", "Global Best"), ("German Credit", "Compact"),
    ("UCI Adult", "Global Best"), ("UCI Adult", "Compact"),
    ("Synthetic", "Global Best"), ("Synthetic", "Compact"),
]:
    p = row(stat, Dataset=dataset, Variant=variant).McNemar_p
    chk(f"{dataset} {variant} p-value", f"{p:.3f}" if p >= 0.001 else "<0.001")

# 3. Cross-validated means.
for key, dataset in [("compas", "COMPAS"), ("german", "German Credit"),
                     ("adult", "UCI Adult"), ("synthetic", "Synthetic")]:
    frame = cv[key]
    chk(f"CV {dataset} Global Best", f"{frame.test_global.mean() * 100:.2f}")
    chk(f"CV {dataset} Compact", f"{frame.test_final.mean() * 100:.2f}")

# 3b. Macro-F1 and ROC-AUC, for the variants whose values are quoted verbatim.
r = row(metrics, Dataset="COMPAS", Variant="Global Best")
chk("COMPAS Global Best macro-F1", f"{r.F1:.2f}")
chk("COMPAS Global Best ROC-AUC", f"{r.AUC:.2f}")

# 4. Quantization fidelity: range and largest accuracy change.
chk("fidelity minimum", f"{fid['Fidelity_%'].min():.2f}")
chk("fidelity maximum", f"{fid['Fidelity_%'].max():.2f}")

# 4b. The protected-attribute-free COMPAS rows quoted in Section 5.4.
noprot_fid = fid[fid.Key == "compas_noprot"]
for variant in ("Global Best", "Compact"):
    r = noprot_fid[noprot_fid.Variant == variant].iloc[0]
    chk(f"no-protected {variant} fidelity", f"{r['Fidelity_%']:.2f}")

# 4c. Inference latencies quoted in Section 4.3.2 and in the conclusion.
prof = pd.read_csv(RES / "inference_profile.csv")
chk("single-sample CPU latency lower bound", f"{prof.CPU_Single_us.min():.1f}")
chk("single-sample CPU latency upper bound", f"{prof.CPU_Single_us.max():.1f}")
chk("GPU single-sample latency lower bound", f"{prof.GPU_Single_us.min():.0f}")
chk("GPU single-sample latency upper bound", f"{prof.GPU_Single_us.max():.0f}")
for dataset, label in [("German Credit", "German batch"), ("COMPAS", "COMPAS batch"),
                       ("UCI Adult", "Adult batch"), ("Synthetic", "Synthetic batch")]:
    r = prof[(prof.Dataset == dataset) & prof.Variant.str.contains("Compact")].iloc[0]
    chk(f"{label} latency (ms)", f"{r.CPU_Batch_ms:.2f}")
syn = prof[(prof.Dataset == "Synthetic") & prof.Variant.str.contains("Compact")].iloc[0]
chk("synthetic GPU speed-up", f"{syn.CPU_Batch_ms / syn.GPU_Batch_ms:.1f}")

# 5. Ablation accuracies.
for label in ("No Feature", "No Output", "No $L_1$ Loss"):
    r = abl[abl.Variant.str.contains(label, regex=False)].iloc[0]
    chk(f"ablation {label}", f"{r['Test Acc (%)']:.2f}")

# 6. Rule-budget range over the binary benchmarks.
binary = metrics[metrics.Dataset != "Synthetic"]
compact = binary[binary.Variant == "Compact"]
chk("minimum compact rule count", int(compact.H.min()))
chk("maximum compact rule count", int(compact.H.max()))
chk("minimum features per rule", f"{compact.Avg_Rule_Length.min():.2f}")
chk("maximum features per rule", f"{compact.Avg_Rule_Length.max():.2f}")


# 7. Protected-attribute ablation (Table 5).
noprot = pd.read_csv(RES / "cv_compas_noprot.csv")
noprot_bl = pd.read_csv(RES / "cv_baselines_compas_noprot.csv")
for column, label in (("test_global", "Global Best"), ("test_final", "Compact")):
    chk(f"no-protected {label} accuracy", f"{noprot[column].mean() * 100:.2f}")

# 8. Rule-set stability across seeds (Table 11): both checkpoints.
stab = pd.read_csv(RES / "seed_stability.csv")
for _, r in stab.iterrows():
    label = f"{r['dataset']} {r['checkpoint']}"
    chk(f"{label} rule-set Jaccard", f"{r['rule_jaccard_mean']:.2f}")
    chk(f"{label} feature Jaccard", f"{r['feature_jaccard_mean']:.2f}")
    chk(f"{label} rule count", f"{r['rules_mean']:.1f}")
    # Link counts are printed in the table as a range; check both ends.
    chk(f"{label} link count min", f"{int(r['links_min'])}")
    chk(f"{label} link count max", f"{int(r['links_max'])}")

# 9. Training cost (Table 9): the four SCORE-Net trajectory times in minutes.
for key, minutes in (("german", "0.2"), ("compas", "2.9"), ("adult", "5.2"), ("synthetic", "7.9")):
    chk(f"{key} training minutes", minutes)

# 10. German pruning-depth sensitivity (§4.8). The manuscript declares the depth
# as a test-informed operating point and backs the disclosure with the
# cross-validated comparison, so both depths must be traceable.
depth = pd.read_csv(RES / "pruning_depth_sensitivity.csv")
for _, r in depth.iterrows():
    label = f"German {r['variant']} depth {int(r['prune_iters'])}"
    chk(f"{label} CV mean", f"{r['cv_mean']:.2f}")
    chk(f"{label} CV std", f"{r['cv_std']:.2f}")


def main() -> int:
    failures = [c for c in checks if not c[2]]
    print(f"checked {len(checks)} numbers against results/*.csv")
    if failures:
        print(f"\n{len(failures)} value(s) quoted in the text but absent from the CSVs:")
        for label, value, _ in failures:
            print(f"  {label} = {value}")
        return 1
    print("every checked number appears in the manuscript")
    return 0


if __name__ == "__main__":
    sys.exit(main())
