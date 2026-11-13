#!/usr/bin/env python3
"""Run every verification gate in one command.

Why this exists
---------------
Several avoidable defects during preparation had the same shape: a change was
made in one file, a *cached* artefact still looked correct, and the breakage
surfaced much later (or not at all). Three concrete examples:

* ``utils/train.py`` used ``WEIGHT_THRESHOLD`` without importing it, so every
  call to the training pipeline raised ``NameError`` while all cached
  checkpoints and tables remained valid. Nothing caught it because nothing
  re-ran training.
* A verification script imported a module whose ``__pycache__`` entry was stale,
  so a broken version was tested instead of the fixed one.
* A rule signature was built from raw weights instead of the thresholded mask,
  which inflated a reported overlap from 0.13 to 0.60. The number was plausible
  and nothing recomputed it.

The common factor is *trusting a cached result instead of executing the code*.
This script therefore, before anything else:

1. deletes every ``__pycache__`` directory, so no stale bytecode can be imported;
2. byte-compiles ``utils/``, ``scripts/`` and the manuscript tools, which catches
   syntax errors immediately;
3. imports every module once, which catches missing imports at definition time
   rather than at call time;
4. runs the artefact, notebook and manuscript-number checks;
5. optionally rebuilds the PDF and audits figure typography.

Usage
-----
    python scripts/check_all.py            # checks only (fast, ~1 min)
    python scripts/check_all.py --build    # also rebuild the manuscript
    python scripts/check_all.py --quiet    # one line per gate

Exit code 1 if any gate fails.
"""

from __future__ import annotations

import argparse
import compileall
import importlib
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PAPER = REPO / "paper" / "neurocomputing"

# Running ``python scripts/check_all.py`` puts ``scripts/`` on sys.path, not the
# repository root, so ``import utils`` would fail. Fix the path before any import.
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

COMPILE_DIRS = ("utils", "scripts", "paper/neurocomputing/tools")
IMPORT_MODULES = (
    "utils.experiments",
    "utils.rules",
    "utils.model",
    "utils.train",
    "utils.analysis",
    "utils.custom_utils",
    "utils.plot_utils",
    "utils.baselines",
    "utils.cross_validation",
    "utils.datasets",
    "utils.frames",
    "utils.statistics",
    "utils.reproducibility",
)


def purge_pycache() -> int:
    """Remove every ``__pycache__`` directory under the repository."""
    removed = 0
    for path in REPO.rglob("__pycache__"):
        if ".git" in path.parts:
            continue
        shutil.rmtree(path, ignore_errors=True)
        removed += 1
    return removed


def compile_all() -> list[str]:
    problems = []
    for folder in COMPILE_DIRS:
        base = REPO / folder
        if not base.exists():
            continue
        if not compileall.compile_dir(
            str(base), quiet=2, force=True, legacy=False
        ):
            problems.append(f"{folder}: byte-compilation failed (syntax error)")
    return problems


def import_all() -> list[str]:
    problems = []
    for name in IMPORT_MODULES:
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - any import failure is the finding
            problems.append(f"{name}: {type(exc).__name__}: {exc}")
    return problems


def run(script: list[str], cwd: Path) -> tuple[bool, str]:
    proc = subprocess.run(script, cwd=str(cwd), capture_output=True, text=True)
    output = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode == 0, output.strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", action="store_true", help="also rebuild the PDF")
    ap.add_argument("--quiet", action="store_true", help="one line per gate")
    args = ap.parse_args()

    py = sys.executable
    failures: list[str] = []

    removed = purge_pycache()
    print(f"[1] removed {removed} __pycache__ director{'y' if removed == 1 else 'ies'}")

    problems = compile_all()
    print(f"[2] byte-compile {COMPILE_DIRS}: "
          + ("OK" if not problems else "FAILED"))
    failures += problems

    problems = import_all()
    print(f"[3] import {len(IMPORT_MODULES)} modules: "
          + ("OK" if not problems else "FAILED"))
    for problem in problems:
        print(f"      {problem}")
    failures += problems

    gates = [
        ("notebooks", [py, "scripts/check_notebooks.py"], REPO, True),
        ("artefacts", [py, "scripts/verify_artifacts.py"], REPO, True),
        ("manuscript numbers",
         [py, "paper/neurocomputing/tools/check_manuscript_numbers.py"], REPO, True),
        # Elsevier's artwork guidance requires >= 6 pt at the printed size.
        # All eleven figures now pass; the gate stays blocking so a future
        # oversized export cannot slip through as a warning.
        ("figure typography", [py, "audit_figures.py"], PAPER, True),
    ]
    for index, (label, script, cwd, blocking) in enumerate(gates, start=4):
        ok, output = run(script, cwd)
        status = "OK" if ok else ("WARN" if not blocking else "FAILED")
        print(f"[{index}] {label}: {status}")
        if ok and args.quiet:
            continue
        if not ok:
            if not args.quiet:
                for line in output.splitlines()[-25:]:
                    print(f"      {line}")
            if blocking:
                failures.append(label)

    if args.build:
        ok, output = run(["./build.sh"], PAPER)
        print(f"[{len(gates) + 4}] manuscript build: " + ("OK" if ok else "FAILED"))
        if not ok:
            for line in output.splitlines()[-25:]:
                print(f"      {line}")
            failures.append("build")

    print()
    if failures:
        print(f"{len(failures)} gate(s) failed: {', '.join(sorted(set(failures)))}")
        return 1
    print("all gates passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
