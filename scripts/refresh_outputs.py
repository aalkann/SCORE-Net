#!/usr/bin/env python3
"""Re-execute the display notebooks in place so their outputs are committed.

``scripts/build_notebooks.py`` writes ``01_baselines`` and
``02_cross_validation`` from source, which means those two files are always
regenerated **without** outputs. The analysis notebooks (03-07) are stored in the
repository together with their outputs so a reader can see the numbers without
running anything.

This script refreshes the stored outputs. It only *reads* cached artefacts, so it
is fast: no model is retrained and no notebook recomputes a benchmark.

Usage
-----
    python scripts/refresh_outputs.py                # 01, 02, 07
    python scripts/refresh_outputs.py --only 07_tables

Run it after a cross-validation finishes, so that notebook 02 shows the
completed tables. Then commit the notebooks.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NOTEBOOKS = REPO / "notebooks"

DEFAULT = ("01_baselines", "02_cross_validation", "07_tables")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="+", default=list(DEFAULT),
                    help="notebook stems to refresh (default: 01 02 07)")
    args = ap.parse_args()

    failures = 0
    for stem in args.only:
        path = NOTEBOOKS / f"{stem}.ipynb"
        if not path.exists():
            print(f"skipped (missing): {path.name}")
            continue
        print(f"==> {path.name}")
        proc = subprocess.run(
            [
                sys.executable, "-m", "jupyter", "nbconvert",
                "--to", "notebook", "--execute", "--inplace",
                "--ExecutePreprocessor.timeout=3600",
                path.name,
            ],
            cwd=NOTEBOOKS,
        )
        if proc.returncode != 0:
            failures += 1
            print(f"    FAILED (exit {proc.returncode})")
        else:
            print("    OK")

    print()
    if failures:
        print(f"{failures} notebook(s) failed to execute.")
        return 1
    print("Outputs refreshed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
