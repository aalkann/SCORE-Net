#!/usr/bin/env python3
"""Derive the Neurocomputing table copies from the canonical generated tables.

The canonical tables live in ``results/tables/*.tex`` and are produced by
``notebooks/07_tables.ipynb``. They are tuned for the IEEE
two-column page (517.9 pt, Times). Elsevier's ``elsarticle`` 5p layout uses
522.0 pt with a slightly different measure, and two canonical tables wrap
themselves in ``\\resizebox{\\textwidth}``, which *enlarges* narrow tables to
12 pt and breaks consistency with the rest.

This script never edits the canonical files and never changes a value. It only
normalises presentation so that every table is set at the same size:

1. ``\\footnotesize`` is applied to every table (8.03 pt at the 10 pt base),
   Elsevier's usual table size.
2. Existing ``\\resizebox{\\textwidth}{!}{...}`` wrappers are removed, because
   they scale narrow tables up.
3. Each table is measured. If it overflows its column, ``\\tabcolsep`` is
   tightened (2.0 -> 1.5 -> 1.0 -> 0.8 pt) until it fits at 8.03 pt.
4. As a final guarantee the tabular is wrapped in
   ``\\adjustbox{max width=\\linewidth}{...}``, which shrinks only if the table
   is still too wide and never enlarges it.

Step 3 needs ``pdflatex``. If it is unavailable the script still emits correct
tables (step 4 alone guarantees they fit); they may simply be scaled slightly.

The output is idempotent: re-running on its own output is a no-op.

Run from the ``neurocomputing`` directory:  python prepare_tables.py
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1] / "results" / "tables"
DST = HERE / "generated"

TABULAR = re.compile(r"(\\begin\{tabular\}.*?\\end\{tabular\})", re.DOTALL)
RESIZEBOX = re.compile(
    r"\\resizebox\{\\textwidth\}\{!\}\{%\s*\n(.*?)\n\}\s*\n", re.DOTALL
)
ADJUST_OPEN = r"\adjustbox{max width=\linewidth}{%"
ADJUST_CLOSE = "}"
TABCOLSEP = re.compile(r"\\setlength\{\\tabcolsep\}\{[\d.]+pt\}")

# Width of the text block and of a single column in elsarticle 5p, 10 pt.
TARGET = {"table*": 522.0, "table": 252.2}

# Inter-column padding tried, largest first.
TABCOLSEPS = (2.0, 1.5, 1.0, 0.8)


# ---------------------------------------------------------------------------
# Measuring
# ---------------------------------------------------------------------------
PROBE = r"""\documentclass[5p,10pt]{elsarticle}
\usepackage{amsmath,amssymb,booktabs,multirow,tabularx,adjustbox}
\usepackage{newtxtext,newtxmath}
\newcommand{\modelname}{\textsc{SCORE-Net}}
\begin{document}
\newsavebox\mybox
\sbox\mybox{\footnotesize TABCOLSEP TABULARBODY}
\typeout{MEASUREDWIDTH=\the\wd\mybox}
x
\end{document}
"""


def measure(tabular: str, tabcolsep: float, tmp: Path) -> float | None:
    """Return the natural width of ``tabular`` at \\footnotesize, in points.

    ``\\setlength{\\tabcolsep}`` normally sits outside the ``tabular``
    environment, so it has to be re-applied inside the measuring box.
    """
    tex = tmp / "probe.tex"
    body = f"\\setlength{{\\tabcolsep}}{{{tabcolsep}pt}} " + tabular
    tex.write_text(PROBE.replace("TABCOLSEP", "").replace("TABULARBODY", body))
    try:
        proc = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-output-directory", str(tmp), str(tex)],
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, OSError):
        return None
    match = re.search(r"MEASUREDWIDTH=([\d.]+)pt", proc.stdout + proc.stderr)
    return float(match.group(1)) if match else None


# ---------------------------------------------------------------------------
# Transforming
# ---------------------------------------------------------------------------
def strip_resizebox(text: str) -> str:
    """Replace the canonical enlarging resizebox with its bare contents."""
    return RESIZEBOX.sub(lambda m: m.group(1) + "\n", text)


def add_footnotesize(text: str) -> str:
    if r"\footnotesize" in text:
        return text
    return text.replace(r"\centering", "\\centering\n\\footnotesize", 1)


def set_tabcolsep(text: str, value: float) -> str:
    """Set \\tabcolsep for this table, inserting it just before \\begin{tabular}."""
    line = "\\setlength{\\tabcolsep}{%gpt}" % value
    if CURRENT_SEP.search(text):
        return CURRENT_SEP.sub(lambda m: line, text, count=1)
    return text.replace(r"\begin{tabular}", line + "\n" + r"\begin{tabular}", 1)


def add_adjustbox(text: str) -> str:
    if r"\adjustbox{" in text:
        return text

    def wrap(match: re.Match[str]) -> str:
        return f"{ADJUST_OPEN}\n{match.group(1)}\n{ADJUST_CLOSE}"

    return TABULAR.sub(wrap, text)


CURRENT_SEP = re.compile(r"\\setlength\{\\tabcolsep\}\{([\d.]+)pt\}")
DEFAULT_SEP = 6.0  # elsarticle default


def current_tabcolsep(text: str) -> float:
    match = CURRENT_SEP.search(text)
    return float(match.group(1)) if match else DEFAULT_SEP


def derive(text: str, tmp: Path | None) -> str:
    text = strip_resizebox(text)
    text = add_footnotesize(text)

    env = "table*" if "table*" in text else "table"
    target = TARGET[env]

    if tmp is not None:
        sep = current_tabcolsep(text)
        # Try the table's own padding first; tighten only while it overflows.
        candidates = [sep] + [s for s in TABCOLSEPS if s < sep]
        for candidate in candidates:
            tabular = TABULAR.search(text)
            if tabular is None:
                break
            width = measure(tabular.group(1), candidate, tmp)
            if width is None:
                break
            text = set_tabcolsep(text, candidate)
            if width <= target:
                break

    return add_adjustbox(text)


def main() -> int:
    DST.mkdir(parents=True, exist_ok=True)
    sources = sorted(SRC.glob("*.tex"))
    has_pdflatex = shutil.which("pdflatex") is not None
    if not has_pdflatex:
        print("[warning] pdflatex not found: tabcolsep adjustment skipped "
              "(tables still fit via adjustbox).")

    changed = 0
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) if has_pdflatex else None
        for src in sources:
            out = derive(src.read_text(), tmp)
            dst = DST / src.name
            if not dst.exists() or dst.read_text() != out:
                dst.write_text(out)
                changed += 1
                print(f"written: {dst.name}")
            else:
                print(f"unchanged: {dst.name}")
    print(f"\n{changed} files updated ({len(sources)} tables).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
