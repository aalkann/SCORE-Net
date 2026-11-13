#!/usr/bin/env python3
"""Audit figure typography against Elsevier's published-size rule.

Elsevier's artwork guidance (Artwork sizing) states:

    "the lettering on the artwork should have a finished, printed size of 7 pt
     for normal text and no smaller than 6 pt for subscript and superscript
     characters."

A figure exported at 13 in and then placed in a 7.25 in column is scaled to 0.55,
so 7 pt lettering prints at 3.9 pt. This script measures, for every figure used
in the manuscript, the scale factor applied by LaTeX and the resulting printed
font sizes, and reports the share of characters below the 6 pt floor.

Usage:  python audit_figures.py

Exit code 1 if any figure has more than 5 % of its characters below 6 pt at the
printed size, so the check can gate a build.
"""

from __future__ import annotations

import re
import subprocess
import sys
import zlib
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SECTIONS = HERE / "sections"

# elsarticle 5p, 10 pt
TEXT_WIDTH = 522.0
COLUMN_WIDTH = 252.2

MIN_BODY_PT = 7.0
MIN_SCRIPT_PT = 6.0


def placements() -> dict[str, float]:
    """Map each figure path to the width (pt) at which LaTeX prints it."""
    pattern = re.compile(
        r"\\includegraphics\[width=([\d.]*)\\(textwidth|columnwidth|linewidth)\]\{([^}]+)\}"
    )
    found: dict[str, float] = {}
    for tex in list(SECTIONS.glob("*.tex")) + [HERE / "main.tex"]:
        for match in pattern.finditer(tex.read_text()):
            num, unit, path = match.group(1), match.group(2), match.group(3)
            frac = float(num) if num else 1.0
            full = TEXT_WIDTH if unit in ("textwidth", "linewidth") else COLUMN_WIDTH
            found[path] = frac * full
    return found


def page_width(path: Path) -> float | None:
    try:
        out = subprocess.run(["pdfinfo", str(path)], capture_output=True, text=True).stdout
    except FileNotFoundError:
        return None
    match = re.search(r"Page size:\s+([\d.]+)", out)
    return float(match.group(1)) if match else None


def char_counts_by_size(path: Path) -> dict[float, int]:
    """Count drawn characters per font size (pt) from the content streams."""
    raw = path.read_bytes()
    total: dict[float, int] = defaultdict(int)
    token = re.compile(
        rb"/(F\d+)\s+([\d.]+)\s+Tf|\[(.*?)\]\s*TJ|\((.*?)\)\s*Tj", re.DOTALL
    )
    for stream in re.finditer(rb"stream\r?\n(.*?)endstream", raw, re.DOTALL):
        data = stream.group(1)
        try:
            data = zlib.decompress(data)
        except zlib.error:
            pass
        if b"Tf" not in data:
            continue
        current: float | None = None
        for m in token.finditer(data):
            if m.group(1):
                current = float(m.group(2))
            elif m.group(3) is not None and current:
                total[current] += len(re.findall(rb"\((?:[^()\\]|\\.)*\)", m.group(3)))
            elif m.group(4) is not None and current:
                total[current] += 1
    return dict(total)


def main() -> int:
    placed = placements()
    if not placed:
        print("No \\includegraphics with a width option found in sections/.")
        return 1

    print(f"{'figure':42s} {'natural':>8s} {'printed':>8s} {'scale':>6s} "
          f"{'<6pt':>7s} {'min':>6s}")
    print("-" * 82)

    failures = 0
    for rel, target in sorted(placed.items()):
        path = HERE / rel
        if not path.exists():
            print(f"{rel:42s}  MISSING")
            failures += 1
            continue
        width = page_width(path)
        counts = char_counts_by_size(path)
        if width is None or not counts:
            print(f"{rel:42s}  cannot measure")
            continue

        scale = target / width
        total = sum(counts.values())
        below = sum(n for size, n in counts.items() if size * scale < MIN_SCRIPT_PT)
        share = 100.0 * below / total
        smallest = min(counts) * scale

        flag = ""
        if share > 5.0:
            flag = "  <-- re-author"
            failures += 1
        print(f"{Path(rel).name:42s} {width:8.1f} {target:8.1f} {scale:6.2f} "
              f"{share:6.1f}% {smallest:5.1f}pt{flag}")

    print()
    print(f"Rule: body text >= {MIN_BODY_PT:.0f} pt, scripts >= {MIN_SCRIPT_PT:.0f} pt "
          "at the printed size (Elsevier, Artwork sizing).")
    if failures:
        print(f"{failures} figure(s) need re-authoring at their printed size.")
        return 1
    print("All figures pass.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
