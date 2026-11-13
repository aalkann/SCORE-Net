#!/usr/bin/env python3
"""Validate every notebook in the repository.

Checks that a reviewer would notice immediately if broken:

* the file is valid JSON in nbformat 4;
* every ``source`` field is a list of lines, not a character-split string (the
  bug that turned the notebooks into one letter per line);
* every line except the last ends with a newline, otherwise the cell renders as
  one long unwrapped line;
* every code cell compiles as Python, and no cell still points at a directory
  that was removed in the restructure.

Usage
-----
    python scripts/check_notebooks.py            # check
    python scripts/check_notebooks.py --fix-newlines   # repair missing newlines

Exit code 1 when a notebook fails, so this can gate a commit or a CI job.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Legacy Turkish directory names from the pre-restructure layout.
STALE_PATHS = (
    "Model Analiz Örnekleri",
    "Model Ablasyon Analizi",
    "Diğer Modellerin Analiz Örnekleri",
    "Eklenmesi Gerekenler/",
)


def notebooks() -> list[Path]:
    return sorted(p for p in REPO.rglob("*.ipynb") if ".ipynb_checkpoints" not in p.parts)


def text_of(cell: dict) -> str:
    src = cell.get("source", "")
    return src if isinstance(src, str) else "".join(src)


def check(path: Path, fix_newlines: bool) -> list[str]:
    problems: list[str] = []
    try:
        nb = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return [f"invalid JSON: {exc}"]

    if nb.get("nbformat") != 4:
        problems.append(f"nbformat is {nb.get('nbformat')}, expected 4")

    changed = False
    for index, cell in enumerate(nb.get("cells", [])):
        source = cell.get("source", "")
        if not isinstance(source, list):
            problems.append(f"cell {index}: source is {type(source).__name__}, expected list")
            continue

        if len(source) > 20:
            singles = sum(1 for line in source if len(line) == 1 and line != "\n")
            if singles / len(source) > 0.3:
                problems.append(
                    f"cell {index}: character-split ({singles}/{len(source)} single chars)"
                )

        missing = [i for i, line in enumerate(source[:-1]) if not line.endswith("\n")]
        if missing:
            if fix_newlines:
                for i in missing:
                    source[i] = source[i] + "\n"
                changed = True
            else:
                problems.append(f"cell {index}: {len(missing)} line(s) without trailing newline")

        if cell.get("cell_type") == "code":
            try:
                compile(text_of(cell), f"{path.name}:cell{index}", "exec")
            except SyntaxError as exc:
                problems.append(f"cell {index}: does not compile ({exc.msg})")

        blob = text_of(cell)
        for stale in STALE_PATHS:
            if stale in blob:
                problems.append(f"cell {index}: stale path '{stale}'")

    if fix_newlines and changed:
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
        problems.append("(newlines repaired)")

    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fix-newlines", action="store_true")
    args = ap.parse_args()

    files = notebooks()
    failures = 0
    for path in files:
        rel = path.relative_to(REPO)
        problems = check(path, args.fix_newlines)
        if problems:
            failures += 1
            print(f"FAIL  {rel}")
            for problem in problems:
                print(f"      {problem}")
        else:
            print(f"OK    {rel}")

    print()
    if failures:
        print(f"{failures} of {len(files)} notebooks failed.")
        return 1
    print(f"All {len(files)} notebooks are well formed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
