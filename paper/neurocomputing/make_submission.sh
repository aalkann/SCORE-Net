#!/usr/bin/env bash
# Build the flattened submission archive required by Editorial Manager.
#
# Editorial Manager cannot process subfolders: every .tex, .bbl, .bib, .cls and
# figure file must sit at the same level. This script copies the manuscript into
# a single flat folder, rewrites the \input and \includegraphics paths, compiles
# the flattened copy to prove it still builds, and zips the result.
#
#   ./make_submission.sh
#
# Output: submission/SCORE-Net_submission.zip
set -euo pipefail

cd "$(dirname "$0")"

FLAT="submission/flat"
OUT="submission/SCORE-Net_submission.zip"

echo "==> [1/5] preparing flat folder"
rm -rf "$FLAT" "$OUT"
mkdir -p "$FLAT"

python3 prepare_tables.py >/dev/null

# main.tex, sections, tables, figures, bibliography
cp main.tex "$FLAT/"
cp references.bib "$FLAT/" 2>/dev/null || true
cp sections/*.tex "$FLAT/"
cp generated/*.tex "$FLAT/"
cp figures/*.pdf "$FLAT/"

echo "==> [2/5] rewriting paths (sections/, generated/, figures/ -> .)"
python3 - "$FLAT" <<'PY'
import re, sys
from pathlib import Path

flat = Path(sys.argv[1])
for tex in flat.glob("*.tex"):
    text = tex.read_text()
    new = re.sub(r"\\input\{(?:sections|generated)/", r"\\input{", text)
    new = re.sub(r"\\includegraphics(\[[^\]]*\])?\{(?:figures)/", r"\\includegraphics\1{", new)
    if new != text:
        tex.write_text(new)
print("    paths rewritten")
PY

echo "==> [3/5] test-building the flattened copy"
( cd "$FLAT" \
  && latexmk -C >/dev/null 2>&1 \
  && if latexmk -pdf -interaction=nonstopmode main.tex > flatbuild.log 2>&1; then
         pages=$(grep -oE "Output written on main.pdf \([0-9]+ page" main.log | grep -oE "[0-9]+" | tail -1)
         echo "    flat build OK (${pages:-?} pages)"
     else
         echo "    FLAT BUILD FAILED — see $FLAT/flatbuild.log"
         grep -E "^! " flatbuild.log | head -10
         exit 1
     fi )

echo "==> [4/5] building the supplementary material"
latexmk -pdf -g -interaction=nonstopmode supplementary.tex > suppbuild.log 2>&1 || {
    echo "    SUPPLEMENTARY BUILD FAILED — see suppbuild.log"
    grep -E "^! " suppbuild.log | head -10
    exit 1
}
cp supplementary.pdf "submission/SCORE-Net_supplementary.pdf"
supp_pages=$(grep -oE "Output written on supplementary.pdf \([0-9]+ page" suppbuild.log | grep -oE "[0-9]+" | tail -1 || true)
echo "    supplementary OK (${supp_pages:-?} pages)"

echo "==> [5/5] zipping"
( cd "$FLAT" && zip -q -r "../$(basename "$OUT")" . -x '*.aux' '*.log' '*.out' '*.fls' '*.fdb_latexmk' '*.synctex.gz' 'main.pdf' )

# The proof PDF, the supplementary PDF and the non-source artifacts are
# uploaded separately.
cp "$FLAT/main.pdf" "submission/SCORE-Net_proof.pdf" 2>/dev/null || true

echo "    wrote $OUT"
echo
echo "Upload order (Editorial Manager):"
echo "  1 Cover Letter       submission/cover_letter.pdf"
echo "  2 Manuscript         main.tex, main.bbl, references.bib"
echo "  3 Table              tab_*.tex"
echo "  4 Figure             fig_*.pdf"
echo "  5 Highlights         submission/highlights.docx (or highlights.txt)"
echo "  6 Supplementary      submission/SCORE-Net_supplementary.pdf"
echo "  7 Declaration        Editorial Manager web form (no file is uploaded)"
