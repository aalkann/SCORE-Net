#!/usr/bin/env bash
# Build the Neurocomputing manuscript and verify that the result is clean.
#
#   ./build.sh            build + verify
#   ./build.sh --fast     build only (no verification of the log)
#
# Requires: latexmk, pdflatex, bibtex (TeX Live).
set -uo pipefail

cd "$(dirname "$0")"

echo "==> [1/3] deriving table copies from the canonical IEEE tables"
python3 prepare_tables.py || exit 1

echo "==> [2/3] building main.pdf"
latexmk -C >/dev/null 2>&1
if ! latexmk -pdf -interaction=nonstopmode main.tex > build.log 2>&1; then
    echo "BUILD FAILED — see build.log"
    grep -E "^! " build.log | head -20
    exit 1
fi

echo "==> [2b/3] building supplementary.pdf"
latexmk -pdf -g -interaction=nonstopmode supplementary.tex > suppbuild.log 2>&1 || {
    echo "SUPPLEMENTARY BUILD FAILED — see suppbuild.log"
    grep -E "^! " suppbuild.log | head -20
    exit 1
}
supp_pages=$(grep -oE "Output written on supplementary.pdf \([0-9]+ page" suppbuild.log | grep -oE "[0-9]+" | tail -1 || true)
supp_err=$(grep -cE "^! " supplementary.log || true)
supp_over=$(grep -acE "Overfull \\\\(hbox|vbox)" supplementary.log || true)
printf '    supplement : %s pages, %s errors, %s overfull\n' "${supp_pages:-?}" "$supp_err" "$supp_over"

if [[ "${1:-}" == "--fast" ]]; then
    echo "OK (fast mode, log not verified)"
    exit 0
fi

echo "==> [3/3] verifying the build log"
status=0

errors=$(grep -cE "^! " main.log || true)
overfull=$(grep -acE "Overfull \\\\(hbox|vbox)" main.log || true)
undefined=$(grep -aci "undefined" main.log || true)
pages=$(grep -oE "Output written on main.pdf \([0-9]+ page" main.log | grep -oE "[0-9]+" | tail -1)

printf '    pages      : %s\n' "${pages:-?}"
printf '    errors     : %s\n' "$errors"
printf '    overfull   : %s\n' "$overfull"
printf '    undefined  : %s\n' "$undefined"

if [[ "$errors" -ne 0 || "$overfull" -ne 0 || "$undefined" -ne 0 ]]; then
    echo "CHECK FAILED"
    status=1
else
    echo "CLEAN BUILD"
fi

echo "==> abstract word count (limit 250)"
python3 - <<'PY'
import re
t = open("sections/abstract.tex").read()
t = t.replace("\\modelname{}", "SCORE-Net")
t = re.sub(r"\\emph\{([^}]*)\}", r"\1", t)
t = re.sub(r"\$[^$]*\$", "L1", t)
t = re.sub(r"\\[a-zA-Z]+", "", t).replace("{", "").replace("}", "")
t = t.replace("---", "-").replace("--", "-")
n = len(t.split())
print(f"    words      : {n}  ({'OK' if n <= 250 else 'TOO LONG'})")
PY

# --- Submission gate -------------------------------------------------------
# The Data availability statement tells readers the code is openly available.
# That claim is false while the repository is private, and a reviewer will not
# be able to verify anything. Warn (without failing the build) if the link is
# not reachable from the public internet.
echo "==> numbers quoted in the text vs results/*.csv"
# Pick an interpreter that actually has pandas; the system python3 often does not.
NUMCHECK_PY=""
for candidate in "${SCORENET_PYTHON:-}" python3 python; do
    [[ -z "$candidate" ]] && continue
    if command -v "$candidate" >/dev/null 2>&1 && \
       "$candidate" -c "import pandas" >/dev/null 2>&1; then
        NUMCHECK_PY="$candidate"
        break
    fi
done
if [[ -n "$NUMCHECK_PY" ]]; then
    if "$NUMCHECK_PY" tools/check_manuscript_numbers.py; then
        :
    else
        status=1
    fi
else
    echo "    [SKIP] pandas bulunamadi; SCORENET_PYTHON=/yol/python ile calistirin"
fi

echo "==> submission gates"
gates=0
if grep -q "TODO" sections/declarations.tex 2>/dev/null; then
    echo "    [WARN] declarations.tex still contains TODO placeholders"
    gates=1
fi
repo=$(grep -oE 'https://github\.com/[^}[:space:]]+' sections/declarations.tex | head -1)
if [[ -n "$repo" ]]; then
    if command -v curl >/dev/null 2>&1; then
        code=$(curl -s -o /dev/null -w '%{http_code}' -L --max-time 15 "$repo" 2>/dev/null || echo "000")
        if [[ "$code" == "200" ]]; then
            echo "    repository : public (HTTP $code)"
        else
            echo "    repository : NOT publicly reachable (HTTP $code) -> $repo"
            echo "                 make it public before uploading, or the Data"
            echo "                 availability statement is false."
            gates=1
        fi
    fi
fi
if [[ "$gates" -eq 0 ]]; then
    echo "    all clear"
fi

exit "$status"
