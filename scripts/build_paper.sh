#!/usr/bin/env bash
# Compile docs/paper/main.tex with latexmk inside the official TeX Live image (no local TeX needed).
# Usage: scripts/build_paper.sh   -> docs/paper/main.pdf ; prints LaTeX errors/warnings summary
set -euo pipefail
cd "$(dirname "$0")/.."
docker run --rm -v "$PWD":/w -w /w/docs/paper texlive/texlive:latest \
  latexmk -pdf -interaction=nonstopmode main.tex > /tmp/build_paper.out 2>&1 && status=0 || status=$?
grep -E "^!|LaTeX Warning|Overfull" docs/paper/main.log | sed 's/ on input line.*//' | sort | uniq -c | sort -rn || true
echo "latexmk exit $status -> docs/paper/main.pdf"
exit $status
