#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
pandoc manuscript.md --standalone --number-sections -t latex -o manuscript.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=. manuscript.tex > build-pass1.log
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=. manuscript.tex > build-pass2.log
