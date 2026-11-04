#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
export SOURCE_DATE_EPOCH=1793750400
export FORCE_SOURCE_DATE=1
pandoc manuscript.md --standalone --number-sections -t latex \
  -V mainfont='DejaVu Serif' -V sansfont='DejaVu Sans' \
  -V monofont='DejaVu Sans Mono' -V geometry:margin=24mm \
  -V fontsize=10pt -V colorlinks=true -o manuscript.tex
xelatex -interaction=nonstopmode -halt-on-error manuscript.tex > build-pass1.log
xelatex -interaction=nonstopmode -halt-on-error manuscript.tex > build-pass2.log
