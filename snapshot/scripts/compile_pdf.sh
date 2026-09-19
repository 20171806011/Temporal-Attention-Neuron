#!/bin/bash
# Compile the TAN manuscript PDF
# Usage: bash scripts/compile_pdf.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MANUSCRIPT_DIR="$(dirname "$SCRIPT_DIR")/manuscript"

cd "$MANUSCRIPT_DIR"

echo "Compiling manuscript..."

# Try tectonic first (handles bibliography automatically)
if command -v tectonic &> /dev/null; then
    echo "Using Tectonic..."
    tectonic main.tex
# Fallback to pdflatex + bibtex
elif command -v pdflatex &> /dev/null; then
    echo "Using pdflatex + bibtex..."
    pdflatex main.tex
    bibtex main
    pdflatex main.tex
    pdflatex main.tex
else
    echo "ERROR: Neither tectonic nor pdflatex found."
    echo "Install tectonic: https://tectonic-typesetting.github.io/"
    exit 1
fi

echo ""
echo "PDF generated: $MANUSCRIPT_DIR/main.pdf"
echo "Pages: $(python3 -c 'from pypdf import PdfReader; print(len(PdfReader("'"'"'$MANUSCRIPT_DIR/main.pdf'"'"'").pages))' 2>/dev/null || echo 'unknown')"
