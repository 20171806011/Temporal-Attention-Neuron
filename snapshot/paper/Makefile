# TAN-I manuscript build
.PHONY: all clean

all: main.pdf

main.pdf: main.tex sections/*.tex supplementary/*.tex tables/*.tex \
          references.bib figures/*.pdf
	latexmk -pdf -interaction=nonstopmode main.tex

clean:
	latexmk -c
	rm -f main.bbl main.blg main.run.xml
