all: figs pdf

figs:
	python3 figs/make_figs.py

pdf:
	latexmk -pdf -interaction=nonstopmode main.tex

clean:
	latexmk -C main.tex

.PHONY: all figs pdf clean
