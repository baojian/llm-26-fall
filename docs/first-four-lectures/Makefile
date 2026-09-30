LATEXMK ?= latexmk
OUTDIR := ../../output/pdf

.PHONY: outline
outline:
	mkdir -p $(OUTDIR)
	$(LATEXMK) -silent -pdf -interaction=nonstopmode -halt-on-error -outdir=$(OUTDIR) outline.tex
