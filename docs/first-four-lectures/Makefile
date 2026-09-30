LATEXMK ?= latexmk
OUTDIR := ../../output/pdf

.PHONY: notes outline
notes:
	mkdir -p $(OUTDIR)
	$(LATEXMK) -silent -pdf -interaction=nonstopmode -halt-on-error -jobname=lectures-01-04-notes -outdir=$(OUTDIR) main.tex

outline:
	mkdir -p $(OUTDIR)
	$(LATEXMK) -silent -pdf -interaction=nonstopmode -halt-on-error -outdir=$(OUTDIR) outline.tex
