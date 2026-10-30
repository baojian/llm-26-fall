ROOT := ../..
OUT := $(ROOT)/output/pdf
NAME := lecture-07-scaling-laws-board-notes

.PHONY: all
all:
	mkdir -p $(OUT)
	latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=$(OUT) -jobname=$(NAME) board-notes.tex
