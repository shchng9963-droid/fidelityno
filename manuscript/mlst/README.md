# FidelityFormer MLST submission source

This directory contains the source files for the manuscript submitted to
*Machine Learning: Science and Technology*.

## Files

- main.tex: main manuscript using the 12 pt iopart class
- supplementary.tex: independently compiled supplementary material
- bibliography.bib: BibTeX database
- figures/: figures cited by the manuscript and supplement
- iopart.cls, iopart12.clo, iopams.sty, iopart-num.bst: IOP class and
  bibliography files needed for a self-contained source build

The cover letter is maintained separately as cover_letter_MLST.md.

## Compile the main manuscript

Run the following commands from this directory:

    pdflatex -interaction=nonstopmode main.tex
    bibtex main
    pdflatex -interaction=nonstopmode main.tex
    pdflatex -interaction=nonstopmode main.tex

## Compile the supplementary material

    pdflatex -interaction=nonstopmode supplementary.tex
    pdflatex -interaction=nonstopmode supplementary.tex

Both PDFs were compile-checked from a clean copy of the source bundle before
packaging.
