# The Calibration Turn in AI-Assisted Research

Public source repository for the preprint:

**The Calibration Turn in AI-Assisted Research: A Conceptual and Methodological Framework for Evidence-Licensed Claims**

The repository supports manuscript inspection and reproduction of the AISim-Cal synthetic simulation figures and tables. AISim-Cal is an illustrative methodological diagnostic, not an empirical forecast.

## Contents

- `ai_science_epistemology_en.tex`: manuscript source.
- `ai_science_epistemology_en.pdf`: compiled PDF.
- `references.bib`: bibliography.
- `simulation/`: AISim-Cal code and generated output tables.
- `figures/`: generated manuscript figures.
- `tests/`: regression tests.

## Reproduce

```bash
python3 -m pip install -r requirements.txt
python3 -m pytest -q
python3 -m simulation.run_route_simulation
latexmk -xelatex -interaction=nonstopmode ai_science_epistemology_en.tex
```

See `REPRODUCIBILITY.md` for the expected outputs.

## Boundary

This repository does not redistribute third-party article PDFs, publisher HTML snapshots, or supplementary files. External sources should be accessed through the papers, official pages, and news reports cited in `references.bib`.
