# Reproducibility

This repository supports two reproducible artifacts: the AISim-Cal synthetic outputs and the compiled manuscript PDF.

## Environment

Required:

- Python 3.11 or newer
- `latexmk` with XeLaTeX for PDF compilation

Python packages:

```bash
python3 -m pip install -r requirements.txt
```

## Run Checks

```bash
python3 -m pytest -q
```

Expected result:

```text
18 passed
```

## Regenerate AISim-Cal Outputs

```bash
python3 -m simulation.run_route_simulation
```

This regenerates:

- `simulation/outputs/*.csv`
- `simulation/outputs/simulation_config.json`
- `simulation/outputs/result_notes.md`
- `figures/*.png`
- `figures/domain_score_landscape.pdf`

The simulation is deterministic under the tracked random seed in `simulation/outputs/simulation_config.json`.

## Rebuild the Manuscript

```bash
latexmk -xelatex -interaction=nonstopmode ai_science_epistemology_en.tex
```

Expected output:

- `ai_science_epistemology_en.pdf`

## Interpretation Boundary

AISim-Cal outputs are synthetic diagnostics for inspecting the calibration framework. They are not empirical measurements, route rankings, or forecasts of scientific productivity.
