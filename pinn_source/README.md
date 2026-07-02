# PINN Source Inversion

This folder contains a modularized PINN-based source inversion pipeline. The entrypoint remains a single command.

## Quick Start

Run from the project root:

```bash
python pinn_source/pinn_source_pinn.py
```

## Configuration

Edit paths and hyperparameters in:

- `pinn_source/config.py`

Key settings:
- `SITE_PATH`, `CONC_PATH`, `WIND_PATH`
- `WIND_DIR_IS_FROM`
- `FIELD_MODE`
- `EPOCHS`, `LR`
- Recurrent plume controls such as `RECURRENT_GRID_NX`,
  `RECURRENT_SUBSTEPS`, and `RECURRENT_INITIAL_RELEASE_FRACTION`

## Module Layout

- `config.py` - Paths and hyperparameters
- `geo_utils.py` - DMS parsing, lat/lon conversions
- `data_io.py` - Data loading and wind conversion
- `models/` - PINN model definitions
- `pipeline.py` - Training, filtering, and inference pipeline
- `viz.py` - Plotting and diffusion animation
- `pinn_source_pinn.py` - One-click entrypoint

## Notes

- Wind direction uses meteorological convention when `WIND_DIR_IS_FROM = True`.
- Zero concentration is treated as valid data; rows are dropped only when required
  wind or station values are missing.
- The default `FIELD_MODE = "recurrent_pde"` recursively advances a gridded plume
  field through the observed wind sequence before sampling concentrations at
  station locations.
- Output includes copied input Excel files, source location, diagnostics,
  station time-series plots, confidence plots, and `diffusion.gif`.

## Model Selection

Set the model in `pinn_source/config.py`:

```python
MODEL_NAME = "pinn"
```

Add new models under `pinn_source/models/` and register them in `model_registry.py`.
