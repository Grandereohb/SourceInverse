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
- `EPOCHS`, `LR`, `N_COLLOCATION`

## Module Layout

- `config.py` - Paths and hyperparameters
- `geo_utils.py` - DMS parsing, lat/lon conversions
- `data_io.py` - Data loading and wind conversion
- `model.py` - PINN model definition
- `pipeline.py` - Training, filtering, and inference pipeline
- `viz.py` - Plotting and diffusion animation
- `pinn_source_pinn.py` - One-click entrypoint

## Notes

- Wind direction uses meteorological convention when `WIND_DIR_IS_FROM = True`.
- The pipeline filters out rows where `dir == 0` and any station concentration is 0/NaN.
- Output includes estimated source location and a diffusion animation saved as `diffusion.gif`.

## Model Selection

Set the model in `pinn_source/config.py`:

```python
MODEL_NAME = "pinn"
```

Add new models under `pinn_source/models/` and register them in `model_registry.py`.
