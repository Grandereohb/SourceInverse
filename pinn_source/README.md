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
  `RECURRENT_SUBSTEPS`, `RECURRENT_ADAPTIVE_SUBSTEPS`,
  `RECURRENT_MAX_ADVECTION_CELLS`, `RECURRENT_MAX_SUBSTEPS`, and
  `RECURRENT_INITIAL_RELEASE_FRACTION`

## Inversion Text Outputs

Every recurrent-PDE run exports database-friendly TXT files twice: under the
case result directory and under `<OUTPUT_DIR>/溯源输出/<case-name>/`.

- `污染源点坐标.txt`: one tab-separated `longitude, latitude` row.
- `源强.txt`: one tab-separated `time, model source strength` row per whole hour.
- `浓度场/浓度场_YYYYMMDD_hHH.txt`: one row per recurrent grid cell with
  tab-separated `longitude, latitude, predicted_total_concentration` values.

Hourly fields are evaluated by the recurrent transport model at each whole
clock hour in the event window. Concentration is plume contribution plus the
time-interpolated baseline, matching the quantity displayed in `diffusion.gif`.

## Consolidated Diagnostics

Each case writes one `diagnostics.xlsx` workbook instead of separate diagnostic
CSV files. Depending on the enabled model features, it contains these sheets:

- `q_time_series`
- `q_segments`
- `training`
- `station_peaks`
- `source_landscape`

The complete source-landscape metadata is embedded under `source_landscape` in
`result_quality_report.json`. When landscape confidence is enabled, the single
`source_confidence.png` combines stations, the trained source, the landscape
best point, relative-probability regions, confidence contours, and loss
contours.

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
- Output includes copied input Excel files, source location, consolidated
  diagnostics, station time-series plots, one source-confidence plot, and
  `diffusion.gif`.

## Model Selection

Set the model in `pinn_source/config.py`:

```python
MODEL_NAME = "pinn"
```

Add new models under `pinn_source/models/` and register them in `model_registry.py`.
