"""Export recurrent source-inversion fields as database-friendly text files."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from field import recurrent_plume_fields_at_times
from geo_utils import xy_to_latlon


OUTPUT_FOLDER_NAME = "溯源输出"
FIELD_FOLDER_NAME = "浓度场"
SOURCE_LOCATION_FILENAME = "污染源点坐标.txt"
SOURCE_STRENGTH_FILENAME = "源强.txt"


def _hourly_timestamps(time_labels):
    timestamps = pd.to_datetime(pd.Series(time_labels), errors="raise")
    start = timestamps.min().ceil("h")
    end = timestamps.max().floor("h")
    if start > end:
        return pd.DatetimeIndex([])
    return pd.date_range(start=start, end=end, freq="h")


def _output_directories(output_dir, result_root_dir):
    output_dir = Path(output_dir)
    case_name = output_dir.name
    directories = [
        output_dir / OUTPUT_FOLDER_NAME,
        Path(result_root_dir) / OUTPUT_FOLDER_NAME / case_name,
    ]
    unique_directories = []
    for directory in directories:
        if directory not in unique_directories:
            unique_directories.append(directory)
    return unique_directories


def _write_tabular_txt(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(values).to_csv(
        path,
        sep="\t",
        header=False,
        index=False,
        encoding="utf-8",
        float_format="%.8f",
    )


def export_hourly_concentration_text_outputs(
    *,
    model,
    output_dir,
    result_root_dir,
    time_labels,
    t_w,
    u_w,
    v_w,
    baseline_w,
    lon0,
    lat0,
    x0,
    y0,
    length_m,
    duration_hours,
    c_scale,
    sigma_src,
    source_lon,
    source_lat,
):
    """Save hourly concentration fields, source point, and source strength.

    Concentration fields use the same quantity displayed by ``diffusion.gif``:
    recurrent plume concentration scaled to the observation unit plus the
    time-interpolated baseline. Each TXT row is ``longitude, latitude,
    concentration`` without a header. The source TXT contains one
    ``longitude, latitude`` row without a header. The source-strength TXT
    contains ``time, Q`` rows without a header; Q is the model source strength.
    """
    hourly_times = _hourly_timestamps(time_labels)
    output_directories = _output_directories(output_dir, result_root_dir)
    for directory in output_directories:
        _write_tabular_txt(
            directory / SOURCE_LOCATION_FILENAME,
            {"longitude": [float(source_lon)], "latitude": [float(source_lat)]},
        )

    if hourly_times.empty:
        for directory in output_directories:
            _write_tabular_txt(
                directory / SOURCE_STRENGTH_FILENAME,
                {"time": [], "source_strength": []},
            )
        return {
            "hourly_field_count": 0,
            "output_directories": [str(directory) for directory in output_directories],
            "source_paths": [
                str(directory / SOURCE_LOCATION_FILENAME)
                for directory in output_directories
            ],
            "source_strength_paths": [
                str(directory / SOURCE_STRENGTH_FILENAME)
                for directory in output_directories
            ],
            "field_paths": [],
        }

    reference_time = pd.to_datetime(time_labels[0])
    normalized_times = (
        (hourly_times - reference_time).total_seconds().to_numpy(dtype=np.float64)
        / 3600.0
        / max(float(duration_hours), 1e-12)
    )
    t_w = np.asarray(t_w, dtype=np.float64)
    u_w = np.asarray(u_w, dtype=np.float64)
    v_w = np.asarray(v_w, dtype=np.float64)
    baseline_w = np.asarray(baseline_w, dtype=np.float64)
    u_hourly = np.interp(normalized_times, t_w, u_w)
    v_hourly = np.interp(normalized_times, t_w, v_w)
    baseline_hourly = np.interp(normalized_times, t_w, baseline_w)

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            device = next(model.parameters()).device
            hourly_t_tensor = torch.as_tensor(
                normalized_times,
                dtype=torch.float32,
                device=device,
            ).view(-1, 1)
            source_strength = model.Q(hourly_t_tensor).detach().cpu().numpy().reshape(-1)
            fields = recurrent_plume_fields_at_times(
                model,
                sigma_src=sigma_src,
                t_values=normalized_times,
                u_values=u_hourly,
                v_values=v_hourly,
            )
    finally:
        if was_training:
            model.train()

    source_strength_values = {
        "time": [timestamp.strftime("%Y-%m-%d %H:%M:%S") for timestamp in hourly_times],
        "source_strength": source_strength,
    }
    for directory in output_directories:
        _write_tabular_txt(
            directory / SOURCE_STRENGTH_FILENAME,
            source_strength_values,
        )

    x_norm = model.recurrent_x_grid.detach().cpu().numpy()
    y_norm = model.recurrent_y_grid.detach().cpu().numpy()
    x_m, y_m = np.meshgrid(x_norm * float(length_m) + float(x0), y_norm * float(length_m) + float(y0))
    lon_grid, lat_grid = xy_to_latlon(x_m, y_m, float(lon0), float(lat0))
    longitude = lon_grid.reshape(-1)
    latitude = lat_grid.reshape(-1)

    all_field_paths = []
    for timestamp, field, baseline in zip(hourly_times, fields, baseline_hourly):
        concentration = field.detach().cpu().numpy() * float(c_scale) + float(baseline)
        concentration = np.clip(concentration, a_min=0.0, a_max=None)
        values = {
            "longitude": longitude,
            "latitude": latitude,
            "concentration": concentration.reshape(-1),
        }
        filename = f"浓度场_{timestamp.strftime('%Y%m%d_h%H')}.txt"
        field_paths = []
        for directory in output_directories:
            path = directory / FIELD_FOLDER_NAME / filename
            _write_tabular_txt(path, values)
            field_paths.append(str(path))
        all_field_paths.append(field_paths)

    return {
        "hourly_field_count": len(hourly_times),
        "output_directories": [str(directory) for directory in output_directories],
        "source_paths": [
            str(directory / SOURCE_LOCATION_FILENAME) for directory in output_directories
        ],
        "source_strength_paths": [
            str(directory / SOURCE_STRENGTH_FILENAME)
            for directory in output_directories
        ],
        "field_paths": all_field_paths,
    }
