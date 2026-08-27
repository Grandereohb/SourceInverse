"""Reconstruct a saved event and compare recurrent transport solvers.

The script is intentionally inference-only: it reads the copied inputs and
diagnostics in a result directory, restores the learned scalar/time-series
parameters, and performs no optimization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
PINN_DIR = ROOT / "pinn_source"
if str(PINN_DIR) not in sys.path:
    sys.path.insert(0, str(PINN_DIR))

import field as production  # noqa: E402
from config import (  # noqa: E402
    BASELINE_MODE,
    ENABLE_EVENT_WINDOW_CROP,
    EVENT_WINDOW_MIN_MAX,
    EVENT_WINDOW_MIN_RELIEF,
    EVENT_WINDOW_PAD_STEPS,
    Q_MAX,
    Q_MIN,
    SIGMA_SRC,
)
from data_io import load_conc, load_sites, load_wind  # noqa: E402
from models.pinn import PINN  # noqa: E402
from pipeline import _apply_wind_vector_smoothing  # noqa: E402
from research_recurrent import recurrent_plume_fields_characteristic  # noqa: E402
from transport_units import normalize_velocity_mps  # noqa: E402


def _inverse_softplus(value: float) -> float:
    value = max(float(value), 1e-12)
    return value + math.log(-math.expm1(-value))


def _baseline(frame: pd.DataFrame, stations: list[str]) -> pd.Series:
    values = frame[stations].astype(float)
    if BASELINE_MODE == "q25":
        return values.quantile(0.25, axis=1)
    if BASELINE_MODE == "q40":
        return values.quantile(0.40, axis=1)
    return values.median(axis=1)


def _crop_event(frame: pd.DataFrame, stations: list[str]) -> pd.DataFrame:
    if not ENABLE_EVENT_WINDOW_CROP:
        return frame.reset_index(drop=True)
    baseline = _baseline(frame, stations).to_numpy(dtype=np.float64)
    residual = np.clip(
        frame[stations].to_numpy(dtype=np.float64) - baseline[:, None],
        a_min=0.0,
        a_max=None,
    )
    maxima = residual.max(axis=1)
    medians = np.median(residual, axis=1)
    relief = (maxima - medians) / np.maximum(np.abs(maxima), 1e-6)
    indices = np.flatnonzero(
        (maxima >= EVENT_WINDOW_MIN_MAX) & (relief >= EVENT_WINDOW_MIN_RELIEF)
    )
    if indices.size == 0:
        return frame.reset_index(drop=True)
    start = max(0, int(indices[0]) - int(EVENT_WINDOW_PAD_STEPS))
    end = min(len(frame) - 1, int(indices[-1]) + int(EVENT_WINDOW_PAD_STEPS))
    return frame.iloc[start : end + 1].reset_index(drop=True)


def _sample_fields(
    fields: torch.Tensor,
    x_grid: torch.Tensor,
    y_grid: torch.Tensor,
    x_query: np.ndarray,
    y_query: np.ndarray,
) -> np.ndarray:
    values = []
    x_tensor = torch.as_tensor(x_query, dtype=fields.dtype).view(-1, 1)
    y_tensor = torch.as_tensor(y_query, dtype=fields.dtype).view(-1, 1)
    for layer in fields:
        sampled = production._sample_grid_bilinear(
            layer, x_tensor, y_tensor, x_grid, y_grid
        )
        values.append(sampled.detach().cpu().numpy().reshape(-1))
    return np.stack(values, axis=0)


def _relative_l2(left: np.ndarray, right: np.ndarray) -> float:
    denom = max(float(np.linalg.norm(right.reshape(-1))), 1e-12)
    return float(np.linalg.norm((left - right).reshape(-1)) / denom)


def _load_exported_fields(event_dir: Path, hourly_times: pd.DatetimeIndex) -> np.ndarray:
    values = []
    field_dir = event_dir / "溯源输出" / "浓度场"
    for timestamp in hourly_times:
        path = field_dir / f"浓度场_{timestamp.strftime('%Y%m%d_h%H')}.txt"
        array = np.loadtxt(path, delimiter="\t")
        values.append(array[:, 2])
    return np.stack(values, axis=0)


def compare_event(event_dir: Path) -> dict:
    report_path = event_dir / "result_quality_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    recurrent = report["recurrent_pde"]
    model_report = report["model"]
    source_report = report["source"]

    sites, _, _ = load_sites(event_dir / "sites.xlsx")
    concentration = load_conc(event_dir / "concentration.xlsx")
    wind = load_wind(event_dir / "wind.xlsx")
    selected = list(report["station_ablation"]["selected_stations"])
    data = concentration.merge(wind, on="time", how="inner")
    data = data[["time", "dir", "sp"] + selected]
    data = data.dropna(subset=["dir", "sp"] + selected).copy()
    data = _crop_event(data, selected)
    data = _apply_wind_vector_smoothing(data)
    baseline = _baseline(data, selected).to_numpy(dtype=np.float64)

    q_workbook = pd.read_excel(event_dir / "diagnostics.xlsx", sheet_name="q_time_series")
    training = pd.read_excel(event_dir / "diagnostics.xlsx", sheet_name="training")
    q_times = pd.to_datetime(q_workbook["time"])
    duration_hours = float((q_times.iloc[-1] - q_times.iloc[0]).total_seconds() / 3600.0)
    t_train = (
        (q_times - q_times.iloc[0]).dt.total_seconds().to_numpy(dtype=np.float64)
        / 3600.0
        / duration_hours
    )

    data = data.set_index("time").loc[q_times].reset_index()
    baseline = _baseline(data, selected).to_numpy(dtype=np.float64)
    u_mps = data["u_eff"].to_numpy(dtype=np.float64)
    v_mps = data["v_eff"].to_numpy(dtype=np.float64)
    wind_factor = float(recurrent["wind_factor"])
    x_min_m = float(source_report["domain_x_min_m"])
    x_max_m = float(source_report["domain_x_max_m"])
    y_min_m = float(source_report["domain_y_min_m"])
    y_max_m = float(source_report["domain_y_max_m"])
    length_m = x_max_m - x_min_m
    x_center_m = 0.5 * (x_min_m + x_max_m)
    y_center_m = 0.5 * (y_min_m + y_max_m)
    u_train = normalize_velocity_mps(u_mps, duration_hours, length_m, wind_factor)
    v_train = normalize_velocity_mps(v_mps, duration_hours, length_m, wind_factor)

    model = PINN().to(torch.device("cpu"))
    model.set_q_bounds(Q_MIN, Q_MAX)
    model.configure_smooth_time_q(t_train)
    q_values = q_workbook["Q"].to_numpy(dtype=np.float64).copy()
    d_total_norm = float(training["D_norm"].iloc[-1])
    d_min_norm = float(recurrent["d_min_norm"])
    d_scale_norm = float(recurrent["d_scale_norm"])
    learned_d_multiplier = (d_total_norm - d_min_norm) / d_scale_norm
    with torch.no_grad():
        model.xs.fill_((float(source_report["x_m"]) - x_center_m) / length_m)
        model.ys.fill_((float(source_report["y_m"]) - y_center_m) / length_m)
        model.logQ.zero_()
        model.logQ_time.copy_(torch.log(torch.as_tensor(q_values, dtype=torch.float32)))
        model.rawD.fill_(_inverse_softplus(learned_d_multiplier - 1e-6))

    x_min = (x_min_m - x_center_m) / length_m
    x_max = (x_max_m - x_center_m) / length_m
    y_min = (y_min_m - y_center_m) / length_m
    y_max = (y_max_m - y_center_m) / length_m
    production.configure_recurrent_context(
        model=model,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        t_values=t_train,
        u_values=u_train,
        v_values=v_train,
        d_min_norm=d_min_norm,
        d_scale_norm=d_scale_norm,
        decay_norm=float(recurrent["decay_norm"]),
        nx=int(recurrent["grid_nx"]),
        ny=int(recurrent["grid_ny"]),
    )

    with torch.no_grad():
        restored_q = model.Q(torch.as_tensor(t_train, dtype=torch.float32).view(-1, 1))
        old_train_fields = production.recurrent_plume_fields(model, SIGMA_SRC)
        new_train_fields = recurrent_plume_fields_characteristic(model, SIGMA_SRC)

    station_rows = sites.set_index("station").loc[selected]
    x_station = (station_rows["x"].to_numpy(dtype=np.float64) - x_center_m) / length_m
    y_station = (station_rows["y"].to_numpy(dtype=np.float64) - y_center_m) / length_m
    old_station_plume = _sample_fields(
        old_train_fields, model.recurrent_x_grid, model.recurrent_y_grid, x_station, y_station
    )
    new_station_plume = _sample_fields(
        new_train_fields, model.recurrent_x_grid, model.recurrent_y_grid, x_station, y_station
    )
    observed = data[selected].to_numpy(dtype=np.float64)
    c_scale = float(model_report["c_scale"])
    old_station_raw = old_station_plume * c_scale + baseline[:, None]
    new_station_raw = new_station_plume * c_scale + baseline[:, None]

    source_strength_path = event_dir / "溯源输出" / "源强.txt"
    source_strength = pd.read_csv(
        source_strength_path,
        sep="\t",
        header=None,
        names=["time", "Q"],
        parse_dates=["time"],
    )
    hourly_times = pd.DatetimeIndex(source_strength["time"])
    t_hourly = (
        (hourly_times - q_times.iloc[0]).total_seconds().to_numpy(dtype=np.float64)
        / 3600.0
        / duration_hours
    )
    u_hourly = np.interp(t_hourly, t_train, u_train)
    v_hourly = np.interp(t_hourly, t_train, v_train)
    baseline_hourly = np.interp(t_hourly, t_train, baseline)
    with torch.no_grad():
        old_hourly_fields = production.recurrent_plume_fields_at_times(
            model, SIGMA_SRC, t_hourly, u_hourly, v_hourly
        )
    production.configure_recurrent_context(
        model=model,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        t_values=t_hourly,
        u_values=u_hourly,
        v_values=v_hourly,
        d_min_norm=d_min_norm,
        d_scale_norm=d_scale_norm,
        decay_norm=float(recurrent["decay_norm"]),
        initial_release_dt=float(t_train[1] - t_train[0]),
        nx=int(recurrent["grid_nx"]),
        ny=int(recurrent["grid_ny"]),
    )
    with torch.no_grad():
        new_hourly_fields = recurrent_plume_fields_characteristic(model, SIGMA_SRC)

    old_hourly = old_hourly_fields.detach().cpu().numpy().reshape(len(t_hourly), -1)
    new_hourly = new_hourly_fields.detach().cpu().numpy().reshape(len(t_hourly), -1)
    reconstructed_export = np.clip(
        old_hourly * c_scale + baseline_hourly[:, None], a_min=0.0, a_max=None
    )
    exported = _load_exported_fields(event_dir, hourly_times)

    first_initialization = source_report.get("initialization", {})
    wind_reference_error = math.hypot(
        float(u_mps[0]) - float(first_initialization.get("wind_u_mps", u_mps[0])),
        float(v_mps[0]) - float(first_initialization.get("wind_v_mps", v_mps[0])),
    )
    per_time_rel_l2 = [
        _relative_l2(new_hourly[i], old_hourly[i]) for i in range(len(t_hourly))
    ]
    export_per_time_rel_l2 = [
        _relative_l2(reconstructed_export[i], exported[i])
        for i in range(len(t_hourly))
    ]
    export_per_time_max_abs = [
        float(np.max(np.abs(reconstructed_export[i] - exported[i])))
        for i in range(len(t_hourly))
    ]
    output = {
        "event_dir": str(event_dir.resolve()),
        "mode": "fixed_reconstructed_parameters_no_retraining",
        "reconstruction_checks": {
            "training_time_count": int(len(t_train)),
            "hourly_export_time_count": int(len(t_hourly)),
            "dropped_hourly_times": [
                str(t) for t in hourly_times if t not in set(pd.DatetimeIndex(q_times))
            ],
            "q_grid_max_abs_error": float(
                np.max(np.abs(restored_q.detach().cpu().numpy().reshape(-1) - q_values))
            ),
            "source_x_norm": float(model.xs.detach()),
            "source_y_norm": float(model.ys.detach()),
            "learned_diffusivity_m2s": float(1.0 + learned_d_multiplier),
            "first_smoothed_wind_vector_error_mps": float(wind_reference_error),
            "export_field_max_abs_error_raw": float(
                np.max(np.abs(reconstructed_export - exported))
            ),
            "export_field_relative_l2_raw": _relative_l2(reconstructed_export, exported),
            "export_field_error_by_time": [
                {
                    "time": str(timestamp),
                    "relative_l2_raw": rel_l2,
                    "max_abs_error_raw": max_abs,
                }
                for timestamp, rel_l2, max_abs in zip(
                    hourly_times, export_per_time_rel_l2, export_per_time_max_abs
                )
            ],
            "reported_fit_raw_rmse": float(report["fit_raw_rmse"]),
            "reconstructed_fit_raw_rmse": float(np.sqrt(np.mean((old_station_raw - observed) ** 2))),
        },
        "solver_comparison": {
            "old_station_rmse_raw": float(np.sqrt(np.mean((old_station_raw - observed) ** 2))),
            "new_station_rmse_raw": float(np.sqrt(np.mean((new_station_raw - observed) ** 2))),
            "old_station_mae_raw": float(np.mean(np.abs(old_station_raw - observed))),
            "new_station_mae_raw": float(np.mean(np.abs(new_station_raw - observed))),
            "train_grid_relative_l2_new_vs_old": _relative_l2(
                new_train_fields.detach().cpu().numpy(), old_train_fields.detach().cpu().numpy()
            ),
            "hourly_grid_relative_l2_new_vs_old": _relative_l2(new_hourly, old_hourly),
            "hourly_relative_l2_by_time": [
                {"time": str(timestamp), "relative_l2": value}
                for timestamp, value in zip(hourly_times, per_time_rel_l2)
            ],
            "new_source_panels_per_interval": list(
                getattr(model, "research_source_quadrature_panels_per_interval", ())
            ),
            "new_source_panel_cap_hit_count": int(
                getattr(model, "research_source_quadrature_cap_hit_count", 0)
            ),
        },
        "provenance_caveats": [
            "The result bundle contains no serialized model state_dict.",
            "The final restored diffusion parameter is not recorded; D is reconstructed from the last saved training diagnostic row, which need not be the restored best epoch.",
            "Wind smoothing and event-crop settings are reconstructed from the current code configuration and validated against saved outputs.",
            "The 03:00 Q value is an export-time interpolation because that row was dropped from training after a selected station had a missing value.",
        ],
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("event_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = compare_event(args.event_dir)
    payload = json.dumps(output, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        digest = hashlib.sha256(args.output.read_bytes()).hexdigest().upper()
        print(f"wrote={args.output.resolve()}")
        print(f"sha256={digest}")
    print(payload, end="")


if __name__ == "__main__":
    main()
