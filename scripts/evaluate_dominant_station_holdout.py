"""Evaluate dominant-station deletion drift and held-out field prediction."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PINN_DIR = ROOT / "pinn_source"
if str(PINN_DIR) not in sys.path:
    sys.path.insert(0, str(PINN_DIR))

from data_io import load_sites  # noqa: E402


FIELD_RE = re.compile(r"_(?P<date>\d{8})_h(?P<hour>\d{2})\.txt$")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _nearest_field_value(path: Path, lon: float, lat: float) -> float:
    values = np.loadtxt(path, delimiter="\t")
    distance2 = (values[:, 0] - lon) ** 2 + (values[:, 1] - lat) ** 2
    return float(values[int(np.argmin(distance2)), 2])


def evaluate_scenario(scenario_root: Path, sites_path: Path) -> dict:
    prep_path = scenario_root / "holdout_preparation_manifest.json"
    prep = json.loads(prep_path.read_text(encoding="utf-8"))
    run_manifests = sorted(
        (scenario_root / "refit_fixed_window_v2").rglob("run_manifest.json")
    )
    if len(run_manifests) != 1:
        raise ValueError(f"expected one holdout refit below {scenario_root}")
    run_path = run_manifests[0]
    run = json.loads(run_path.read_text(encoding="utf-8"))
    heldout = prep["heldout_station"]
    sites, _, _ = load_sites(sites_path)
    station = sites.loc[sites["station"] == heldout]
    if len(station) != 1:
        raise ValueError(f"held-out station coordinates not found: {heldout}")
    lon = float(station.iloc[0]["lon"])
    lat = float(station.iloc[0]["lat"])

    concentration = pd.read_csv(prep["concentration_path"])
    time_column = concentration.columns[0]
    concentration[time_column] = pd.to_datetime(concentration[time_column])
    station_columns = [
        column
        for column in concentration.columns[1:]
        if column != "TARGET_POLLUTANT"
    ]
    remaining = [column for column in station_columns if column != heldout]
    baseline = concentration[remaining].astype(float).median(axis=1)
    lookup = concentration.set_index(time_column)
    baseline_lookup = pd.Series(baseline.to_numpy(), index=concentration[time_column])

    records = []
    for field_path in sorted(run_path.parent.rglob("*_h*.txt")):
        match = FIELD_RE.search(field_path.name)
        if not match:
            continue
        timestamp = pd.Timestamp(
            f"{match.group('date')} {match.group('hour')}:00:00"
        )
        if timestamp not in lookup.index:
            continue
        observed = float(lookup.loc[timestamp, heldout])
        predicted = _nearest_field_value(field_path, lon, lat)
        base = float(baseline_lookup.loc[timestamp])
        records.append(
            {
                "time": str(timestamp),
                "observed_raw": observed,
                "predicted_raw": predicted,
                "remaining_station_median": base,
                "observed_positive_anomaly": max(observed - base, 0.0),
                "predicted_positive_anomaly": max(predicted - base, 0.0),
            }
        )
    if not records:
        raise ValueError(f"no held-out field predictions matched {scenario_root}")
    raw_errors = [row["predicted_raw"] - row["observed_raw"] for row in records]
    anomaly_errors = [
        row["predicted_positive_anomaly"] - row["observed_positive_anomaly"]
        for row in records
    ]
    dx = float(run["source"]["x_m"]) - float(prep["selected_source_x_m"])
    dy = float(run["source"]["y_m"]) - float(prep["selected_source_y_m"])
    observed_anomaly = [row["observed_positive_anomaly"] for row in records]
    return {
        "scenario_id": scenario_root.name,
        "heldout_station": heldout,
        "heldout_station_energy_ratio": float(prep["heldout_station_energy_ratio"]),
        "source_drift_m": math.hypot(dx, dy),
        "heldout_time_count": len(records),
        "heldout_raw_rmse": math.sqrt(float(np.mean(np.square(raw_errors)))),
        "heldout_anomaly_rmse": math.sqrt(
            float(np.mean(np.square(anomaly_errors)))
        ),
        "heldout_anomaly_scale": float(np.percentile(observed_anomaly, 95)),
        "heldout_anomaly_nrmse": math.sqrt(
            float(np.mean(np.square(anomaly_errors)))
        )
        / max(float(np.percentile(observed_anomaly, 95)), 1e-12),
        "original_source_x_m": float(prep["selected_source_x_m"]),
        "original_source_y_m": float(prep["selected_source_y_m"]),
        "refit_source_x_m": float(run["source"]["x_m"]),
        "refit_source_y_m": float(run["source"]["y_m"]),
        "refit_best_raw_loss": float(run["checkpoint"]["best_raw_loss"]),
        "preparation_manifest_sha256": _sha256(prep_path),
        "run_manifest_sha256": _sha256(run_path),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-root", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--scenario-ids", nargs="+", required=True)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    args = parser.parse_args()
    rows = [
        evaluate_scenario(
            args.pilot_root / scenario_id,
            args.data_root / scenario_id / "sites.xlsx",
        )
        for scenario_id in args.scenario_ids
    ]
    payload = {
        "schema_version": 1,
        "analysis": "dominant_station_holdout_pilot",
        "truth_isolation": "source truth is not read; labels are joined only in later development interpretation",
        "rows": rows,
    }
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    fields = [key for key in rows[0] if key != "records"]
    with args.output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})
    print(f"json_sha256={_sha256(args.output_json)}")
    print(f"csv_sha256={_sha256(args.output_csv)}")


if __name__ == "__main__":
    main()
