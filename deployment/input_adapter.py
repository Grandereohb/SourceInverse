from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pandas as pd

from .validation import TIME_FORMAT, validate_input


def load_and_validate_input(path: str | Path) -> dict:
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    return validate_input(payload)


def write_algorithm_inputs(payload: dict, input_dir: str | Path) -> dict[str, Path]:
    input_dir = Path(input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)

    sites_path = input_dir / "sites.xlsx"
    concentration_path = input_dir / "concentration.xlsx"
    wind_path = input_dir / "wind.xlsx"

    sites_df = pd.DataFrame(payload["stations"]).rename(
        columns={"station_id": "station", "longitude": "lon", "latitude": "lat"}
    )
    sites_df.to_excel(sites_path, index=False)

    concentration_long = pd.DataFrame(payload["concentrations"])
    concentration_df = (
        concentration_long.pivot(index="time", columns="station_id", values="value")
        .reindex(columns=sites_df["station"].tolist())
        .sort_index()
        .reset_index()
    )
    concentration_df.columns.name = None
    concentration_df.insert(
        len(concentration_df.columns), "TARGET_POLLUTANT", payload["pollutant"]
    )
    concentration_df.to_excel(concentration_path, index=False)

    wind_df = pd.DataFrame(payload["wind"])[["time", "dir", "sp"]].sort_values("time")
    wind_df.to_excel(wind_path, index=False)

    normalized_json = deepcopy(payload)
    for record in normalized_json["wind"]:
        record["time"] = record["time"].strftime(TIME_FORMAT)
    for record in normalized_json["concentrations"]:
        record["time"] = record["time"].strftime(TIME_FORMAT)
    (input_dir / "input.normalized.json").write_text(
        json.dumps(normalized_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return {
        "sites": sites_path,
        "concentration": concentration_path,
        "wind": wind_path,
    }
