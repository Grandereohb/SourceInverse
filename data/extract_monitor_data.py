from __future__ import annotations

import math
import re
from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parent
DEFAULT_CONCENTRATION_NAME = "concentration.xlsx"
DEFAULT_WIND_NAME = "wind.xlsx"

# =========================
# Manual Inputs
# =========================
# INPUT_FILE_PATH: source workbook path. Relative paths are resolved from the repo root.
INPUT_FILE_PATH = r"data\shsh_js\自动审核小时数据_标准单位_2025-10-16 00_00_00_2026-04-16 12_00_00.xlsx"

# START_TIME / END_TIME: inclusive extraction time range.
START_TIME = "2026-01-19 12:00:00"
END_TIME = "2026-01-21 00:00:00"

# TARGET_POLLUTANT: pollutant column name to extract.
TARGET_POLLUTANT = "间-二甲苯+对-二甲苯"

# OUTPUT_FOLDER:
# - empty string: save directly into data/
# - relative path: save into data/<OUTPUT_FOLDER>/
# - absolute path: save into that absolute directory
OUTPUT_FOLDER = "shsh_js"

SHEET_SUFFIX_PATTERNS = [
    r"\(带标识\)$",
    r"（带标识）$",
]


def clean_station_name(sheet_name: str) -> str:
    name = str(sheet_name).strip()
    for pattern in SHEET_SUFFIX_PATTERNS:
        name = re.sub(pattern, "", name)
    return name.strip()


def extract_numeric(value) -> float | None:
    if pd.isna(value):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)

    text = str(value).strip()
    if not text:
        return None

    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if match is None:
        return None
    return float(match.group(0))


def normalize_time_column(df: pd.DataFrame) -> pd.DataFrame:
    if "时间" not in df.columns:
        raise ValueError("Input sheet does not contain '时间' column.")
    out = df.copy()
    out["时间"] = pd.to_datetime(out["时间"], errors="coerce", format="mixed")
    out = out.dropna(subset=["时间"])
    return out


def filter_time_range(
    df: pd.DataFrame,
    start_time: str | pd.Timestamp | None,
    end_time: str | pd.Timestamp | None,
) -> pd.DataFrame:
    out = normalize_time_column(df)
    if start_time is not None and str(start_time).strip():
        start_ts = pd.to_datetime(start_time)
        out = out[out["时间"] >= start_ts]
    if end_time is not None and str(end_time).strip():
        end_ts = pd.to_datetime(end_time)
        out = out[out["时间"] <= end_ts]
    return out.sort_values("时间").reset_index(drop=True)


def find_column(columns: list[str], target: str) -> str | None:
    if target in columns:
        return target

    target_norm = str(target).strip().lower()
    for col in columns:
        if str(col).strip().lower() == target_norm:
            return col
    return None


def circular_mean_deg(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None

    radians = values.to_numpy() * math.pi / 180.0
    sin_mean = float(pd.Series([math.sin(v) for v in radians]).mean())
    cos_mean = float(pd.Series([math.cos(v) for v in radians]).mean())

    if abs(sin_mean) < 1e-12 and abs(cos_mean) < 1e-12:
        return None

    angle = math.degrees(math.atan2(sin_mean, cos_mean))
    if angle < 0:
        angle += 360.0
    return angle


def resolve_output_dir(output_folder: str | None) -> Path:
    if output_folder is None or not str(output_folder).strip():
        out_dir = DATA_DIR
    else:
        out_path = Path(output_folder)
        out_dir = out_path if out_path.is_absolute() else (DATA_DIR / out_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def load_workbook(input_path: str | Path) -> pd.ExcelFile:
    path = Path(input_path)
    if not path.is_absolute():
        path = (DATA_DIR.parent / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input workbook not found: {path}")
    return pd.ExcelFile(path)


def build_concentration_table(
    workbook: pd.ExcelFile,
    start_time: str | pd.Timestamp | None,
    end_time: str | pd.Timestamp | None,
    pollutant: str,
) -> pd.DataFrame:
    station_frames: list[pd.DataFrame] = []
    missing_stations: list[str] = []

    for sheet_name in workbook.sheet_names:
        station_name = clean_station_name(sheet_name)
        df = pd.read_excel(workbook, sheet_name=sheet_name)
        df = filter_time_range(df, start_time, end_time)
        if df.empty:
            continue

        pollutant_col = find_column([str(c) for c in df.columns], pollutant)
        if pollutant_col is None:
            missing_stations.append(station_name)
            continue

        station_df = df[["时间", pollutant_col]].copy()
        station_df[station_name] = station_df[pollutant_col].map(extract_numeric)
        station_df = station_df[["时间", station_name]]
        station_frames.append(station_df)

    if not station_frames:
        detail = (
            f"pollutant '{pollutant}' not found in any sheet"
            if missing_stations
            else "no rows matched the requested time range"
        )
        raise ValueError(f"Failed to build concentration table: {detail}.")

    merged = station_frames[0]
    for frame in station_frames[1:]:
        merged = merged.merge(frame, on="时间", how="outer")

    merged = merged.sort_values("时间").reset_index(drop=True)
    return merged


def build_wind_table(
    workbook: pd.ExcelFile,
    start_time: str | pd.Timestamp | None,
    end_time: str | pd.Timestamp | None,
) -> pd.DataFrame:
    wind_frames: list[pd.DataFrame] = []

    for sheet_name in workbook.sheet_names:
        df = pd.read_excel(workbook, sheet_name=sheet_name)
        df = filter_time_range(df, start_time, end_time)
        if df.empty:
            continue

        columns = [str(c) for c in df.columns]
        dir_col = find_column(columns, "风向")
        sp_col = find_column(columns, "风速")
        if dir_col is None or sp_col is None:
            continue

        station_wind = df[["时间", dir_col, sp_col]].copy()
        station_wind["dir"] = station_wind[dir_col].map(extract_numeric)
        station_wind["sp"] = station_wind[sp_col].map(extract_numeric)
        station_wind = station_wind[["时间", "dir", "sp"]]
        wind_frames.append(station_wind)

    if not wind_frames:
        raise ValueError(
            "Failed to build wind table: no sheet contains both 风向 and 风速 columns."
        )

    wind_long = pd.concat(wind_frames, ignore_index=True)

    grouped_rows = []
    for ts, group in wind_long.groupby("时间", sort=True):
        dir_mean = circular_mean_deg(group["dir"])
        sp_mean = pd.to_numeric(group["sp"], errors="coerce").dropna()
        grouped_rows.append(
            {
                "时间": ts,
                "dir": dir_mean,
                "sp": float(sp_mean.mean()) if not sp_mean.empty else None,
            }
        )

    wind_df = pd.DataFrame(grouped_rows).sort_values("时间").reset_index(drop=True)
    return wind_df


def extract_monitor_data(
    input_path: str | Path,
    start_time: str | pd.Timestamp | None,
    end_time: str | pd.Timestamp | None,
    pollutant: str,
    output_folder: str | None = None,
) -> tuple[Path, Path]:
    workbook = load_workbook(input_path)
    out_dir = resolve_output_dir(output_folder)

    concentration_df = build_concentration_table(
        workbook=workbook,
        start_time=start_time,
        end_time=end_time,
        pollutant=pollutant,
    )
    wind_df = build_wind_table(
        workbook=workbook,
        start_time=start_time,
        end_time=end_time,
    )

    concentration_path = out_dir / DEFAULT_CONCENTRATION_NAME
    wind_path = out_dir / DEFAULT_WIND_NAME

    concentration_df.to_excel(concentration_path, index=False)
    wind_df.to_excel(wind_path, index=False)

    return concentration_path, wind_path


def main() -> None:
    concentration_path, wind_path = extract_monitor_data(
        input_path=INPUT_FILE_PATH,
        start_time=START_TIME,
        end_time=END_TIME,
        pollutant=TARGET_POLLUTANT,
        output_folder=OUTPUT_FOLDER,
    )

    print(f"Input workbook: {INPUT_FILE_PATH}")
    print(f"Time range: {START_TIME} -> {END_TIME}")
    print(f"Pollutant: {TARGET_POLLUTANT}")
    print(f"Saved concentration file: {concentration_path}")
    print(f"Saved wind file: {wind_path}")


if __name__ == "__main__":
    main()
