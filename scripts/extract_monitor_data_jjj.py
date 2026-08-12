from __future__ import annotations

import math
import re
from datetime import datetime
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_JJJ_DIR = REPO_ROOT / "data" / "jjj"

DEFAULT_CONCENTRATION_NAME = "concentration.xlsx"
DEFAULT_WIND_NAME = "wind.xlsx"
DEFAULT_SITES_NAME = "sites.xlsx"

# =========================
# Manual Inputs
# =========================
# INPUT_FILE_PATH: combined JJJ workbook with one station per sheet. Generate it
# first with scripts/combine_jjj_hourly_monitor_workbooks.py.
INPUT_FILE_PATH = 'data/jjj/2026年小时数据/567月小时数据_标准单位_汇总.xlsx'
# START_TIME / END_TIME: inclusive extraction time range.
START_TIME = '2026-05-03 19:00:00'
END_TIME = '2026-05-04 07:00:00'
# TARGET_POLLUTANT: pollutant to extract. It may be either the bare pollutant
# name, such as "非甲烷总烃", or the full Excel column name, such as
# "非甲烷总烃(μg/m³)".
TARGET_POLLUTANT = '氨'
# WIND_STATION_NAME: use this station's wind direction/speed to build wind.xlsx.
# Leave empty to keep the old behavior: average wind from all station files.
WIND_STATION_NAME = 'H5站点（淮河道区域点位）'
# SITES_FILE_PATH: workbook containing current station location information.
SITES_FILE_PATH = r"data\jjj\2026年小时数据\当前数据点位信息.xlsx"

# OUTPUT_FOLDER:
# - empty string: save directly into data/jjj/
# - relative path: save into data/jjj/<OUTPUT_FOLDER>/
# - absolute path: save into that absolute directory
OUTPUT_FOLDER = ''
# LOCKED_OUTPUT_POLICY:
# - "timestamp_folder": if concentration/wind/sites.xlsx is open in Excel/WPS,
#   write all three standard files into data/jjj/extracted_<timestamp>/.
# - "error": stop and tell you which output file is locked.
LOCKED_OUTPUT_POLICY = "timestamp_folder"

TIME_COLUMN = "时间"
WIND_DIR_COLUMN = "风向"
WIND_SPEED_COLUMN = "风速"


def resolve_path(path: str | Path, base_dir: Path = REPO_ROOT) -> Path:
    out = Path(path)
    if not out.is_absolute():
        out = (base_dir / out).resolve()
    return out


def resolve_output_dir(output_folder: str | None) -> Path:
    if output_folder is None or not str(output_folder).strip():
        out_dir = DATA_JJJ_DIR
    else:
        out_path = Path(output_folder)
        out_dir = out_path if out_path.is_absolute() else (DATA_JJJ_DIR / out_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


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


def normalize_pollutant_name(name: str) -> str:
    text = str(name).strip().lower()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"\([^)]*\)", "", text)
    text = re.sub(r"（[^）]*）", "", text)
    return text


def find_column(columns: list[str], target: str) -> str | None:
    if target in columns:
        return target

    target_norm = normalize_pollutant_name(target)
    for col in columns:
        if normalize_pollutant_name(col) == target_norm:
            return col
    return None


def clean_station_name(sheet_name: str) -> str:
    return re.sub(r"[（(]带标识[）)]$", "", str(sheet_name).strip()).strip()


def station_name_matches(station_name: str, target_station_name: str | None) -> bool:
    if target_station_name is None or not str(target_station_name).strip():
        return True
    return str(station_name).strip() == str(target_station_name).strip()


def load_station_hourly_table(
    workbook: pd.ExcelFile, sheet_name: str
) -> pd.DataFrame:
    table = pd.read_excel(workbook, sheet_name=sheet_name)
    table = table.loc[
        :, [pd.notna(column) and bool(str(column).strip()) for column in table.columns]
    ]
    table.columns = [str(column).strip() for column in table.columns]
    table = table.dropna(how="all")

    if TIME_COLUMN not in table.columns:
        raise ValueError(
            f"Parsed station sheet does not contain '{TIME_COLUMN}': {sheet_name}"
        )

    table[TIME_COLUMN] = pd.to_datetime(
        table[TIME_COLUMN], errors="coerce", format="mixed"
    )
    table = table.dropna(subset=[TIME_COLUMN])
    return table


def filter_time_range(
    df: pd.DataFrame,
    start_time: str | pd.Timestamp | None,
    end_time: str | pd.Timestamp | None,
) -> pd.DataFrame:
    out = df.copy()
    if start_time is not None and str(start_time).strip():
        out = out[out[TIME_COLUMN] >= pd.to_datetime(start_time)]
    if end_time is not None and str(end_time).strip():
        out = out[out[TIME_COLUMN] <= pd.to_datetime(end_time)]
    return out.sort_values(TIME_COLUMN).reset_index(drop=True)


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


def build_concentration_and_wind_tables(
    input_path: str | Path,
    start_time: str | pd.Timestamp | None,
    end_time: str | pd.Timestamp | None,
    pollutant: str,
    wind_station_name: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    concentration_by_station: dict[str, list[pd.DataFrame]] = {}
    wind_frames: list[pd.DataFrame] = []
    missing_pollutant: list[str] = []
    station_names: list[str] = []

    path = resolve_path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Monitor workbook not found: {path}")
    if not path.is_file():
        raise IsADirectoryError(f"Monitor input must be an Excel workbook: {path}")

    workbook = pd.ExcelFile(path)
    for sheet_name in workbook.sheet_names:
        station_name = clean_station_name(sheet_name)
        table = filter_time_range(
            load_station_hourly_table(workbook, sheet_name),
            start_time,
            end_time,
        )
        if table.empty:
            continue

        pollutant_col = find_column(list(table.columns), pollutant)
        if pollutant_col is None:
            missing_pollutant.append(station_name)
        else:
            station_df = table[[TIME_COLUMN, pollutant_col]].copy()
            station_df[station_name] = station_df[pollutant_col].map(extract_numeric)
            station_df = station_df[[TIME_COLUMN, station_name]]
            concentration_by_station.setdefault(station_name, []).append(station_df)
            if station_name not in station_names:
                station_names.append(station_name)

        if (
            station_name_matches(station_name, wind_station_name)
            and WIND_DIR_COLUMN in table.columns
            and WIND_SPEED_COLUMN in table.columns
        ):
            wind_df = table[[TIME_COLUMN, WIND_DIR_COLUMN, WIND_SPEED_COLUMN]].copy()
            wind_df["dir"] = wind_df[WIND_DIR_COLUMN].map(extract_numeric)
            wind_df["sp"] = wind_df[WIND_SPEED_COLUMN].map(extract_numeric)
            wind_frames.append(wind_df[[TIME_COLUMN, "dir", "sp"]])

    if not concentration_by_station:
        detail = (
            f"pollutant '{pollutant}' not found in any station sheet"
            if missing_pollutant
            else "no rows matched the requested time range"
        )
        raise ValueError(f"Failed to build concentration table: {detail}.")
    if not wind_frames:
        if wind_station_name is None or not str(wind_station_name).strip():
            raise ValueError(
                "Failed to build wind table: no station table has wind columns."
            )
        raise ValueError(
            "Failed to build wind table: no wind data found for station "
            f"'{wind_station_name}'."
        )

    concentration_frames: list[pd.DataFrame] = []
    for station_name in station_names:
        station_all = pd.concat(
            concentration_by_station[station_name], ignore_index=True
        )
        station_all = (
            station_all.groupby(TIME_COLUMN, as_index=False)[station_name]
            .mean(numeric_only=True)
            .sort_values(TIME_COLUMN)
            .reset_index(drop=True)
        )
        concentration_frames.append(station_all)

    concentration_df = concentration_frames[0]
    for frame in concentration_frames[1:]:
        concentration_df = concentration_df.merge(frame, on=TIME_COLUMN, how="outer")

    concentration_df = concentration_df.sort_values(TIME_COLUMN).reset_index(drop=True)
    concentration_df["TARGET_POLLUTANT"] = pollutant

    wind_long = pd.concat(wind_frames, ignore_index=True)
    grouped_rows = []
    for ts, group in wind_long.groupby(TIME_COLUMN, sort=True):
        sp_values = pd.to_numeric(group["sp"], errors="coerce").dropna()
        grouped_rows.append(
            {
                TIME_COLUMN: ts,
                "dir": circular_mean_deg(group["dir"]),
                "sp": float(sp_values.mean()) if not sp_values.empty else None,
            }
        )
    wind_df = pd.DataFrame(grouped_rows).sort_values(TIME_COLUMN).reset_index(drop=True)

    return concentration_df, wind_df, station_names


def load_sites_table(
    sites_path: str | Path, station_order: list[str] | None = None
) -> pd.DataFrame:
    path = resolve_path(sites_path)
    if not path.exists():
        raise FileNotFoundError(f"Sites workbook not found: {path}")

    raw = pd.read_excel(path, header=None)
    coords: dict[str, tuple[float, float]] = {}
    for _, row in raw.iterrows():
        station = row.iloc[1] if len(row) > 1 else None
        coord_text = row.iloc[3] if len(row) > 3 else None
        if pd.isna(station) or pd.isna(coord_text):
            continue
        if "," not in str(coord_text):
            continue

        lon_text, lat_text = str(coord_text).split(",", 1)
        lon = extract_numeric(lon_text)
        lat = extract_numeric(lat_text)
        if lon is None or lat is None:
            continue
        coords[str(station).strip()] = (lon, lat)

    if not coords:
        raise ValueError(f"No station coordinates parsed from: {path}")

    if station_order is None:
        ordered_names = list(coords)
    else:
        ordered_names = [name for name in station_order if name in coords]

    site_df = pd.DataFrame({"station": ["lon", "lat"]})
    for name in ordered_names:
        lon, lat = coords[name]
        site_df[name] = [lon, lat]
    return site_df


def write_excel(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_excel(path, index=False)
    except PermissionError as exc:
        raise PermissionError(
            f"Cannot write output file because it is locked or not writable: {path}. "
            "Close this workbook in Excel/WPS and run the script again, or set "
            "OUTPUT_FOLDER to another directory."
        ) from exc


def output_file_is_locked(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with path.open("a+b"):
            return False
    except PermissionError:
        return True


def prepare_output_paths(out_dir: Path) -> tuple[Path, Path, Path]:
    paths = (
        out_dir / DEFAULT_CONCENTRATION_NAME,
        out_dir / DEFAULT_WIND_NAME,
        out_dir / DEFAULT_SITES_NAME,
    )
    locked_paths = [path for path in paths if output_file_is_locked(path)]
    if not locked_paths:
        return paths

    if LOCKED_OUTPUT_POLICY != "timestamp_folder":
        locked_text = ", ".join(str(path) for path in locked_paths)
        raise PermissionError(
            "Output file is locked or not writable. Close it in Excel/WPS and "
            f"run again: {locked_text}"
        )

    fallback_dir = out_dir / f"extracted_{datetime.now():%Y%m%d_%H%M%S}"
    fallback_dir.mkdir(parents=True, exist_ok=True)
    print(
        "Output file is locked, writing this extraction to fallback folder: "
        f"{fallback_dir}"
    )
    return (
        fallback_dir / DEFAULT_CONCENTRATION_NAME,
        fallback_dir / DEFAULT_WIND_NAME,
        fallback_dir / DEFAULT_SITES_NAME,
    )


def extract_hourly_monitor_data(
    input_path: str | Path,
    start_time: str | pd.Timestamp | None,
    end_time: str | pd.Timestamp | None,
    pollutant: str,
    sites_path: str | Path,
    wind_station_name: str | None = None,
    output_folder: str | None = None,
) -> tuple[Path, Path, Path]:
    out_dir = resolve_output_dir(output_folder)

    concentration_df, wind_df, station_names = build_concentration_and_wind_tables(
        input_path=input_path,
        start_time=start_time,
        end_time=end_time,
        pollutant=pollutant,
        wind_station_name=wind_station_name,
    )
    sites_df = load_sites_table(sites_path, station_order=station_names)

    concentration_path, wind_path, sites_path_out = prepare_output_paths(out_dir)

    write_excel(concentration_df, concentration_path)
    write_excel(wind_df, wind_path)
    write_excel(sites_df, sites_path_out)

    return concentration_path, wind_path, sites_path_out


def main() -> None:
    concentration_path, wind_path, sites_path = extract_hourly_monitor_data(
        input_path=INPUT_FILE_PATH,
        start_time=START_TIME,
        end_time=END_TIME,
        pollutant=TARGET_POLLUTANT,
        sites_path=SITES_FILE_PATH,
        wind_station_name=WIND_STATION_NAME,
        output_folder=OUTPUT_FOLDER,
    )

    print(f"Monitor workbook: {INPUT_FILE_PATH}")
    print(f"Time range: {START_TIME} -> {END_TIME}")
    print(f"Pollutant: {TARGET_POLLUTANT}")
    print(f"Wind station: {WIND_STATION_NAME or 'all stations mean'}")
    print(f"Saved concentration file: {concentration_path}")
    print(f"Saved wind file: {wind_path}")
    print(f"Saved sites file: {sites_path}")


if __name__ == "__main__":
    main()
