from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

from extract_abnormal_high_monitor_data import (
    DEFAULT_SKIP_POLLUTANTS,
    extract_numeric,
    find_abnormal_high_records,
    normalize_pollutant_name,
    resolve_path,
    write_results,
)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "abnormal_high_monitor_data"

DEFAULT_INPUT_DIR = DATA_DIR / "jjj" / "2026年03-04-05月小时数据" / "5月小时数据"
DEFAULT_OUTPUT_NAME = "abnormal_high_monitor_data_jjj_may.xlsx"

# Only output abnormal records whose concentration is at least this value.
# Overall pollutant means are still calculated from all data.
MIN_CONCENTRATION_THRESHOLD = 800.0
TIME_COLUMN = "时间"
SHEET_INDEX = 1
NON_POLLUTANT_COLUMNS = {
    TIME_COLUMN,
    "温度",
    "湿度",
    "气压",
    "风速",
    "风向",
}


def clean_column_name(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def clean_station_name_from_path(path: Path) -> str:
    stem = path.stem.strip()
    marker = "_站点监测数据_"
    if marker in stem:
        return stem.split(marker, 1)[0].strip()
    return stem


def clean_pollutant_name_for_output(name: str) -> str:
    text = str(name).strip()
    unit_pattern = (
        r"\s*[\(（]\s*"
        r"(?:ug|μg|µg|mg|g|ng|ppm|ppb|mol|℃|%|hpa|m/s|°)"
        r"(?:\s*/\s*[^）\)]*)?"
        r"\s*[\)）]\s*$"
    )
    return re.sub(unit_pattern, "", text, flags=re.IGNORECASE).strip()


def is_non_pollutant_column(column: str) -> bool:
    text = str(column).strip()
    if not text:
        return True
    return any(
        text == name or text.startswith(f"{name}(") for name in NON_POLLUTANT_COLUMNS
    )


def find_header_row(raw_df: pd.DataFrame, source_path: Path) -> int:
    for idx, row in raw_df.iterrows():
        values = {clean_column_name(value) for value in row.tolist()}
        if TIME_COLUMN in values:
            return int(idx)
    raise ValueError(f"Could not find '{TIME_COLUMN}' header row in {source_path}")


def load_jjj_station_table(path: Path, sheet_index: int = SHEET_INDEX) -> pd.DataFrame:
    workbook = pd.ExcelFile(path)
    if len(workbook.sheet_names) <= sheet_index:
        raise ValueError(
            f"Workbook does not contain sheet index {sheet_index + 1}: {path}"
        )

    sheet_name = workbook.sheet_names[sheet_index]
    raw_df = pd.read_excel(workbook, sheet_name=sheet_name, header=None)
    header_row = find_header_row(raw_df, path)

    columns = [clean_column_name(value) for value in raw_df.iloc[header_row].tolist()]
    table = raw_df.iloc[header_row + 1 :].copy()
    table.columns = columns
    table = table.loc[:, [bool(str(col).strip()) for col in table.columns]]

    if TIME_COLUMN not in table.columns:
        raise ValueError(f"Parsed table does not contain '{TIME_COLUMN}': {path}")

    table[TIME_COLUMN] = pd.to_datetime(
        table[TIME_COLUMN], errors="coerce", format="mixed"
    )
    table = table.dropna(subset=[TIME_COLUMN]).reset_index(drop=True)
    return table


def pollutant_columns(
    df: pd.DataFrame, skip_pollutants: list[str] | None = None
) -> list[str]:
    skip_set: set[str] = set()
    for name in skip_pollutants or []:
        if not str(name).strip():
            continue
        skip_set.add(normalize_pollutant_name(name))
        skip_set.add(normalize_pollutant_name(clean_pollutant_name_for_output(name)))

    out: list[str] = []
    for col in df.columns:
        column = str(col).strip()
        if is_non_pollutant_column(column):
            continue
        column_keys = {
            normalize_pollutant_name(column),
            normalize_pollutant_name(clean_pollutant_name_for_output(column)),
        }
        if column_keys & skip_set:
            continue
        out.append(column)
    return out


def build_long_monitor_table(
    input_dir: str | Path,
    skip_pollutants: list[str] | None = None,
    sheet_index: int = SHEET_INDEX,
) -> pd.DataFrame:
    directory = resolve_path(input_dir)
    if not directory.exists():
        raise FileNotFoundError(f"Input directory not found: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {directory}")

    files = sorted(
        [
            path
            for pattern in ("*.xls", "*.xlsx")
            for path in directory.glob(pattern)
            if not path.name.startswith("~$")
        ]
    )
    if not files:
        raise FileNotFoundError(f"No Excel files found in input directory: {directory}")

    rows: list[pd.DataFrame] = []
    for path in files:
        station_name = clean_station_name_from_path(path)
        table = load_jjj_station_table(path, sheet_index=sheet_index)
        if table.empty:
            continue

        for pollutant in pollutant_columns(table, skip_pollutants=skip_pollutants):
            values = table[pollutant].map(extract_numeric)
            output_pollutant = clean_pollutant_name_for_output(pollutant)
            pollutant_rows = pd.DataFrame(
                {
                    "station": station_name,
                    "pollutant": output_pollutant,
                    "time": table[TIME_COLUMN],
                    "concentration": values,
                    "source_file": path.name,
                }
            )
            rows.append(pollutant_rows)

    if not rows:
        raise ValueError("No monitor rows were parsed from the input directory.")

    long_df = pd.concat(rows, ignore_index=True)
    long_df = long_df.dropna(subset=["concentration"]).reset_index(drop=True)
    if long_df.empty:
        raise ValueError("No numeric concentration values were parsed.")
    return long_df


def extract_abnormal_high_jjj_monitor_data(
    input_dir: str | Path,
    output_path: str | Path | None = None,
    multiplier: float = 5.0,
    min_concentration: float = MIN_CONCENTRATION_THRESHOLD,
    skip_pollutants: list[str] | None = None,
    sheet_index: int = SHEET_INDEX,
) -> Path:
    if output_path is None:
        output_path = OUTPUT_DIR / DEFAULT_OUTPUT_NAME

    long_df = build_long_monitor_table(
        input_dir=input_dir,
        skip_pollutants=skip_pollutants,
        sheet_index=sheet_index,
    )
    abnormal_df, means_df = find_abnormal_high_records(
        long_df=long_df,
        multiplier=multiplier,
        min_concentration=min_concentration,
    )
    return write_results(abnormal_df, means_df, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find abnormal high concentration records from jjj station workbooks. "
            "Each station is stored in a separate Excel file and sheet2 contains "
            "the hourly monitor table."
        )
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        default=str(DEFAULT_INPUT_DIR),
        help="Directory containing station Excel workbooks.",
    )
    parser.add_argument(
        "--output",
        default=str(OUTPUT_DIR / DEFAULT_OUTPUT_NAME),
        help="Output Excel path.",
    )
    parser.add_argument(
        "--multiplier",
        type=float,
        default=5.0,
        help="Abnormal threshold multiplier of the overall pollutant mean.",
    )
    parser.add_argument(
        "--min-concentration",
        type=float,
        default=MIN_CONCENTRATION_THRESHOLD,
        help=(
            "Only output abnormal records whose concentration is at least this "
            "value. Overall pollutant means are still calculated from all data."
        ),
    )
    parser.add_argument(
        "--skip-pollutants",
        nargs="*",
        default=DEFAULT_SKIP_POLLUTANTS,
        help="Pollutant names to skip entirely.",
    )
    parser.add_argument(
        "--sheet-index",
        type=int,
        default=SHEET_INDEX,
        help="Zero-based sheet index containing monitor data. Default 1 means sheet2.",
    )
    return parser.parse_args()


def safe_console_text(value) -> str:
    encoding = sys.stdout.encoding or "utf-8"
    return str(value).encode(encoding, errors="replace").decode(encoding)


def main() -> None:
    args = parse_args()
    output_path = extract_abnormal_high_jjj_monitor_data(
        input_dir=args.input_dir,
        output_path=args.output,
        multiplier=args.multiplier,
        min_concentration=args.min_concentration,
        skip_pollutants=args.skip_pollutants,
        sheet_index=args.sheet_index,
    )
    print(f"Input directory: {resolve_path(args.input_dir)}")
    print(f"Monitor data sheet index: {args.sheet_index} (zero-based)")
    print(f"Threshold: concentration > {args.multiplier:g} * pollutant overall mean")
    print(f"Minimum output concentration: {args.min_concentration:g}")
    if args.skip_pollutants:
        print(
            safe_console_text("Skipped pollutants: " + ", ".join(args.skip_pollutants))
        )
    print(f"Saved abnormal high monitor table: {output_path}")


if __name__ == "__main__":
    main()
