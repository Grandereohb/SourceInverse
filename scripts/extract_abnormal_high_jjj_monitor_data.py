from __future__ import annotations

# Before running this script, use combine_jjj_hourly_monitor_workbooks.py to
# generate the multi-station JJJ monitor workbook from the raw station files.

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

from extract_abnormal_high_monitor_data import (
    DEFAULT_SKIP_POLLUTANTS as BASE_DEFAULT_SKIP_POLLUTANTS,
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

DEFAULT_INPUT_PATH = (
    DATA_DIR / "jjj" / "2026年小时数据" / "567月小时数据_标准单位_汇总.xlsx"
)
DEFAULT_OUTPUT_NAME = "abnormal_high_monitor_data_jjj_567.xlsx"

# Only output abnormal records whose concentration is at least this value.
# Overall pollutant means are still calculated from all data.
MIN_CONCENTRATION_THRESHOLD = 600.0
TIME_COLUMN = "时间"

# Pollutants in this list are excluded from both threshold calculation and output.
DEFAULT_SKIP_POLLUTANTS: list[str] = list(BASE_DEFAULT_SKIP_POLLUTANTS)

# Empty means keep all pollutants not present in DEFAULT_SKIP_POLLUTANTS.
# When non-empty, only pollutants matching this list are retained.
MANUAL_INPUT: list[str] = []

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


def clean_station_name(sheet_name: str) -> str:
    return re.sub(r"[（(]带标识[）)]$", "", str(sheet_name).strip()).strip()


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


def load_jjj_station_table(workbook: pd.ExcelFile, sheet_name: str) -> pd.DataFrame:
    table = pd.read_excel(workbook, sheet_name=sheet_name)
    table.columns = [clean_column_name(column) for column in table.columns]
    table = table.loc[:, [bool(str(column).strip()) for column in table.columns]]

    if TIME_COLUMN not in table.columns:
        raise ValueError(
            f"Parsed station sheet does not contain '{TIME_COLUMN}': {sheet_name}"
        )

    table[TIME_COLUMN] = pd.to_datetime(
        table[TIME_COLUMN], errors="coerce", format="mixed"
    )
    table = table.dropna(subset=[TIME_COLUMN]).reset_index(drop=True)
    return table


def pollutant_name_keys(name: str) -> set[str]:
    return {
        normalize_pollutant_name(name),
        normalize_pollutant_name(clean_pollutant_name_for_output(name)),
    }


def pollutant_columns(
    df: pd.DataFrame,
    skip_pollutants: list[str] | None = None,
    manual_input: list[str] | None = None,
) -> list[str]:
    skip_set = {
        key
        for name in (skip_pollutants or [])
        if str(name).strip()
        for key in pollutant_name_keys(name)
    }
    manual_set = {
        key
        for name in (manual_input or [])
        if str(name).strip()
        for key in pollutant_name_keys(name)
    }

    out: list[str] = []
    for col in df.columns:
        column = str(col).strip()
        if is_non_pollutant_column(column):
            continue
        column_keys = pollutant_name_keys(column)
        if column_keys & skip_set:
            continue
        if manual_set and not column_keys & manual_set:
            continue
        out.append(column)
    return out


def build_long_monitor_table(
    input_path: str | Path,
    skip_pollutants: list[str] | None = None,
    manual_input: list[str] | None = None,
) -> pd.DataFrame:
    path = resolve_path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input workbook not found: {path}")
    if not path.is_file():
        raise IsADirectoryError(f"Input path must be an Excel workbook: {path}")

    rows: list[pd.DataFrame] = []
    workbook = pd.ExcelFile(path)
    for sheet_name in workbook.sheet_names:
        station_name = clean_station_name(sheet_name)
        table = load_jjj_station_table(workbook, sheet_name)
        if table.empty:
            continue

        for pollutant in pollutant_columns(
            table,
            skip_pollutants=skip_pollutants,
            manual_input=manual_input,
        ):
            values = table[pollutant].map(extract_numeric)
            output_pollutant = clean_pollutant_name_for_output(pollutant)
            pollutant_rows = pd.DataFrame(
                {
                    "station": station_name,
                    "pollutant": output_pollutant,
                    "time": table[TIME_COLUMN],
                    "concentration": values,
                    "source_sheet": sheet_name,
                }
            )
            rows.append(pollutant_rows)

    if not rows:
        raise ValueError("No monitor rows were parsed from the input workbook.")

    long_df = pd.concat(rows, ignore_index=True)
    long_df = long_df.dropna(subset=["concentration"]).reset_index(drop=True)
    if long_df.empty:
        raise ValueError("No numeric concentration values were parsed.")
    return long_df


def extract_abnormal_high_jjj_monitor_data(
    input_path: str | Path,
    output_path: str | Path | None = None,
    multiplier: float = 5.0,
    min_concentration: float = MIN_CONCENTRATION_THRESHOLD,
    skip_pollutants: list[str] | None = None,
    manual_input: list[str] | None = MANUAL_INPUT,
) -> Path:
    if output_path is None:
        output_path = OUTPUT_DIR / DEFAULT_OUTPUT_NAME

    long_df = build_long_monitor_table(
        input_path=input_path,
        skip_pollutants=skip_pollutants,
        manual_input=manual_input,
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
            "Find abnormal high concentration records from a JJJ multi-sheet "
            "hourly monitor workbook. Each sheet represents one station."
        )
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        default=str(DEFAULT_INPUT_PATH),
        help="Multi-sheet JJJ hourly monitor workbook.",
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
        "--manual-input",
        nargs="*",
        default=MANUAL_INPUT,
        help="Only retain these pollutant names. Empty means retain all non-skipped pollutants.",
    )
    return parser.parse_args()


def safe_console_text(value) -> str:
    encoding = sys.stdout.encoding or "utf-8"
    return str(value).encode(encoding, errors="replace").decode(encoding)


def main() -> None:
    args = parse_args()
    output_path = extract_abnormal_high_jjj_monitor_data(
        input_path=args.input_path,
        output_path=args.output,
        multiplier=args.multiplier,
        min_concentration=args.min_concentration,
        skip_pollutants=args.skip_pollutants,
        manual_input=args.manual_input,
    )
    print(f"Input workbook: {resolve_path(args.input_path)}")
    print(f"Threshold: concentration > {args.multiplier:g} * pollutant overall mean")
    print(f"Minimum output concentration: {args.min_concentration:g}")
    if args.skip_pollutants:
        print(
            safe_console_text("Skipped pollutants: " + ", ".join(args.skip_pollutants))
        )
    if args.manual_input:
        print(safe_console_text("Retained pollutants: " + ", ".join(args.manual_input)))
    print(f"Saved abnormal high monitor table: {output_path}")


if __name__ == "__main__":
    main()
