from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent
REPO_ROOT = DATA_DIR.parent

DEFAULT_INPUT_FILE_PATH = (
    DATA_DIR
    / "shsh_js"
    / "自动审核小时数据_标准单位_2025-10-16 00_00_00_2026-04-16 12_00_00.xlsx"
)
DEFAULT_OUTPUT_NAME = "abnormal_high_monitoar_data.xlsx"

# Only output abnormal records whose concentration is at least this value.
# Overall pollutant means are still calculated from all data.
MIN_CONCENTRATION_THRESHOLD = 200.0

# Pollutants in this list are not included in the mean calculation or output.
DEFAULT_SKIP_POLLUTANTS: list[str] = [
    "总氮",
    "总氮-参况",
    "氮氧化物(NOx)",
    "一氧化氮(NO)-参况",
    "氮氧化物(NOx)-参况",
    "氮氧化物(NOx)",
    "氨气(NH₃)-参况",
    "硫化氢(H₂S)-参况",
    "二氧化氮(NO₂)-参况",
    "二氧化硫(SO₂)-参况",
]

TIME_COLUMN = "时间"
SHEET_SUFFIX_PATTERNS = [
    r"\(带标识\)$",
    r"（带标识）$",
]
NON_POLLUTANT_COLUMNS = {
    TIME_COLUMN,
    "风向",
    "风速",
    "温度",
    "湿度",
    "气压",
}


def resolve_path(path: str | Path) -> Path:
    out = Path(path)
    if not out.is_absolute():
        out = (REPO_ROOT / out).resolve()
    return out


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


def load_station_table(workbook: pd.ExcelFile, sheet_name: str) -> pd.DataFrame:
    df = pd.read_excel(workbook, sheet_name=sheet_name)
    if TIME_COLUMN not in df.columns:
        raise ValueError(f"Sheet does not contain '{TIME_COLUMN}' column: {sheet_name}")

    out = df.copy()
    out[TIME_COLUMN] = pd.to_datetime(out[TIME_COLUMN], errors="coerce", format="mixed")
    out = out.dropna(subset=[TIME_COLUMN]).reset_index(drop=True)
    return out


def pollutant_columns(df: pd.DataFrame) -> list[str]:
    return [
        str(col) for col in df.columns if str(col).strip() not in NON_POLLUTANT_COLUMNS
    ]


def normalize_pollutant_name(name: str) -> str:
    text = str(name).strip().lower()
    text = text.replace("（", "(").replace("）", ")")
    text = re.sub(r"\s+", "", text)
    return text


def build_long_monitor_table(
    input_path: str | Path,
    skip_pollutants: list[str] | None = None,
) -> pd.DataFrame:
    path = resolve_path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input workbook not found: {path}")

    workbook = pd.ExcelFile(path)
    rows: list[pd.DataFrame] = []
    skip_set = {
        normalize_pollutant_name(name)
        for name in (skip_pollutants or [])
        if str(name).strip()
    }

    for sheet_name in workbook.sheet_names:
        station_name = clean_station_name(sheet_name)
        table = load_station_table(workbook, sheet_name)
        if table.empty:
            continue

        for pollutant in pollutant_columns(table):
            if normalize_pollutant_name(pollutant) in skip_set:
                continue

            values = table[pollutant].map(extract_numeric)
            pollutant_rows = pd.DataFrame(
                {
                    "station": station_name,
                    "pollutant": pollutant,
                    "time": table[TIME_COLUMN],
                    "concentration": values,
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


def find_abnormal_high_records(
    long_df: pd.DataFrame,
    multiplier: float = 5.0,
    min_concentration: float = MIN_CONCENTRATION_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    means = (
        long_df.groupby("pollutant", as_index=False)["concentration"]
        .mean()
        .rename(columns={"concentration": "mean_concentration"})
    )
    means["threshold_multiplier"] = float(multiplier)
    means["abnormal_threshold"] = means["mean_concentration"] * float(multiplier)

    with_threshold = long_df.merge(means, on="pollutant", how="left")
    abnormal = with_threshold[
        (with_threshold["concentration"] > with_threshold["abnormal_threshold"])
        & (with_threshold["concentration"] >= float(min_concentration))
    ].copy()
    abnormal["ratio_to_mean"] = (
        abnormal["concentration"] / abnormal["mean_concentration"]
    )
    abnormal["min_concentration_threshold"] = float(min_concentration)
    abnormal = abnormal[
        [
            "station",
            "pollutant",
            "time",
            "concentration",
            "mean_concentration",
            "abnormal_threshold",
            "min_concentration_threshold",
            "ratio_to_mean",
        ]
    ].sort_values(["time", "station", "pollutant"])

    means = means.sort_values("pollutant").reset_index(drop=True)
    return abnormal.reset_index(drop=True), means


def write_results(
    abnormal_df: pd.DataFrame,
    means_df: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    path = resolve_path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        abnormal_df.to_excel(writer, sheet_name="abnormal_high_records", index=False)
        means_df.to_excel(writer, sheet_name="pollutant_thresholds", index=False)
    return path


def extract_abnormal_high_monitor_data(
    input_path: str | Path,
    output_path: str | Path | None = None,
    multiplier: float = 5.0,
    min_concentration: float = MIN_CONCENTRATION_THRESHOLD,
    skip_pollutants: list[str] | None = None,
) -> Path:
    if output_path is None:
        output_path = SCRIPT_DIR / DEFAULT_OUTPUT_NAME

    long_df = build_long_monitor_table(
        input_path=input_path,
        skip_pollutants=skip_pollutants,
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
            "Find time points where a station pollutant concentration is higher "
            "than N times the pollutant's overall mean concentration."
        )
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        default=str(DEFAULT_INPUT_FILE_PATH),
        help="Path to the multi-sheet monitor Excel workbook.",
    )
    parser.add_argument(
        "--output",
        default=str(SCRIPT_DIR / DEFAULT_OUTPUT_NAME),
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
        help="Pollutant names to skip entirely. Example: --skip-pollutants 风向 风速",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = extract_abnormal_high_monitor_data(
        input_path=args.input_path,
        output_path=args.output,
        multiplier=args.multiplier,
        min_concentration=args.min_concentration,
        skip_pollutants=args.skip_pollutants,
    )
    print(f"Input workbook: {resolve_path(args.input_path)}")
    print(f"Threshold: concentration > {args.multiplier:g} * pollutant overall mean")
    print(f"Minimum output concentration: {args.min_concentration:g}")
    if args.skip_pollutants:
        print("Skipped pollutants: " + ", ".join(args.skip_pollutants))
    print(f"Saved abnormal high monitor table: {output_path}")


if __name__ == "__main__":
    main()
