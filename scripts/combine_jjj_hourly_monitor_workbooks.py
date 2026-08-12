from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_DIR = REPO_ROOT / "data"

DEFAULT_INPUT_DIR = DATA_DIR / "jjj" / "2026年小时数据" / "7月小时数据"
DEFAULT_OUTPUT_PATH = (
    DATA_DIR / "jjj" / "2026年小时数据" / "7月小时数据_标准单位_汇总.xlsx"
)

TIME_COLUMN = "时间"
SHEET_INDEX = 1
EXCEL_SUFFIXES = {".xls", ".xlsx", ".xlsm"}


def resolve_path(path: str | Path) -> Path:
    out = Path(path)
    if not out.is_absolute():
        out = (REPO_ROOT / out).resolve()
    return out


def clean_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def station_name_from_file(path: Path) -> str:
    stem = path.stem.strip()
    marker = "_站点监测数据_"
    if marker in stem:
        return stem.split(marker, 1)[0].strip()
    return stem


def safe_sheet_name(name: str, used: set[str]) -> str:
    invalid = r"[]:*?/\\"
    out = "".join("_" if char in invalid else char for char in str(name).strip())
    out = out or "Sheet"
    out = out[:31]
    base = out
    counter = 1
    while out in used:
        suffix = f"_{counter}"
        out = f"{base[: 31 - len(suffix)]}{suffix}"
        counter += 1
    used.add(out)
    return out


def find_header_row(raw_df: pd.DataFrame, source_path: Path) -> int:
    for idx, row in raw_df.iterrows():
        values = {clean_text(value) for value in row.tolist()}
        if TIME_COLUMN in values:
            return int(idx)
    raise ValueError(f"Could not find '{TIME_COLUMN}' header row in {source_path}")


def split_column_and_unit(name: str) -> tuple[str, str]:
    text = str(name).strip()
    if text == TIME_COLUMN:
        return TIME_COLUMN, ""

    match = re.match(r"^(?P<name>.+?)[\(（](?P<unit>[^()（）]+)[\)）]\s*$", text)
    if not match:
        return text, ""

    unit = match.group("unit").strip()
    if is_unit_text(unit):
        return match.group("name").strip(), unit
    return text, ""


def is_unit_text(text: str) -> bool:
    unit = str(text).strip().lower()
    return any(
        token in unit
        for token in [
            "μg",
            "µg",
            "ug",
            "mg",
            "ng",
            "m³",
            "m3",
            "ppm",
            "ppb",
            "℃",
            "%rh",
            "hpa",
            "m/s",
            "°",
        ]
    )


def load_station_sheet(path: Path, sheet_index: int = SHEET_INDEX) -> pd.DataFrame:
    workbook = pd.ExcelFile(path)
    if len(workbook.sheet_names) <= sheet_index:
        raise ValueError(
            f"Workbook does not contain sheet index {sheet_index + 1}: {path}"
        )

    raw_df = pd.read_excel(
        workbook, sheet_name=workbook.sheet_names[sheet_index], header=None
    )
    header_row = find_header_row(raw_df, path)
    raw_columns = [clean_text(value) for value in raw_df.iloc[header_row].tolist()]

    table = raw_df.iloc[header_row + 1 :].copy()
    table.columns = raw_columns
    table = table.loc[:, [bool(str(col).strip()) for col in table.columns]]
    table = table.dropna(how="all")

    if TIME_COLUMN not in table.columns:
        raise ValueError(f"Parsed table does not contain '{TIME_COLUMN}': {path}")

    table[TIME_COLUMN] = pd.to_datetime(
        table[TIME_COLUMN], errors="coerce", format="mixed"
    )
    table = (
        table.dropna(subset=[TIME_COLUMN])
        .sort_values(TIME_COLUMN)
        .reset_index(drop=True)
    )

    clean_columns: list[str] = []
    units: list[str] = []
    used_columns: dict[str, int] = {}
    for col in table.columns:
        clean_col, unit = split_column_and_unit(str(col))
        count = used_columns.get(clean_col, 0)
        used_columns[clean_col] = count + 1
        if count:
            clean_col = f"{clean_col}_{count + 1}"
        clean_columns.append(clean_col)
        units.append(unit)

    table.columns = clean_columns
    unit_row = {col: unit for col, unit in zip(clean_columns, units)}
    unit_row[TIME_COLUMN] = "标准单位"
    return pd.concat([pd.DataFrame([unit_row]), table], ignore_index=True)


def list_station_workbooks(input_dir: str | Path) -> list[Path]:
    directory = resolve_path(input_dir)
    if not directory.exists():
        raise FileNotFoundError(f"Input directory not found: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {directory}")

    files = [
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower() in EXCEL_SUFFIXES
        and not path.name.startswith("~$")
    ]
    if not files:
        raise FileNotFoundError(f"No station Excel files found in: {directory}")
    return sorted(files)


def combine_jjj_hourly_workbooks(
    input_dir: str | Path = DEFAULT_INPUT_DIR,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    sheet_index: int = SHEET_INDEX,
) -> Path:
    files = list_station_workbooks(input_dir)
    output = resolve_path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    used_sheet_names: set[str] = set()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for path in files:
            station_name = station_name_from_file(path)
            sheet_name = safe_sheet_name(f"{station_name}(带标识)", used_sheet_names)
            table = load_station_sheet(path, sheet_index=sheet_index)
            table.to_excel(writer, sheet_name=sheet_name, index=False)

    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine JJJ hourly monitor workbooks into one multi-sheet workbook "
            "similar to the SHSH-JS standard-unit hourly data file."
        )
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        default=str(DEFAULT_INPUT_DIR),
        help="Directory containing one station Excel workbook per file.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Output workbook path.",
    )
    parser.add_argument(
        "--sheet-index",
        type=int,
        default=SHEET_INDEX,
        help="Zero-based sheet index containing hourly monitor data. Default 1 means sheet2.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = combine_jjj_hourly_workbooks(
        input_dir=args.input_dir,
        output_path=args.output,
        sheet_index=args.sheet_index,
    )
    print(f"Input directory: {resolve_path(args.input_dir)}")
    print(f"Monitor data sheet index: {args.sheet_index} (zero-based)")
    print(f"Saved combined workbook: {output}")


if __name__ == "__main__":
    main()
