from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_DIR = REPO_ROOT / "data"
RESULT_DIR = REPO_ROOT / "result"

ABNORMAL_DIR = DATA_DIR / "abnormal_high_monitor_data"
PINN_SCRIPT = REPO_ROOT / "pinn_source" / "pinn_source_pinn.py"
PINN_CONFIG = REPO_ROOT / "pinn_source" / "config.py"

# INPUT_FILE_PATH = ABNORMAL_DIR / "abnormal_high_vocs_odorous_gases.xlsx"

# MONITOR_INPUT_PATH = (
#     DATA_DIR
#     / "shsh_js"
#     / "自动审核小时数据_标准单位_2026-05-15 00_00_00_2026-06-14 23_00_00.xlsx"
# )

# EXTRACT_SCRIPT_KEY = "shsh_js"
# EXTRACT_OUTPUT_FOLDER = "shsh_js"

# Abnormal-high event workbook to traverse. This must point to a concrete Excel
# file with an `abnormal_high_records` sheet, not just a directory.
INPUT_FILE_PATH = ABNORMAL_DIR / "abnormal_high_monitor_data_jjj_may.xlsx"
MONITOR_INPUT_PATH = DATA_DIR / "jjj" / "2026年03-04-05月小时数据" / "5月小时数据"
EXTRACT_SCRIPT_KEY = "jjj"
EXTRACT_OUTPUT_FOLDER = ""


# =========================
# Manual Inputs
# =========================
# SOURCE_INVERSION_COUNT: number of leak events to run in sequence.
SOURCE_INVERSION_COUNT = 15

# TRAVERSE_DIRECTION:
# - "backward": traverse leak events from later time to earlier time.
# - "forward": traverse leak events from earlier time to later time.
TRAVERSE_DIRECTION = "backward"

# START_TRAVERSE_TIME:
# - empty string: start from latest leak for "backward", earliest leak for "forward".
# - "YYYY-MM-DD HH:MM:SS": start traversing from this time.
START_TRAVERSE_TIME = ""

# POLLUTANT_CONTAINS:
# - empty string: include all pollutants.
# - non-empty string: only include leak events whose pollutant name contains it.
POLLUTANT_CONTAINS = ""


def resolve_extract_script(output_folder: str) -> Path:
    candidates = []
    if output_folder:
        candidates.append(SCRIPT_DIR / f"extract_monitor_data_{output_folder}.py")
    candidates.extend(
        [
            SCRIPT_DIR / "extract_monitor_data.py",
            DATA_DIR / "extract_monitor_data.py",
        ]
    )
    for path in candidates:
        if path.exists():
            return path
    searched = "\n".join(f"- {path}" for path in candidates)
    raise FileNotFoundError(f"No extract monitor script found. Searched:\n{searched}")


def safe_text(value) -> str:
    return str(value).encode("gbk", errors="backslashreplace").decode("gbk")


def safe_print(message: str) -> None:
    print(safe_text(message), flush=True)


def build_leaks(abnormal_path: Path) -> list[dict]:
    df = pd.read_excel(abnormal_path, sheet_name="abnormal_high_records")
    required = {"station", "pollutant", "time", "concentration"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"Abnormal workbook is missing required columns: {sorted(missing)}"
        )

    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    leaks: list[dict] = []

    for pollutant, group in df.sort_values("time").groupby("pollutant"):
        times = [pd.Timestamp(t) for t in sorted(group["time"].dropna().unique())]
        if not times:
            continue

        time_groups: list[list[pd.Timestamp]] = []
        current = [times[0]]
        for ts in times[1:]:
            if ts - current[-1] == pd.Timedelta(hours=1):
                current.append(ts)
            else:
                time_groups.append(current)
                current = [ts]
        time_groups.append(current)

        for time_group in time_groups:
            leak_rows = group[group["time"].isin(time_group)].copy()
            max_row = leak_rows.loc[leak_rows["concentration"].idxmax()]
            start_time = min(time_group)
            end_time = max(time_group)
            leaks.append(
                {
                    "pollutant": str(pollutant),
                    "leak_start": start_time,
                    "leak_end": end_time,
                    "start_time": start_time - pd.Timedelta(hours=6),
                    "end_time": end_time + pd.Timedelta(hours=6),
                    "wind_station": str(max_row["station"]),
                    "max_concentration": float(max_row["concentration"]),
                    "n_hours": len(time_group),
                    "n_records": int(len(leak_rows)),
                }
            )

    return sorted(
        leaks,
        key=lambda row: (row["leak_end"], row["leak_start"], row["pollutant"]),
        reverse=True,
    )


def parse_optional_time(value: str | None) -> pd.Timestamp | None:
    if value is None or not str(value).strip():
        return None
    return pd.to_datetime(value)


def select_leaks(
    leaks: list[dict],
    count: int,
    direction: str,
    start_time: str | None = None,
    start_rank: int | None = None,
    pollutant_contains: str | None = None,
) -> list[tuple[int, dict]]:
    if count <= 0:
        raise ValueError("count must be > 0")

    direction_norm = str(direction).strip().lower()
    if direction_norm not in {"backward", "forward"}:
        raise ValueError("direction must be 'backward' or 'forward'")

    reverse = direction_norm == "backward"
    ordered = sorted(
        leaks,
        key=lambda row: (row["leak_end"], row["leak_start"], row["pollutant"]),
        reverse=reverse,
    )

    start_ts = parse_optional_time(start_time)
    if start_ts is not None:
        if reverse:
            ordered = [row for row in ordered if row["leak_end"] <= start_ts]
        else:
            ordered = [row for row in ordered if row["leak_start"] >= start_ts]

    pollutant_filter = str(pollutant_contains or "").strip()
    if pollutant_filter:
        ordered = [row for row in ordered if pollutant_filter in str(row["pollutant"])]

    if start_rank is not None:
        if start_rank < 1:
            raise ValueError("start_rank must be >= 1")
        ordered = ordered[start_rank - 1 :]
        first_rank = start_rank
    else:
        first_rank = 1

    selected = ordered[:count]
    return [(first_rank + offset, leak) for offset, leak in enumerate(selected)]


def python_literal(value: str) -> str:
    return repr(str(value))


def python_config_value(value: str | Path) -> str:
    if isinstance(value, Path):
        return value.resolve().as_posix()
    return str(value)


def resolve_repo_path(path: str | Path) -> Path:
    out = Path(path)
    if not out.is_absolute():
        out = (REPO_ROOT / out).resolve()
    return out


def display_path_for_config(path: str | Path) -> str:
    out = resolve_repo_path(path)
    try:
        return out.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return out.as_posix()


def replace_python_string_assignment(text: str, name: str, value: str | Path) -> str:
    pattern = (
        rf"^{name}\s*=\s*"
        rf'(?:\(\s*(?:r)?["\'][\s\S]*?["\']\s*\)|(?:r)?["\'].*?["\'])'
        rf"\s*$"
    )
    replacement = f"{name} = {python_literal(python_config_value(value))}"
    text, count = re.subn(pattern, replacement, text, count=1, flags=re.M)
    if count != 1:
        raise ValueError(f"Could not update {name}")
    return text


def resolved_extract_output_dir(
    extract_script_key: str = EXTRACT_SCRIPT_KEY,
    extract_output_folder: str | Path = EXTRACT_OUTPUT_FOLDER,
) -> Path:
    output = Path(extract_output_folder)
    if output.is_absolute():
        return output

    if extract_script_key == "jjj":
        base_dir = DATA_DIR / "jjj"
    else:
        base_dir = DATA_DIR

    if not str(extract_output_folder).strip():
        return base_dir
    return base_dir / output


def parse_extracted_input_paths(log_path: Path) -> dict[str, Path]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    patterns = {
        "CONC_PATH": r"^Saved concentration file:\s*(.+?)\s*$",
        "WIND_PATH": r"^Saved wind file:\s*(.+?)\s*$",
    }
    parsed: dict[str, Path] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.M)
        if match is None:
            raise ValueError(f"Could not parse {key} from extraction log: {log_path}")
        path = Path(match.group(1).strip())
        if not path.is_absolute():
            path = (REPO_ROOT / path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Parsed extracted input does not exist: {path}")
        parsed[key] = path

    site_match = re.search(r"^Saved sites file:\s*(.+?)\s*$", text, flags=re.M)
    if site_match is not None:
        site_path = Path(site_match.group(1).strip())
        if not site_path.is_absolute():
            site_path = (REPO_ROOT / site_path).resolve()
    else:
        site_path = parsed["CONC_PATH"].parent / "sites.xlsx"
    if not site_path.exists():
        raise FileNotFoundError(
            f"Could not find sites.xlsx for extracted inputs. Expected: {site_path}"
        )
    parsed["SITE_PATH"] = site_path
    return parsed


def update_pinn_config_inputs(
    leak: dict,
    extracted_paths: dict[str, Path] | None = None,
    extract_script_key: str = EXTRACT_SCRIPT_KEY,
    extract_output_folder: str | Path = EXTRACT_OUTPUT_FOLDER,
) -> None:
    data_dir = resolved_extract_output_dir(
        extract_script_key=extract_script_key,
        extract_output_folder=extract_output_folder,
    )
    replacements = {
        "SITE_PATH": data_dir / "sites.xlsx",
        "CONC_PATH": data_dir / "concentration.xlsx",
        "WIND_PATH": data_dir / "wind.xlsx",
        "TARGET_POLLUTANT": leak["pollutant"],
    }
    if extracted_paths:
        replacements.update(extracted_paths)

    text = PINN_CONFIG.read_text(encoding="utf-8")
    for name, value in replacements.items():
        text = replace_python_string_assignment(text, name, value)
    PINN_CONFIG.write_text(text, encoding="utf-8")


def update_extract_monitor_inputs(
    leak: dict,
    extract_script: Path,
    monitor_input_path: str | Path,
    extract_output_folder: str | Path,
) -> None:
    text = extract_script.read_text(encoding="utf-8")
    replacements = {
        "START_TIME": leak["start_time"].strftime("%Y-%m-%d %H:%M:%S"),
        "END_TIME": leak["end_time"].strftime("%Y-%m-%d %H:%M:%S"),
        "TARGET_POLLUTANT": leak["pollutant"],
        "WIND_STATION_NAME": leak["wind_station"],
        "OUTPUT_FOLDER": extract_output_folder,
    }

    monitor_input_value = display_path_for_config(monitor_input_path)
    input_patterns = [
        ("INPUT_FILE_PATH", rf'^INPUT_FILE_PATH\s*=\s*(?:r)?["\'].*?["\']\s*$'),
        ("MONITOR_DATA_DIR", rf'^MONITOR_DATA_DIR\s*=\s*(?:r)?["\'].*?["\']\s*$'),
    ]
    for name, pattern in input_patterns:
        replacement = f"{name} = {python_literal(monitor_input_value)}"
        text, count = re.subn(pattern, replacement, text, count=1, flags=re.M)
        if count == 1:
            break
    else:
        raise ValueError(
            f"Could not update INPUT_FILE_PATH or MONITOR_DATA_DIR in {extract_script}"
        )

    for name, value in replacements.items():
        try:
            text = replace_python_string_assignment(text, name, value)
        except ValueError as exc:
            raise ValueError(f"Could not update {name} in {extract_script}") from exc

    extract_script.write_text(text, encoding="utf-8")


def run_step(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("> " + " ".join(command) + "\n\n")
        log.flush()
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["MPLBACKEND"] = "Agg"
        env["PINN_AUTO_CLOSE_PLOTS"] = "1"
        proc = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {proc.returncode}: {command}"
        )


def latest_result_dir(before: set[Path]) -> Path:
    candidates = [
        path for path in RESULT_DIR.iterdir() if path.is_dir() and path not in before
    ]
    if not candidates:
        raise FileNotFoundError(
            "PINN run finished but no new result directory was found."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def run_recent_leak_source_inversions(
    count: int = SOURCE_INVERSION_COUNT,
    direction: str = TRAVERSE_DIRECTION,
    start_time: str | None = START_TRAVERSE_TIME,
    start_rank: int | None = None,
    pollutant_contains: str | None = POLLUTANT_CONTAINS,
    input_file_path: str | Path = INPUT_FILE_PATH,
    monitor_input_path: str | Path = MONITOR_INPUT_PATH,
    extract_script_key: str = EXTRACT_SCRIPT_KEY,
    extract_output_folder: str | Path = EXTRACT_OUTPUT_FOLDER,
) -> Path:
    abnormal_path = resolve_repo_path(input_file_path)
    if not abnormal_path.exists():
        raise FileNotFoundError(f"Abnormal event workbook not found: {abnormal_path}")
    monitor_input = resolve_repo_path(monitor_input_path)
    if not monitor_input.exists():
        raise FileNotFoundError(f"Monitor input path not found: {monitor_input}")
    extract_script = resolve_extract_script(extract_script_key)

    all_leaks = build_leaks(abnormal_path)
    selected_leaks = select_leaks(
        leaks=all_leaks,
        count=count,
        direction=direction,
        start_time=start_time,
        start_rank=start_rank,
        pollutant_contains=pollutant_contains,
    )
    if not selected_leaks:
        raise ValueError(
            "No leaks matched the selection conditions in abnormal workbook: "
            f"{abnormal_path}"
        )

    summary_path = (
        RESULT_DIR / f"recent_leak_run_summary_{pd.Timestamp.now():%Y%m%d_%H%M%S}.xlsx"
    )

    summary_rows = []
    with tempfile.TemporaryDirectory(prefix="recent_leak_logs_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        for sequence_index, (event_rank, leak) in enumerate(selected_leaks, start=1):
            leak_dir = temp_dir / f"leak_{sequence_index:02d}"
            leak_dir.mkdir(parents=True, exist_ok=True)

            summary_rows.append(
                {
                    "run_index": sequence_index,
                    "event_rank": event_rank,
                    "traverse_direction": direction,
                    "start_traverse_time": start_time,
                    "pollutant_contains": pollutant_contains,
                    "pollutant": leak["pollutant"],
                    "leak_start": leak["leak_start"],
                    "leak_end": leak["leak_end"],
                    "extract_start_time": leak["start_time"],
                    "extract_end_time": leak["end_time"],
                    "wind_station": leak["wind_station"],
                    "max_concentration": leak["max_concentration"],
                    "n_hours": leak["n_hours"],
                    "n_records": leak["n_records"],
                }
            )
            pd.DataFrame(summary_rows).to_excel(summary_path, index=False)

            safe_print(
                f"[{sequence_index}/{len(selected_leaks)}] rank={event_rank} "
                f"pollutant={leak['pollutant']} "
                f"window={leak['start_time']} -> {leak['end_time']} "
                f"wind_station={leak['wind_station']}",
            )

            update_extract_monitor_inputs(
                leak,
                extract_script=extract_script,
                monitor_input_path=monitor_input,
                extract_output_folder=extract_output_folder,
            )
            extract_log_path = leak_dir / "extract_monitor_data.log"
            run_step(
                [sys.executable, str(extract_script)],
                extract_log_path,
            )
            extracted_paths = parse_extracted_input_paths(extract_log_path)
            update_pinn_config_inputs(
                leak,
                extracted_paths=extracted_paths,
                extract_script_key=extract_script_key,
                extract_output_folder=extract_output_folder,
            )
            summary_rows[-1]["site_path"] = str(extracted_paths["SITE_PATH"])
            summary_rows[-1]["concentration_path"] = str(extracted_paths["CONC_PATH"])
            summary_rows[-1]["wind_path"] = str(extracted_paths["WIND_PATH"])
            pd.DataFrame(summary_rows).to_excel(summary_path, index=False)
            result_dirs_before = {
                path for path in RESULT_DIR.iterdir() if path.is_dir()
            }
            run_step(
                [sys.executable, str(PINN_SCRIPT)],
                leak_dir / "pinn_source_pinn.log",
            )
            result_dir = latest_result_dir(result_dirs_before)
            for log_name in ("extract_monitor_data.log", "pinn_source_pinn.log"):
                source_log = leak_dir / log_name
                target_log = result_dir / log_name
                if source_log.exists():
                    source_log.replace(target_log)
            summary_rows[-1]["result_dir"] = str(result_dir)
            pd.DataFrame(summary_rows).to_excel(summary_path, index=False)

    pd.DataFrame(summary_rows).to_excel(summary_path, index=False)
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run extraction and PINN inversion for the most recent leak events."
    )
    parser.add_argument(
        "--count",
        type=int,
        default=SOURCE_INVERSION_COUNT,
        help="Number of most recent leak events to run.",
    )
    parser.add_argument(
        "--direction",
        choices=["backward", "forward"],
        default=TRAVERSE_DIRECTION,
        help="Traverse leak events backward or forward in time.",
    )
    parser.add_argument(
        "--start-time",
        default=START_TRAVERSE_TIME,
        help=(
            "Traversal start time, formatted as YYYY-MM-DD HH:MM:SS. "
            "Empty means latest for backward or earliest for forward."
        ),
    )
    parser.add_argument(
        "--start-rank",
        type=int,
        default=None,
        help="Optional 1-based rank after time/direction filtering to start from.",
    )
    parser.add_argument(
        "--pollutant-contains",
        default=POLLUTANT_CONTAINS,
        help=(
            "Only run leak events whose pollutant name contains this text. "
            "Empty means all pollutants."
        ),
    )
    parser.add_argument(
        "--input-file",
        default=str(INPUT_FILE_PATH),
        help=(
            "Abnormal-high event workbook to traverse. Must contain an "
            "`abnormal_high_records` sheet."
        ),
    )
    parser.add_argument(
        "--monitor-input",
        default=str(MONITOR_INPUT_PATH),
        help=(
            "Raw monitor data source for the extraction script. For SHSH-JS this "
            "is the source workbook; for JJJ this is the station-workbook directory."
        ),
    )
    parser.add_argument(
        "--extract-script-key",
        default=EXTRACT_SCRIPT_KEY,
        help=(
            "Extraction script key. Example: shsh_js -> "
            "scripts/extract_monitor_data_shsh_js.py, jjj -> "
            "scripts/extract_monitor_data_jjj.py."
        ),
    )
    parser.add_argument(
        "--extract-output-folder",
        default=str(EXTRACT_OUTPUT_FOLDER),
        help=(
            "Output folder passed to the extraction script. Empty uses that "
            "script's default output directory."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_path = run_recent_leak_source_inversions(
        count=args.count,
        direction=args.direction,
        start_time=args.start_time,
        start_rank=args.start_rank,
        pollutant_contains=args.pollutant_contains,
        input_file_path=args.input_file,
        monitor_input_path=args.monitor_input,
        extract_script_key=args.extract_script_key,
        extract_output_folder=args.extract_output_folder,
    )
    safe_print(f"Saved run summary: {summary_path}")


if __name__ == "__main__":
    main()
