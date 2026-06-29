from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_DIR = REPO_ROOT / "data"
RESULT_DIR = REPO_ROOT / "result"

ABNORMAL_DIR = DATA_DIR / "abnormal_high_monitor_data"
PINN_SCRIPT = REPO_ROOT / "pinn_source" / "pinn_source_pinn.py"

INPUT_FILE_PATH = (
    DATA_DIR
    / "shsh_js"
    / "自动审核小时数据_标准单位_2025-10-16 00_00_00_2026-04-16 12_00_00.xlsx"
)
OUTPUT_FOLDER = "shsh_js"


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


EXTRACT_SCRIPT = resolve_extract_script(OUTPUT_FOLDER)

# =========================
# Manual Inputs
# =========================
# SOURCE_INVERSION_COUNT: number of leak events to run in sequence.
SOURCE_INVERSION_COUNT = 5

# TRAVERSE_DIRECTION:
# - "backward": traverse leak events from later time to earlier time.
# - "forward": traverse leak events from earlier time to later time.
TRAVERSE_DIRECTION = "backward"

# START_TRAVERSE_TIME:
# - empty string: start from latest leak for "backward", earliest leak for "forward".
# - "YYYY-MM-DD HH:MM:SS": start traversing from this time.
START_TRAVERSE_TIME = ""


def safe_text(value) -> str:
    return str(value).encode("gbk", errors="backslashreplace").decode("gbk")


def safe_print(message: str) -> None:
    print(safe_text(message), flush=True)


def latest_abnormal_file() -> Path:
    files = [
        path
        for path in ABNORMAL_DIR.glob("*.xlsx")
        if path.is_file() and not path.name.startswith("~$")
    ]
    if not files:
        raise FileNotFoundError(f"No abnormal result workbook found under: {ABNORMAL_DIR}")
    return max(files, key=lambda path: path.stat().st_mtime)


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


def update_extract_monitor_inputs(leak: dict) -> None:
    text = EXTRACT_SCRIPT.read_text(encoding="utf-8")
    replacements = {
        "INPUT_FILE_PATH": INPUT_FILE_PATH.relative_to(REPO_ROOT).as_posix(),
        "START_TIME": leak["start_time"].strftime("%Y-%m-%d %H:%M:%S"),
        "END_TIME": leak["end_time"].strftime("%Y-%m-%d %H:%M:%S"),
        "TARGET_POLLUTANT": leak["pollutant"],
        "WIND_STATION_NAME": leak["wind_station"],
        "OUTPUT_FOLDER": OUTPUT_FOLDER,
    }

    for name, value in replacements.items():
        pattern = rf'^{name}\s*=\s*(?:r)?["\'].*?["\']\s*$'
        replacement = f"{name} = {python_literal(value)}"
        text, count = re.subn(pattern, replacement, text, count=1, flags=re.M)
        if count != 1:
            raise ValueError(f"Could not update {name} in {EXTRACT_SCRIPT}")

    EXTRACT_SCRIPT.write_text(text, encoding="utf-8")


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
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {command}")


def latest_result_dir(before: set[Path]) -> Path:
    candidates = [
        path
        for path in RESULT_DIR.iterdir()
        if path.is_dir() and path not in before
    ]
    if not candidates:
        raise FileNotFoundError("PINN run finished but no new result directory was found.")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def run_recent_leak_source_inversions(
    count: int = SOURCE_INVERSION_COUNT,
    direction: str = TRAVERSE_DIRECTION,
    start_time: str | None = START_TRAVERSE_TIME,
    start_rank: int | None = None,
) -> Path:
    abnormal_path = latest_abnormal_file()
    all_leaks = build_leaks(abnormal_path)
    selected_leaks = select_leaks(
        leaks=all_leaks,
        count=count,
        direction=direction,
        start_time=start_time,
        start_rank=start_rank,
    )
    if not selected_leaks:
        raise ValueError(f"No leaks found in abnormal workbook: {abnormal_path}")

    run_dir = RESULT_DIR / f"recent_leak_runs_{pd.Timestamp.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for sequence_index, (event_rank, leak) in enumerate(selected_leaks, start=1):
        leak_dir = run_dir / f"leak_{sequence_index:02d}"
        leak_dir.mkdir(parents=True, exist_ok=True)

        summary_rows.append(
            {
                "run_index": sequence_index,
                "event_rank": event_rank,
                "traverse_direction": direction,
                "start_traverse_time": start_time,
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
        pd.DataFrame(summary_rows).to_excel(run_dir / "run_summary.xlsx", index=False)

        safe_print(
            f"[{sequence_index}/{len(selected_leaks)}] rank={event_rank} "
            f"pollutant={leak['pollutant']} "
            f"window={leak['start_time']} -> {leak['end_time']} "
            f"wind_station={leak['wind_station']}",
        )

        update_extract_monitor_inputs(leak)
        run_step(
            [sys.executable, str(EXTRACT_SCRIPT)],
            leak_dir / "extract_monitor_data.log",
        )
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
        pd.DataFrame(summary_rows).to_excel(run_dir / "run_summary.xlsx", index=False)

    pd.DataFrame(summary_rows).to_excel(run_dir / "run_summary.xlsx", index=False)
    return run_dir


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = run_recent_leak_source_inversions(
        count=args.count,
        direction=args.direction,
        start_time=args.start_time,
        start_rank=args.start_rank,
    )
    safe_print(f"Saved run logs and summary: {run_dir}")


if __name__ == "__main__":
    main()
