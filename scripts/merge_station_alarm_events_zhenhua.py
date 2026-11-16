"""Merge alarms by pollutant and time, independently of station."""

from pathlib import Path
import math
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# Manual inputs. A two-hour gap permits one missing hourly alarm.
INPUT_FILE = (
    ROOT
    / "data/zhenhua1112/振华前三季度数据/振华第二季度数据/2026年第二季度空气站报警记录.xlsx"
)
OUTPUT_FILE = ROOT / "data/zhenhua1112/merged_alarm_events_q2.xlsx"
MAX_GAP_HOURS = 2.0
MIN_CONCENTRATION = 500  # None: no filter; e.g. 200.0, in the alarm workbook's units.
SKIP_POLLUTANTS = ["VOCS-36"]
MANUAL_INPUT = []

REFERENCE_COLUMNS = [
    "station",
    "pollutant",
    "time",
    "event_start",
    "event_end",
    "concentration",
]


def merge_alarms(frame, max_gap_hours=MAX_GAP_HOURS, min_concentration=None):
    required = ["报警产生时间", "站点", "污染物", "污染物浓度"]
    if missing := set(required) - set(frame.columns):
        raise ValueError(f"Missing alarm columns: {sorted(missing)}")
    if not 0 < max_gap_hours < float("inf"):
        raise ValueError("max_gap_hours must be positive and finite")
    if min_concentration is not None:
        min_concentration = float(min_concentration)
        if not math.isfinite(min_concentration) or min_concentration < 0:
            raise ValueError(
                "min_concentration must be nonnegative and finite, or None"
            )
    data = frame.copy()
    data["报警产生时间"] = pd.to_datetime(data["报警产生时间"], errors="raise")
    data["污染物浓度"] = pd.to_numeric(data["污染物浓度"], errors="raise")
    if data["污染物浓度"].isna().any():
        raise ValueError("Missing alarm concentration")
    if data[required[:3]].isna().any().any():
        raise ValueError("Missing alarm time, station or pollutant")
    for key in ["站点", "污染物"]:
        data[key] = data[key].astype(str).str.strip()
        if data[key].eq("").any():
            raise ValueError(f"Empty {key}")
    rows = []
    for pollutant, group in data.groupby("污染物", sort=True):
        if pollutant in SKIP_POLLUTANTS or (
            MANUAL_INPUT and pollutant not in MANUAL_INPUT
        ):
            continue
        group = group.sort_values("报警产生时间", kind="stable")
        blocks = (
            group["报警产生时间"].diff().gt(pd.Timedelta(hours=max_gap_hours)).cumsum()
        )
        for _, block in group.groupby(blocks):
            start, end = block["报警产生时间"].iloc[[0, -1]]
            middle = start + (end - start) / 2
            # Prefer later midpoint ties, then the largest concentration at that time.
            ranked = block.assign(_distance=(block["报警产生时间"] - middle).abs())
            chosen = ranked.sort_values(
                ["_distance", "报警产生时间", "污染物浓度", "站点"],
                ascending=[True, False, False, True],
                kind="stable",
            ).iloc[0]
            # Filter completed events, not raw alarms: preserve their original boundaries.
            if (
                min_concentration is not None
                and chosen["污染物浓度"] < min_concentration
            ):
                continue
            row = dict.fromkeys(REFERENCE_COLUMNS)
            row.update(
                station=chosen["站点"],
                pollutant=pollutant,
                time=chosen["报警产生时间"],
                concentration=pd.to_numeric(chosen["污染物浓度"], errors="raise"),
                event_start=start,
                event_end=end,
            )
            rows.append(row)
    return (
        pd.DataFrame(rows, columns=REFERENCE_COLUMNS)
        .sort_values(["time", "station", "pollutant"])
        .reset_index(drop=True)
    )


def main():
    frame = pd.read_excel(INPUT_FILE)
    merged = merge_alarms(frame, min_concentration=MIN_CONCENTRATION)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    merged.to_excel(OUTPUT_FILE, sheet_name="abnormal_high_records", index=False)
    print(f"{len(frame)} alarms -> {len(merged)} events: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
