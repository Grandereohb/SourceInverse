"""Summarize the predeclared warm-start local-profile mechanism pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path


RING_RE = re.compile(r"^r(?P<radius>\d{4})_[a-z0-9]+$")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def summarize_scenario(csv_path: Path, scenario_id: str, mechanism: str) -> dict:
    rows = _read_rows(csv_path)
    selected_rows = [row for row in rows if row["candidate_id"] == "selected"]
    if len(selected_rows) != 1:
        raise ValueError(f"{scenario_id} needs exactly one selected profile row")
    selected_loss = float(selected_rows[0]["best_raw_loss"])
    best = min(rows, key=lambda row: float(row["best_raw_loss"]))
    ring_groups: dict[int, list[float]] = {}
    for row in rows:
        match = RING_RE.match(row["candidate_id"])
        if match:
            radius = int(match.group("radius"))
            relative = (float(row["best_raw_loss"]) - selected_loss) / abs(
                selected_loss
            )
            ring_groups.setdefault(radius, []).append(relative)
    rings = {}
    for radius, values in sorted(ring_groups.items()):
        ordered = sorted(values)
        middle = len(ordered) // 2
        median = (
            ordered[middle]
            if len(ordered) % 2
            else (ordered[middle - 1] + ordered[middle]) / 2.0
        )
        rings[str(radius)] = {
            "direction_count": len(values),
            "minimum_relative_rise": min(values),
            "median_relative_rise": median,
            "maximum_relative_rise": max(values),
            "within_selected_5_percent_count": sum(value <= 0.05 for value in values),
        }
    required_rings = [rings.get("250"), rings.get("500")]
    if any(ring is None for ring in required_rings):
        raise ValueError(f"{scenario_id} is missing the 250 m or 500 m ring")
    passes_local_sharpness_rule = all(
        ring["minimum_relative_rise"] > 0.05 for ring in required_rings
    )
    return {
        "scenario_id": scenario_id,
        "mechanism": mechanism,
        "candidate_csv_path": str(csv_path.resolve()),
        "candidate_csv_sha256": _sha256(csv_path),
        "candidate_count": len(rows),
        "selected_loss": selected_loss,
        "best_candidate_id": best["candidate_id"],
        "best_loss": float(best["best_raw_loss"]),
        "best_relative_change_from_selected": (
            float(best["best_raw_loss"]) - selected_loss
        )
        / abs(selected_loss),
        "selected_is_profile_minimum": best["candidate_id"] == "selected",
        "rings": rings,
        "passes_local_sharpness_rule": passes_local_sharpness_rule,
    }


def summarize_pilot(config_path: Path, output_root: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    rows = []
    not_run = []
    for scenario in config["scenario_selection"]:
        scenario_id = scenario["scenario_id"]
        csv_path = output_root / scenario_id / "fixed_source_profile_candidates.csv"
        if not csv_path.is_file():
            not_run.append(scenario_id)
            continue
        rows.append(
            summarize_scenario(csv_path, scenario_id, scenario["mechanism"])
        )
    success = [row for row in rows if row["mechanism"] == "selected_success"]
    failures = [row for row in rows if row["mechanism"] != "selected_success"]
    false_reassurance = [
        row["scenario_id"] for row in failures if row["passes_local_sharpness_rule"]
    ]
    promotion_pass = (
        len(success) == 1
        and success[0]["passes_local_sharpness_rule"]
        and bool(failures)
        and not false_reassurance
    )
    return {
        "schema_version": 1,
        "analysis": "warm_start_local_profile_mechanism_pilot",
        "config_path": str(config_path.resolve()),
        "config_sha256": _sha256(config_path),
        "output_root": str(output_root.resolve()),
        "completed_scenario_count": len(rows),
        "not_run_scenarios": not_run,
        "local_sharpness_rule": "selected is locally sharp when the minimum relative profile-loss rise exceeds 5 percent on both the 250 m and 500 m rings",
        "false_reassurance_failure_scenarios": false_reassurance,
        "promotion_pass": promotion_pass,
        "decision": (
            "promote_to_full_development"
            if promotion_pass
            else "do_not_promote_local_profile_as_standalone_confidence"
        ),
        "rows": rows,
        "interpretation": (
            "This development mechanism pilot is not a calibration or test estimate. "
            "A failed promotion rule is retained as a negative ablation."
        ),
    }


def write_csv(path: Path, payload: dict) -> None:
    fields = [
        "scenario_id",
        "mechanism",
        "candidate_count",
        "selected_loss",
        "best_candidate_id",
        "best_loss",
        "best_relative_change_from_selected",
        "selected_is_profile_minimum",
        "ring_250_minimum_relative_rise",
        "ring_250_median_relative_rise",
        "ring_250_within_5_percent_count",
        "ring_500_minimum_relative_rise",
        "ring_500_median_relative_rise",
        "ring_500_within_5_percent_count",
        "passes_local_sharpness_rule",
        "candidate_csv_sha256",
        "candidate_csv_path",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in payload["rows"]:
            writer.writerow(
                {
                    **{key: row.get(key) for key in fields},
                    "ring_250_minimum_relative_rise": row["rings"]["250"][
                        "minimum_relative_rise"
                    ],
                    "ring_250_median_relative_rise": row["rings"]["250"][
                        "median_relative_rise"
                    ],
                    "ring_250_within_5_percent_count": row["rings"]["250"][
                        "within_selected_5_percent_count"
                    ],
                    "ring_500_minimum_relative_rise": row["rings"]["500"][
                        "minimum_relative_rise"
                    ],
                    "ring_500_median_relative_rise": row["rings"]["500"][
                        "median_relative_rise"
                    ],
                    "ring_500_within_5_percent_count": row["rings"]["500"][
                        "within_selected_5_percent_count"
                    ],
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    args = parser.parse_args()
    payload = summarize_pilot(args.config, args.output_root)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(args.output_csv, payload)
    print(f"json={args.output_json.resolve()}")
    print(f"json_sha256={_sha256(args.output_json)}")
    print(f"csv={args.output_csv.resolve()}")
    print(f"csv_sha256={_sha256(args.output_csv)}")
    print(f"decision={payload['decision']}")


if __name__ == "__main__":
    main()
