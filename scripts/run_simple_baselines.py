"""Run truth-blind B0/B1 source-location baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PINN_DIR = ROOT / "pinn_source"
if str(PINN_DIR) not in sys.path:
    sys.path.insert(0, str(PINN_DIR))

from simple_baselines import maximum_anomaly_station, maximum_anomaly_upwind  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", required=True, type=Path)
    parser.add_argument("--concentration", required=True, type=Path)
    parser.add_argument("--wind", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--upwind-distance-m", type=float, default=1000.0)
    parser.add_argument("--source-position-pad-m", type=float, default=500.0)
    args = parser.parse_args()
    inputs = {
        "sites": args.sites.resolve(),
        "concentration": args.concentration.resolve(),
        "wind": args.wind.resolve(),
    }
    for path in inputs.values():
        if not path.is_file():
            parser.error(f"input file does not exist: {path}")
    results = [
        maximum_anomaly_station(**{
            "site_path": inputs["sites"],
            "concentration_path": inputs["concentration"],
            "wind_path": inputs["wind"],
        }),
        maximum_anomaly_upwind(
            inputs["sites"],
            inputs["concentration"],
            inputs["wind"],
            distance_m=args.upwind_distance_m,
            source_position_pad_m=args.source_position_pad_m,
        ),
    ]
    payload = {
        "schema_version": 1,
        "analysis": "truth_blind_simple_source_localization_baselines",
        "inputs": {
            key: {"path": str(path), "sha256": _sha256(path)}
            for key, path in inputs.items()
        },
        "truth_isolation": "this runner does not accept or read source truth",
        "development_notice": (
            "B1 distance is a development candidate until frozen across the full "
            "development set before test evaluation."
        ),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"output={args.output.resolve()}")
    print(f"sha256={_sha256(args.output)}")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
