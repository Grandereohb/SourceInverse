from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PINN_DIR = REPO_ROOT / "pinn_source"
if str(PINN_DIR) not in sys.path:
    sys.path.insert(0, str(PINN_DIR))

from pipeline import run  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one isolated source inversion job")
    parser.add_argument("--sites", required=True)
    parser.add_argument("--concentration", required=True)
    parser.add_argument("--wind", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--pollutant", required=True)
    parser.add_argument("--result-json", required=True)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--make-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(
        site_path=args.sites,
        conc_path=args.concentration,
        wind_path=args.wind,
        random_seed=args.random_seed,
        output_dir=args.output_root,
        result_root_dir=args.result_root,
        make_plots=args.make_plots,
        result_name_suffix=args.pollutant,
    )
    result_path = Path(args.result_json)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
