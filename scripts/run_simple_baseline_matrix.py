"""Freeze and run truth-blind geometric baselines over a scenario batch."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_simple_baselines.py"
CODE_FILES = (
    "pinn_source/data_io.py",
    "pinn_source/simple_baselines.py",
    "scripts/run_simple_baselines.py",
    "scripts/run_simple_baseline_matrix.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--upwind-distance-m", type=float, default=1000.0)
    args = parser.parse_args()
    batch = json.loads(args.batch_manifest.read_text(encoding="utf-8"))
    args.output_root.mkdir(parents=True, exist_ok=True)
    comparable = {
        "schema_version": 1,
        "experiment_type": "truth_blind_geometric_baseline_matrix",
        "batch_manifest": {
            "path": str(args.batch_manifest.resolve()),
            "sha256": _sha256(args.batch_manifest),
        },
        "code_snapshot": {relative: _sha256(ROOT / relative) for relative in CODE_FILES},
        "settings": {"upwind_distance_m": float(args.upwind_distance_m)},
        "truth_isolation": "estimators receive sites, concentrations, and wind but no source truth",
    }
    plan_path = args.output_root / "experiment_plan.json"
    if plan_path.exists():
        existing = json.loads(plan_path.read_text(encoding="utf-8"))
        existing.pop("created_at_utc", None)
        if existing != comparable:
            raise ValueError("existing plan does not match invocation")
    else:
        payload = dict(comparable)
        payload["created_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write(plan_path, payload)

    outputs = []
    for entry in batch["scenarios"]:
        scenario_path = Path(entry["manifest_path"])
        scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
        output = args.output_root / scenario["scenario_id"] / "simple_baselines.json"
        if not output.exists():
            output.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    sys.executable,
                    str(RUNNER),
                    "--sites", str(scenario["files"]["sites"]["path"]),
                    "--concentration", str(scenario["files"]["concentration"]["path"]),
                    "--wind", str(scenario["files"]["wind"]["path"]),
                    "--output", str(output),
                    "--upwind-distance-m", str(args.upwind_distance_m),
                ],
                cwd=ROOT,
                check=True,
            )
        outputs.append(
            {
                "scenario_id": scenario["scenario_id"],
                "input_id": entry.get("input_id"),
                "design_factors": entry.get("design_factors", {}),
                "scenario_manifest_path": str(scenario_path.resolve()),
                "scenario_manifest_sha256": _sha256(scenario_path),
                "result_path": str(output.resolve()),
                "result_sha256": _sha256(output),
            }
        )
    status = {
        "schema_version": 1,
        "experiment_plan_sha256": _sha256(plan_path),
        "scenario_count": len(outputs),
        "outputs": outputs,
    }
    status_path = args.output_root / "execution_status.json"
    _write(status_path, status)
    print(f"plan_sha256={_sha256(plan_path)}")
    print(f"status_sha256={_sha256(status_path)}")


if __name__ == "__main__":
    main()
