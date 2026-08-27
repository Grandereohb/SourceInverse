"""Run and freeze truth-blind Gaussian-puff candidate surfaces over a matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_gaussian_puff_baseline.py"
CODE_FILES = (
    "pinn_source/data_io.py",
    "pinn_source/gaussian_puff_baseline.py",
    "pinn_source/synthetic_puff.py",
    "scripts/run_gaussian_puff_baseline.py",
    "scripts/run_gaussian_surface_matrix.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _code_snapshot() -> dict:
    return {relative: _sha256(ROOT / relative) for relative in CODE_FILES}


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--diffusivity-m2s", type=float, default=2.0)
    parser.add_argument("--initial-sigma-m", type=float, default=334.713)
    parser.add_argument("--decay-per-hour", type=float, default=0.5)
    parser.add_argument("--wind-factor", type=float, default=0.25)
    parser.add_argument("--grid-size", type=int, default=17)
    parser.add_argument("--refinement-levels", type=int, default=3)
    parser.add_argument("--dynamic-q-node-count", type=int, default=5)
    parser.add_argument(
        "--physics-from-scenario",
        action="store_true",
        help=(
            "Use truth-physics fields from each synthetic scenario manifest. "
            "This is a truth-informed sensitivity analysis, not a deployable baseline."
        ),
    )
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    batch = json.loads(args.batch_manifest.read_text(encoding="utf-8"))
    args.output_root.mkdir(parents=True, exist_ok=True)
    comparable_plan = {
        "schema_version": 1,
        "experiment_type": (
            "truth_informed_matched_physics_gaussian_sensitivity"
            if args.physics_from_scenario
            else "truth_blind_gaussian_candidate_surface_matrix"
        ),
        "batch_manifest": {
            "path": str(args.batch_manifest.resolve()),
            "sha256": _sha256(args.batch_manifest),
        },
        "code_snapshot": _code_snapshot(),
        "settings": {
            "diffusivity_m2s": args.diffusivity_m2s,
            "initial_sigma_m": args.initial_sigma_m,
            "decay_per_hour": args.decay_per_hour,
            "wind_factor": args.wind_factor,
            "grid_size": args.grid_size,
            "refinement_levels": args.refinement_levels,
            "dynamic_q_node_count": args.dynamic_q_node_count,
            "save_candidates": True,
            "physics_from_scenario": bool(args.physics_from_scenario),
        },
        "truth_isolation": (
            "runner reads scenario truth physics but does not use source coordinates; not deployable"
            if args.physics_from_scenario
            else "runner resolves input paths but never reads source coordinates"
        ),
    }
    plan_path = args.output_root / "experiment_plan.json"
    if plan_path.exists():
        existing = json.loads(plan_path.read_text(encoding="utf-8"))
        existing.pop("created_at_utc", None)
        if existing != comparable_plan:
            raise ValueError("existing Gaussian surface plan does not match invocation")
    else:
        plan = dict(comparable_plan)
        plan["created_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json(plan_path, plan)
    print(f"plan={plan_path.resolve()}")
    print(f"plan_sha256={_sha256(plan_path)}")
    if args.plan_only:
        return

    outputs = []
    for entry in batch["scenarios"]:
        scenario_manifest_path = Path(entry["manifest_path"])
        scenario = json.loads(scenario_manifest_path.read_text(encoding="utf-8"))
        scenario_id = scenario["scenario_id"]
        if args.physics_from_scenario:
            physics = scenario["physics"]
            diffusivity_m2s = float(physics["diffusivity_m2s"])
            initial_sigma_m = float(physics["initial_sigma_m"])
            decay_per_hour = float(physics["decay_per_hour"])
            wind_factor = float(physics["truth_wind_factor"])
            parameter_source = "truth_informed_matched_physics_sensitivity"
        else:
            diffusivity_m2s = args.diffusivity_m2s
            initial_sigma_m = args.initial_sigma_m
            decay_per_hour = args.decay_per_hour
            wind_factor = args.wind_factor
            parameter_source = "development_candidate"
        output = args.output_root / scenario_id / "gaussian_candidate_surfaces.json"
        if not output.is_file():
            output.parent.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                str(RUNNER),
                "--sites", str(scenario["files"]["sites"]["path"]),
                "--concentration", str(scenario["files"]["concentration"]["path"]),
                "--wind", str(scenario["files"]["wind"]["path"]),
                "--output", str(output),
                "--diffusivity-m2s", str(diffusivity_m2s),
                "--initial-sigma-m", str(initial_sigma_m),
                "--decay-per-hour", str(decay_per_hour),
                "--wind-factor", str(wind_factor),
                "--grid-size", str(args.grid_size),
                "--refinement-levels", str(args.refinement_levels),
                "--dynamic-q-node-count", str(args.dynamic_q_node_count),
                "--parameter-source", parameter_source,
                "--save-candidates",
            ]
            subprocess.run(command, cwd=ROOT, check=True)
        else:
            print(f"skip_existing scenario={scenario_id}")
        outputs.append(
            {
                "scenario_id": scenario_id,
                "input_id": entry.get("input_id"),
                "design_factors": entry.get("design_factors", {}),
                "scenario_manifest_path": str(scenario_manifest_path.resolve()),
                "scenario_manifest_sha256": _sha256(scenario_manifest_path),
                "surface_path": str(output.resolve()),
                "surface_sha256": _sha256(output),
            }
        )
    status = {
        "schema_version": 1,
        "experiment_plan_sha256": _sha256(plan_path),
        "scenario_count": len(outputs),
        "outputs": outputs,
    }
    status_path = args.output_root / "execution_status.json"
    _write_json(status_path, status)
    print(f"status={status_path.resolve()}")
    print(f"status_sha256={_sha256(status_path)}")


if __name__ == "__main__":
    main()
