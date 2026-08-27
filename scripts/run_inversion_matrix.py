"""Run a frozen, truth-blind inversion design over every scenario in a batch."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_frozen_multistart.py"
CODE_SNAPSHOT_RELATIVE_PATHS = (
    "pinn_source/config.py",
    "pinn_source/data_io.py",
    "pinn_source/field.py",
    "pinn_source/models/pinn.py",
    "pinn_source/pipeline.py",
    "pinn_source/q_parameterization.py",
    "pinn_source/run_artifacts.py",
    "scripts/run_frozen_multistart.py",
    "scripts/run_inversion_matrix.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _code_snapshot(root: Path = ROOT) -> dict:
    snapshot = {}
    for relative in CODE_SNAPSHOT_RELATIVE_PATHS:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"code snapshot file does not exist: {path}")
        snapshot[relative] = _sha256(path)
    return snapshot


def _verify_code_snapshot(expected: dict, root: Path = ROOT) -> None:
    current = _code_snapshot(root)
    if current != expected:
        changed = sorted(
            key
            for key in set(expected) | set(current)
            if expected.get(key) != current.get(key)
        )
        raise RuntimeError(
            "Relevant code changed after the matrix plan was frozen: "
            + ", ".join(changed)
        )


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _comparable_plan(plan: dict) -> dict:
    value = json.loads(json.dumps(plan))
    value.pop("created_at_utc", None)
    value.setdefault("q_mode", "smooth_time")
    value.setdefault("source_position_mode", "single")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-manifest", required=True, type=Path)
    parser.add_argument("--design", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--solvers", nargs="+", default=["production"], choices=["production", "characteristic"]
    )
    parser.add_argument("--epochs", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--q-mode",
        choices=["constant", "neural", "piecewise", "smooth_time"],
        default="smooth_time",
    )
    parser.add_argument(
        "--source-position-mode", choices=["single", "fixed"], default="single"
    )
    parser.add_argument("--wind-scale", type=float, default=0.25)
    parser.add_argument("--decay-per-hour", type=float, default=0.5)
    parser.add_argument("--d-min-m2s", type=float, default=1.0)
    parser.add_argument("--sigma-src-norm", type=float, default=0.05)
    parser.add_argument(
        "--wind-vector-smoothing", choices=["enabled", "disabled"], default="enabled"
    )
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    for path in (args.batch_manifest, args.design):
        if not path.is_file():
            parser.error(f"input file does not exist: {path}")

    batch = json.loads(args.batch_manifest.read_text(encoding="utf-8"))
    scenarios = batch.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        parser.error("batch manifest needs a non-empty scenarios list")
    scenario_ids = [str(entry.get("scenario_id", "")).strip() for entry in scenarios]
    if any(not value for value in scenario_ids) or len(scenario_ids) != len(set(scenario_ids)):
        parser.error("batch manifest scenario IDs must be non-empty and unique")
    args.output_root.mkdir(parents=True, exist_ok=True)

    scenario_inputs = []
    for entry in scenarios:
        manifest_path = Path(entry["manifest_path"])
        actual_hash = _sha256(manifest_path)
        expected_hash = str(entry["manifest_sha256"]).upper()
        if actual_hash != expected_hash:
            raise ValueError(f"scenario manifest hash mismatch: {manifest_path}")
        scenario = json.loads(manifest_path.read_text(encoding="utf-8"))
        if str(scenario.get("scenario_id")) != str(entry["scenario_id"]):
            raise ValueError(f"scenario ID mismatch: {manifest_path}")
        files = scenario["files"]
        scenario_inputs.append(
            {
                "scenario_id": str(entry["scenario_id"]),
                "manifest_path": str(manifest_path.resolve()),
                "manifest_sha256": actual_hash,
                "sites": str(Path(files["sites"]["path"]).resolve()),
                "concentration": str(Path(files["concentration"]["path"]).resolve()),
                "wind": str(Path(files["wind"]["path"]).resolve()),
            }
        )

    plan = {
        "schema_version": 1,
        "experiment_type": "truth_blind_frozen_inversion_matrix",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "batch_manifest": {
            "path": str(args.batch_manifest.resolve()),
            "sha256": _sha256(args.batch_manifest),
        },
        "design": {"path": str(args.design.resolve()), "sha256": _sha256(args.design)},
        "code_snapshot": _code_snapshot(),
        "scenario_count": len(scenario_inputs),
        "retention_rule": "run and retain every scenario listed in the source batch manifest",
        "truth_isolation": (
            "scenario manifests are opened only to resolve observation input paths; source "
            "coordinates are neither forwarded to nor used by the inversion runner"
        ),
        "solvers": args.solvers,
        "epochs": args.epochs,
        "seed": args.seed,
        "q_mode": args.q_mode,
        "source_position_mode": args.source_position_mode,
        "physics_settings": {
            "wind_scale": args.wind_scale,
            "decay_per_hour": args.decay_per_hour,
            "d_min_m2s": args.d_min_m2s,
            "sigma_src_norm": args.sigma_src_norm,
            "wind_vector_smoothing": args.wind_vector_smoothing,
        },
        "scenarios": scenario_inputs,
    }
    plan_path = args.output_root / "matrix_experiment_plan.json"
    if plan_path.exists():
        existing = json.loads(plan_path.read_text(encoding="utf-8"))
        if _comparable_plan(existing) != _comparable_plan(plan):
            raise ValueError("existing matrix experiment plan differs from requested plan")
    else:
        _write_json(plan_path, plan)
    print(f"matrix_plan={plan_path.resolve()}")
    print(f"matrix_plan_sha256={_sha256(plan_path)}")
    if args.plan_only:
        print("plan_only=true")
        return

    progress_path = args.output_root / "matrix_progress.json"
    progress = {"completed": [], "failed": []}
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
    completed = set(progress.get("completed", []))
    failed = {row["scenario_id"]: row for row in progress.get("failed", [])}

    for row in scenario_inputs:
        _verify_code_snapshot(plan["code_snapshot"])
        scenario_id = row["scenario_id"]
        if scenario_id in completed:
            print(f"skip_completed scenario={scenario_id}")
            continue
        command = [
            sys.executable,
            str(RUNNER),
            "--sites", row["sites"],
            "--concentration", row["concentration"],
            "--wind", row["wind"],
            "--design", str(args.design.resolve()),
            "--output-root", str((args.output_root / scenario_id).resolve()),
            "--solvers", *args.solvers,
            "--epochs", str(args.epochs),
            "--seed", str(args.seed),
            "--q-mode", args.q_mode,
            "--source-position-mode", args.source_position_mode,
            "--wind-scale", str(args.wind_scale),
            "--decay-per-hour", str(args.decay_per_hour),
            "--d-min-m2s", str(args.d_min_m2s),
            "--sigma-src-norm", str(args.sigma_src_norm),
            "--wind-vector-smoothing", args.wind_vector_smoothing,
        ]
        if args.continue_on_error:
            command.append("--continue-on-error")
        try:
            subprocess.run(command, cwd=ROOT, check=True)
        except subprocess.CalledProcessError as exc:
            failed[scenario_id] = {"scenario_id": scenario_id, "returncode": exc.returncode}
            _write_json(
                progress_path,
                {"completed": sorted(completed), "failed": list(failed.values())},
            )
            if not args.continue_on_error:
                raise
        else:
            completed.add(scenario_id)
            failed.pop(scenario_id, None)
            _write_json(
                progress_path,
                {"completed": sorted(completed), "failed": list(failed.values())},
            )

    print(f"completed={len(completed)}/{len(scenario_inputs)}")
    print(f"failed={len(failed)}")


if __name__ == "__main__":
    main()
