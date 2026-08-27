"""Run a predeclared multistart inversion design with resumable provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PINN_DIR = ROOT / "pinn_source"
if str(PINN_DIR) not in sys.path:
    sys.path.insert(0, str(PINN_DIR))

import config as runtime_config  # noqa: E402
from data_io import load_sites  # noqa: E402
import pipeline  # noqa: E402


ALLOWED_SOLVERS = {"production", "characteristic"}
CODE_SNAPSHOT_RELATIVE_PATHS = (
    "pinn_source/config.py",
    "pinn_source/data_io.py",
    "pinn_source/field.py",
    "pinn_source/models/pinn.py",
    "pinn_source/pipeline.py",
    "pinn_source/q_parameterization.py",
    "pinn_source/run_artifacts.py",
    "scripts/run_frozen_multistart.py",
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
            "Relevant code changed after the experiment plan was frozen: "
            + ", ".join(changed)
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_design(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("Only multistart design schema_version=1 is supported.")
    starts = payload.get("starts")
    if not isinstance(starts, list) or not starts:
        raise ValueError("The design must contain a non-empty starts list.")
    ids = [str(start.get("id", "")).strip() for start in starts]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("Every start needs a unique non-empty id.")
    for start in starts:
        candidate_id = str(start.get("candidate_id", start["id"])).strip()
        nuisance_start_id = str(start.get("nuisance_start_id", "base")).strip()
        if not candidate_id or not nuisance_start_id:
            raise ValueError(
                f"{start['id']} needs non-empty candidate_id and nuisance_start_id."
            )
        seed_offset = start.get("seed_offset", 0)
        if isinstance(seed_offset, bool) or int(seed_offset) != seed_offset:
            raise ValueError(f"{start['id']} seed_offset must be an integer.")
        if int(seed_offset) < 0:
            raise ValueError(f"{start['id']} seed_offset must be non-negative.")
        mode = start.get("mode")
        if mode == "pipeline_heuristic":
            continue
        if mode != "domain_fraction":
            raise ValueError(f"Unsupported start mode: {mode!r}")
        for key in ("fraction_x", "fraction_y"):
            value = float(start[key])
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{start['id']} {key} must be in [0, 1].")
    return payload


def source_domain_bounds(site_path: Path) -> dict:
    sites, _, _ = load_sites(site_path)
    pad = float(runtime_config.SOURCE_POSITION_PAD_M)
    return {
        "x_min_m": float(sites["x"].min() - pad),
        "x_max_m": float(sites["x"].max() + pad),
        "y_min_m": float(sites["y"].min() - pad),
        "y_max_m": float(sites["y"].max() + pad),
        "source_position_pad_m": pad,
    }


def resolve_starts(design: dict, bounds: dict) -> list[dict]:
    width = bounds["x_max_m"] - bounds["x_min_m"]
    height = bounds["y_max_m"] - bounds["y_min_m"]
    resolved = []
    for start in design["starts"]:
        row = {
            "id": start["id"],
            "candidate_id": str(start.get("candidate_id", start["id"])),
            "nuisance_start_id": str(start.get("nuisance_start_id", "base")),
            "seed_offset": int(start.get("seed_offset", 0)),
            "mode": start["mode"],
        }
        if start["mode"] == "pipeline_heuristic":
            row["source_init_override_m"] = None
        else:
            row["fraction_x"] = float(start["fraction_x"])
            row["fraction_y"] = float(start["fraction_y"])
            row["source_init_override_m"] = [
                bounds["x_min_m"] + row["fraction_x"] * width,
                bounds["y_min_m"] + row["fraction_y"] * height,
            ]
        resolved.append(row)
    return resolved


def _completed_keys(output_root: Path) -> set[tuple[str, str]]:
    completed = set()
    for path in output_root.rglob("run_manifest.json"):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            runtime = manifest["runtime"]
            completed.add((str(runtime["recurrent_solver"]), str(runtime["run_id"])))
        except (OSError, KeyError, TypeError, json.JSONDecodeError):
            continue
    return completed


@contextmanager
def _temporary_environment(values: dict[str, str]):
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def prepare_plan(args, design: dict, resolved_starts: list[dict], bounds: dict) -> dict:
    input_paths = {
        "sites": args.sites.resolve(),
        "concentration": args.concentration.resolve(),
        "wind": args.wind.resolve(),
    }
    return {
        "schema_version": 1,
        "experiment_type": "frozen_multistart_source_inversion",
        "created_at_utc": _utc_now(),
        "design": {
            "path": str(args.design.resolve()),
            "sha256": _sha256(args.design),
            "design_id": design.get("design_id"),
        },
        "inputs": {
            key: {"path": str(path), "sha256": _sha256(path)}
            for key, path in input_paths.items()
        },
        "code_snapshot": _code_snapshot(),
        "source_domain_m": bounds,
        "resolved_starts": resolved_starts,
        "solvers": args.solvers,
        "epochs": int(args.epochs),
        "random_seed": int(args.seed),
        "make_plots": bool(args.make_plots),
        "q_mode": str(args.q_mode),
        "source_position_mode": str(args.source_position_mode),
        "event_window_crop": str(args.event_window_crop),
        "initial_checkpoint": (
            {
                "path": str(args.initial_checkpoint.resolve()),
                "sha256": _sha256(args.initial_checkpoint),
            }
            if args.initial_checkpoint is not None
            else None
        ),
        "physics_settings": {
            "wind_scale": float(args.wind_scale),
            "decay_per_hour": float(args.decay_per_hour),
            "d_min_m2s": float(args.d_min_m2s),
            "sigma_src_norm": float(args.sigma_src_norm),
            "wind_vector_smoothing": str(args.wind_vector_smoothing),
        },
        "selection_rule": design.get("selection_rule", "minimum best_raw_loss"),
        "truth_isolation": "runner does not read a truth manifest or source coordinates",
    }


def _plan_comparison_payload(plan: dict) -> dict:
    comparable = json.loads(json.dumps(plan))
    comparable.pop("created_at_utc", None)
    comparable.setdefault("q_mode", str(runtime_config.Q_MODE))
    comparable.setdefault("source_position_mode", "single")
    comparable.setdefault(
        "physics_settings",
        {
            "wind_scale": float(runtime_config.WIND_SCALE),
            "decay_per_hour": float(runtime_config.RECURRENT_DECAY),
            "d_min_m2s": float(runtime_config.D_MIN_PHYS),
            "sigma_src_norm": float(runtime_config.SIGMA_SRC),
            "wind_vector_smoothing": (
                "enabled"
                if runtime_config.ENABLE_WIND_VECTOR_SMOOTHING
                else "disabled"
            ),
        },
    )
    return comparable


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", required=True, type=Path)
    parser.add_argument("--concentration", required=True, type=Path)
    parser.add_argument("--wind", required=True, type=Path)
    parser.add_argument("--design", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--solvers", nargs="+", default=["production"], choices=sorted(ALLOWED_SOLVERS)
    )
    parser.add_argument("--epochs", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--make-plots", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--q-mode",
        choices=["constant", "neural", "piecewise", "smooth_time"],
        default=runtime_config.Q_MODE,
    )
    parser.add_argument(
        "--source-position-mode", choices=["single", "fixed"], default="single"
    )
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument(
        "--event-window-crop",
        choices=["config", "enabled", "disabled"],
        default="config",
    )
    parser.add_argument("--wind-scale", type=float, default=runtime_config.WIND_SCALE)
    parser.add_argument(
        "--decay-per-hour", type=float, default=runtime_config.RECURRENT_DECAY
    )
    parser.add_argument("--d-min-m2s", type=float, default=runtime_config.D_MIN_PHYS)
    parser.add_argument("--sigma-src-norm", type=float, default=runtime_config.SIGMA_SRC)
    parser.add_argument(
        "--wind-vector-smoothing",
        choices=["enabled", "disabled"],
        default=(
            "enabled" if runtime_config.ENABLE_WIND_VECTOR_SMOOTHING else "disabled"
        ),
    )
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    for path in (args.sites, args.concentration, args.wind, args.design):
        if not path.is_file():
            parser.error(f"input file does not exist: {path}")
    if args.initial_checkpoint is not None and not args.initial_checkpoint.is_file():
        parser.error(f"initial checkpoint does not exist: {args.initial_checkpoint}")

    design = load_design(args.design)
    bounds = source_domain_bounds(args.sites)
    starts = resolve_starts(design, bounds)
    args.output_root.mkdir(parents=True, exist_ok=True)
    plan_path = args.output_root / "experiment_plan.json"
    requested_plan = prepare_plan(args, design, starts, bounds)
    if plan_path.exists():
        existing_plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if _plan_comparison_payload(existing_plan) != _plan_comparison_payload(
            requested_plan
        ):
            raise ValueError(
                "Existing experiment_plan.json does not match this invocation; "
                "use a new output root."
            )
        plan = existing_plan
    else:
        _write_json(plan_path, requested_plan)
        plan = requested_plan

    if args.plan_only:
        print(f"plan={plan_path.resolve()}")
        print(f"plan_sha256={_sha256(plan_path)}")
        print("plan_only=true")
        return

    completed = _completed_keys(args.output_root)
    failures = []
    for solver in args.solvers:
        for start in starts:
            _verify_code_snapshot(plan["code_snapshot"])
            key = (solver, start["id"])
            if key in completed:
                print(f"skip_completed solver={solver} start={start['id']}")
                continue
            print(f"run solver={solver} start={start['id']}")
            try:
                with _temporary_environment(
                    {
                        "PINN_RECURRENT_SOLVER": solver,
                        "PINN_EPOCHS": str(args.epochs),
                        "PINN_AUTO_CLOSE_PLOTS": "1",
                        "PINN_WIND_SCALE": str(args.wind_scale),
                        "PINN_RECURRENT_DECAY": str(args.decay_per_hour),
                        "PINN_D_MIN_PHYS": str(args.d_min_m2s),
                        "PINN_SIGMA_SRC": str(args.sigma_src_norm),
                        "PINN_ENABLE_WIND_VECTOR_SMOOTHING": (
                            "1" if args.wind_vector_smoothing == "enabled" else "0"
                        ),
                        "PINN_Q_MODE": args.q_mode,
                        "PINN_SOURCE_POSITION_MODE": args.source_position_mode,
                    }
                ):
                    result = pipeline.run(
                        site_path=args.sites,
                        conc_path=args.concentration,
                        wind_path=args.wind,
                        random_seed=args.seed + start["seed_offset"],
                        output_dir=args.output_root / "runs" / solver,
                        result_root_dir=args.output_root / "hourly_fields" / solver,
                        make_plots=args.make_plots,
                        run_id=start["id"],
                        result_name_suffix=f"{design.get('design_id', 'multistart')}_{solver}",
                        source_init_override_m=start["source_init_override_m"],
                        initial_checkpoint_path=args.initial_checkpoint,
                        event_window_crop_override=(
                            None
                            if args.event_window_crop == "config"
                            else args.event_window_crop == "enabled"
                        ),
                    )
                print(
                    f"completed solver={solver} start={start['id']} "
                    f"loss={result['best_raw_loss']:.8g} output={result['output_dir']}"
                )
            except Exception as exc:  # preserve partial runs and record the failure
                failure = {
                    "solver": solver,
                    "start_id": start["id"],
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "time_utc": _utc_now(),
                }
                failures.append(failure)
                _write_json(args.output_root / "failures.json", {"failures": failures})
                if not args.continue_on_error:
                    raise

    completed_after = sorted(_completed_keys(args.output_root))
    execution = {
        "schema_version": 1,
        "experiment_plan_sha256": _sha256(plan_path),
        "updated_at_utc": _utc_now(),
        "expected_run_count": len(args.solvers) * len(starts),
        "completed_run_count": len(
            [key for key in completed_after if key[0] in args.solvers]
        ),
        "completed_solver_start_keys": [list(key) for key in completed_after],
        "failures": failures,
        "plan": plan,
    }
    execution_path = args.output_root / "execution_status.json"
    _write_json(execution_path, execution)
    print(f"plan={plan_path.resolve()}")
    print(f"plan_sha256={_sha256(plan_path)}")
    print(f"status={execution_path.resolve()}")


if __name__ == "__main__":
    main()
