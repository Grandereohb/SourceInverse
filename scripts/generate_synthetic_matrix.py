"""Generate every scenario in a predeclared synthetic development matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "scripts" / "generate_synthetic_puff_scenario.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def load_matrix(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("Only synthetic matrix schema_version=1 is supported.")
    scenarios = payload.get("scenarios")
    if scenarios is None and "factorial" in payload:
        scenarios = _expand_factorial(payload["factorial"])
        payload["scenarios"] = scenarios
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("The matrix must contain scenarios.")
    ids = [str(row.get("scenario_id", "")).strip() for row in scenarios]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("Scenario IDs must be unique and non-empty.")
    return payload


def _expand_factorial(factorial: dict) -> list[dict]:
    required = ("input_ids", "positions", "physics_conditions")
    if any(not factorial.get(key) for key in required):
        raise ValueError(f"factorial design requires non-empty {required}")
    defaults = dict(factorial.get("defaults", {}))
    prefix = str(factorial.get("scenario_prefix", "factorial_puff"))
    base_seed = int(factorial.get("base_seed", 0))
    release_conditions = factorial.get("release_conditions")
    if release_conditions is None:
        release_conditions = [None]
    elif not isinstance(release_conditions, list) or not release_conditions:
        raise ValueError("release_conditions must be a non-empty list when supplied")
    scenarios = []
    index = 0
    for input_id in factorial["input_ids"]:
        for position in factorial["positions"]:
            for physics in factorial["physics_conditions"]:
                for release in release_conditions:
                    index += 1
                    design_factors = {
                        "input_id": str(input_id),
                        "position_id": str(position["id"]),
                        "physics_condition_id": str(physics["id"]),
                    }
                    row = dict(defaults)
                    row.update(
                        {
                            "scenario_id": f"{prefix}_{index:04d}",
                            "seed": base_seed + index,
                            "input_id": str(input_id),
                            "fraction_x": float(position["fraction_x"]),
                            "fraction_y": float(position["fraction_y"]),
                            "diffusivity": float(physics["diffusivity"]),
                            "sigma": float(physics["sigma"]),
                            "decay": float(physics["decay"]),
                            "wind_factor": float(physics["wind_factor"]),
                            "design_factors": design_factors,
                        }
                    )
                    for key, value in physics.items():
                        if key not in {
                            "id",
                            "diffusivity",
                            "sigma",
                            "decay",
                            "wind_factor",
                        }:
                            row[key] = value
                    if release is not None:
                        if not str(release.get("id", "")).strip():
                            raise ValueError(
                                "every release condition requires a non-empty id"
                            )
                        design_factors["release_condition_id"] = str(release["id"])
                        for key, value in release.items():
                            if key != "id":
                                row[key] = value
                    scenarios.append(row)
    return scenarios


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def _scenario_inputs(matrix: dict, row: dict, sites: Path | None, wind: Path | None):
    input_id = row.get("input_id")
    if input_id is None:
        if sites is None or wind is None:
            raise ValueError("global --sites/--wind or a scenario input_id is required")
        return sites.resolve(), wind.resolve(), None
    try:
        input_spec = matrix["input_sets"][input_id]
    except KeyError as exc:
        raise ValueError(f"unknown input_id: {input_id}") from exc
    return (
        _resolve_repo_path(input_spec["sites"]),
        _resolve_repo_path(input_spec["wind"]),
        str(input_id),
    )


def _command(row: dict, sites: Path, wind: Path, output_dir: Path, generator: Path = GENERATOR) -> list[str]:
    command = [
        sys.executable,
        str(generator),
        "--sites",
        str(sites),
        "--wind",
        str(wind),
        "--output-dir",
        str(output_dir),
        "--scenario-id",
        str(row["scenario_id"]),
        "--seed",
        str(row["seed"]),
        "--source-fraction-x",
        str(row["fraction_x"]),
        "--source-fraction-y",
        str(row["fraction_y"]),
        "--q-shape",
        str(row["q_shape"]),
        "--target-signal-peak",
        str(row["target_peak"]),
        "--diffusivity-m2s",
        str(row["diffusivity"]),
        "--initial-sigma-m",
        str(row["sigma"]),
        "--decay-per-hour",
        str(row["decay"]),
        "--truth-wind-factor",
        str(row["wind_factor"]),
        "--noise-absolute",
        str(row["noise_absolute"]),
        "--noise-relative",
        str(row["noise_relative"]),
    ]
    optional = {
        "background_level": "--background-level",
        "station_bias_fraction": "--station-bias-fraction",
        "pre_event_hours": "--pre-event-hours",
        "integration_dt_h": "--integration-dt-h",
        "diffusivity_along_m2s": "--diffusivity-along-m2s",
        "diffusivity_cross_m2s": "--diffusivity-cross-m2s",
        "initial_sigma_along_m": "--initial-sigma-along-m",
        "initial_sigma_cross_m": "--initial-sigma-cross-m",
        "sensor_kernel_sigma_m": "--sensor-kernel-sigma-m",
        "meander_amplitude_m": "--meander-amplitude-m",
        "meander_period_h": "--meander-period-h",
        "particles_per_release": "--particles-per-release",
        "release_dt_h": "--release-dt-h",
    }
    for key, flag in optional.items():
        if key in row:
            command.extend([flag, str(row[key])])
    return command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--sites", type=Path)
    parser.add_argument("--wind", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    if not args.matrix.is_file():
        parser.error(f"input file does not exist: {args.matrix}")
    if (args.sites is None) != (args.wind is None):
        parser.error("--sites and --wind must be provided together")
    for path in (args.sites, args.wind):
        if path is not None and not path.is_file():
            parser.error(f"input file does not exist: {path}")
    matrix = load_matrix(args.matrix)
    generator = _resolve_repo_path(matrix.get("generator_script", GENERATOR))
    if not generator.is_file():
        raise ValueError(f"generator script does not exist: {generator}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    generated = []
    for row in matrix["scenarios"]:
        scenario_sites, scenario_wind, input_id = _scenario_inputs(
            matrix, row, args.sites, args.wind
        )
        for path in (scenario_sites, scenario_wind):
            if not path.is_file():
                raise ValueError(f"scenario input does not exist: {path}")
        scenario_dir = args.output_root / row["scenario_id"]
        manifest_path = scenario_dir / "scenario_manifest.json"
        if manifest_path.is_file():
            print(f"skip_existing scenario={row['scenario_id']}")
        elif scenario_dir.exists():
            raise ValueError(
                f"Scenario directory exists without a manifest: {scenario_dir}"
            )
        else:
            subprocess.run(
                _command(row, scenario_sites, scenario_wind, scenario_dir, generator),
                cwd=ROOT,
                check=True,
            )
        generated.append(
            {
                "scenario_id": row["scenario_id"],
                "manifest_path": str(manifest_path.resolve()),
                "manifest_sha256": _sha256(manifest_path),
                "input_id": input_id,
                "design_factors": row.get("design_factors", {}),
            }
        )
    batch = {
        "schema_version": 1,
        "analysis": "synthetic_scenario_matrix_generation",
        "design": {
            "path": str(args.matrix.resolve()),
            "sha256": _sha256(args.matrix),
            "design_id": matrix["design_id"],
            "split": matrix["split"],
            "retention_rule": matrix["retention_rule"],
            "generator_script": str(generator),
            "generator_script_sha256": _sha256(generator),
        },
        "inputs": (
            {
                "sites": {
                    "path": str(args.sites.resolve()),
                    "sha256": _sha256(args.sites),
                },
                "wind": {
                    "path": str(args.wind.resolve()),
                    "sha256": _sha256(args.wind),
                },
            }
            if args.sites is not None
            else {
                input_id: {
                    label: {
                        "path": str(_resolve_repo_path(value)),
                        "sha256": _sha256(_resolve_repo_path(value)),
                    }
                    for label, value in spec.items()
                }
                for input_id, spec in matrix["input_sets"].items()
            }
        ),
        "scenario_count": len(generated),
        "scenarios": generated,
    }
    batch_path = args.output_root / "batch_manifest.json"
    batch_path.write_text(
        json.dumps(batch, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"batch_manifest={batch_path.resolve()}")
    print(f"batch_manifest_sha256={_sha256(batch_path)}")


if __name__ == "__main__":
    main()
