"""Reproducible known-source development experiments, not blind evaluation.

This driver does not alter production configuration. The hourly observation
adapter is experiment-local and all valid observations enter both objectives.
"""

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pinn_source"))
import field
import ia_vpdi as ia
from models.pinn import PINN
import variable_projection as vp

LENGTH_M = 1000.0
SIGMA = 0.15
BOUNDS = (-0.75, 0.75)
HOURS = 6
QUADRATURE = 4


def save_json(path, value):
    Path(path).write_text(json.dumps(vp.json_safe(value), indent=2,
                                    ensure_ascii=False, allow_nan=False), encoding="utf-8")


def hourly_mean_response(response, n_hours=HOURS, samples=QUADRATURE, n_stations=5):
    if response.ndim != 2 or response.shape[0] != n_hours * samples * n_stations:
        raise ValueError("Response rows must be ordered by hour, sample, then station")
    return response.reshape(n_hours, samples, n_stations, -1).mean(1).reshape(
        n_hours * n_stations, -1)


def averaged_response(model, sigma, xyt, basis, *args, **kwargs):
    return hourly_mean_response(vp.build_response_matrix(model, sigma, xyt, basis,
                                                       *args, **kwargs))


@contextmanager
def experiment_context():
    # Known releases start from an empty field, with no artificial t=0 pulse.
    with patch.object(field, "RECURRENT_INITIAL_RELEASE_FRACTION", 0.0), \
         patch.object(field, "RECURRENT_ADAPTIVE_SUBSTEPS", True), \
         patch.object(field, "RECURRENT_SUBSTEPS", 1), \
         patch.object(field, "RECURRENT_MAX_SUBSTEPS", 16), \
         patch.object(field, "RECURRENT_MAX_ADVECTION_CELLS", 1.0), \
         patch.object(ia, "build_response_matrix", averaged_response):
        yield


def case_definition(index):
    if not 0 <= index < 12:
        raise ValueError("The pilot has exactly 12 predeclared development cases")
    return {"case_id": f"pilot_{index + 1:02d}", "seed": 610170 + index,
            "wind": ("steady", "veering")[index // 6],
            "network": ("single_dominant", "distributed")[(index // 3) % 2],
            "release": ("constant", "pulse", "double_peak")[index % 3]}


def make_case(index, device="cpu", grid=36, noise_fraction=0.01):
    definition = case_definition(index)
    rng = np.random.default_rng(definition["seed"])
    torch.manual_seed(definition["seed"])
    model = PINN(hidden=8, depth=1).to(device)
    clock = np.linspace(0.0, HOURS, HOURS * 8 + 1)
    angle = (np.full_like(clock, 0.05) if definition["wind"] == "steady"
             else np.linspace(-0.35, 0.55, len(clock)))
    field.configure_recurrent_context(
        model, -2.0, 4.0, -3.0, 3.0, clock, 1.2 * np.cos(angle),
        1.2 * np.sin(angle), 0.0, 0.05, anisotropic_diffusion=True,
        d_parallel_min_norm=0.001, d_perp_min_norm=0.001,
        decay_norm=0.1, initial_release_dt=0.0, nx=grid, ny=grid,
        source_mode="gaussian", source_sigma_norm=SIGMA,
        diffusion_solver="gaussian", source_decay_integrator="exponential_linear",
        transport_solver="grid")
    source = np.array([-0.35, -0.15]) + rng.uniform(-0.08, 0.08, 2)
    with torch.no_grad():
        model.xs.fill_(source[0])
        model.ys.fill_(source[1])
    ia.set_model_anisotropic_diffusion_norm(model, 0.08, 0.03)
    stations = ([[0.3, -0.1], [0.5, 1.0], [1.0, -1.0], [1.7, 0.9], [2.0, -0.9]]
                if definition["network"] == "single_dominant" else
                [[0.3, -0.1], [0.6, 0.15], [0.95, -0.15], [1.3, 0.3], [1.8, -0.25]])
    stations = torch.tensor(stations, device=device, dtype=torch.float32)
    sample_times = torch.tensor(np.arange(HOURS * QUADRATURE) / QUADRATURE +
                                0.5 / QUADRATURE, device=device, dtype=torch.float32)
    xyt = torch.cat((stations.repeat(sample_times.numel(), 1),
                     sample_times.repeat_interleave(5)[:, None]), 1)
    basis = vp.build_q_basis(model.recurrent_integration_times, 5)
    coefficients = np.array({"constant": [1, 1, 1, 1, 1],
                             "pulse": [0, 0, 2, 0, 0],
                             "double_peak": [0, 2, 0, 1.5, 0]}[definition["release"]], float)
    response = averaged_response(model, SIGMA, xyt, basis).cpu().numpy()
    scale = float((response @ coefficients).max())
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError("The specified synthetic release has no finite detectable response")
    coefficients /= scale
    clean = response @ coefficients
    observed = clean + rng.normal(0, noise_fraction, clean.shape)
    metadata = {**definition, "source_norm": source.tolist(), "length_m": LENGTH_M,
                "d_parallel_m2_s": 0.08 * LENGTH_M**2 / 3600,
                "d_perp_m2_s": 0.03 * LENGTH_M**2 / 3600,
                "noise_std": noise_fraction, "coefficients_model_units": coefficients.tolist(),
                "station_peak_ratio_second_to_first": float(
                    np.sort(clean.reshape(HOURS, 5).max(0))[-2]),
                "q_unit": "model units, not kg/h", "split": "development",
                "generator": "same-model grid Gaussian; NOT independent physical validation"}
    return model, xyt, basis, coefficients, observed, clean, stations, metadata


def joint_refine(model, xyt, basis, observations, start, steps):
    """Matched projected joint Adam; same inner-fit warm start as profiled Adam."""
    ia.set_model_anisotropic_diffusion_norm(
        model, start.diffusion_parallel_norm, start.diffusion_perp_norm)
    with torch.no_grad():
        model.xs.fill_(start.source_x)
        model.ys.fill_(start.source_y)
    # A fixed change of optimization coordinates, not an amplitude constraint.
    coefficient_scale = max(float(np.max(start.coefficients)), 1e-6)
    a = torch.nn.Parameter(torch.tensor(start.coefficients / coefficient_scale,
                                       dtype=model.xs.dtype, device=model.xs.device))
    params = [model.xs, model.ys, model.rawD_perp, model.rawD_parallel_delta, a]
    optimizer = torch.optim.Adam(params, lr=0.02)
    obs = torch.as_tensor(observations, dtype=model.xs.dtype, device=model.xs.device)
    best = None
    trajectory = []
    started = time.perf_counter()
    for iteration in range(steps + 1):
        optimizer.zero_grad()
        response = averaged_response(model, SIGMA, xyt, basis, track_gradients=True)
        coefficients = a * coefficient_scale
        prediction = response @ coefficients
        loss = torch.mean((prediction - obs) ** 2)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite joint objective")
        score = float(loss.detach())
        trajectory.append(score)
        if iteration and iteration % 20 == 0:
            print(f"joint step={iteration}/{steps}, mse={score:.6g}", flush=True)
        if best is None or score < best["score"]:
            dp, dt = ia.model_anisotropic_diffusion_norm(model)
            best = {"score": score, "source": [float(model.xs.detach()), float(model.ys.detach())],
                    "diffusion": [float(dp.detach()), float(dt.detach())],
                    "coefficients": coefficients.detach().cpu().numpy().copy(),
                    "prediction": prediction.detach().cpu().numpy().copy()}
        if iteration == steps:
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 10.0)
        optimizer.step()
        with torch.no_grad():
            model.xs.clamp_(*BOUNDS)
            model.ys.clamp_(*BOUNDS)
            a.clamp_(min=0.0)
    return {**best, "seconds": time.perf_counter() - started,
            "forward_count": (steps + 1) * basis.shape[1], "trajectory": trajectory,
            "coefficient_coordinate_scale": coefficient_scale}


def metrics(source, coefficients, prediction, truth, true_q, basis, clean, observed):
    q = np.asarray(basis.cpu()) @ coefficients
    return {"localization_error_m": float(np.linalg.norm(np.asarray(source) - truth) * LENGTH_M),
            "q_relative_l2": float(np.linalg.norm(q - true_q) / np.linalg.norm(true_q)),
            "concentration_rmse_clean": float(np.sqrt(np.mean((prediction - clean)**2))),
            "objective_mse_observed": float(np.mean((prediction - observed)**2)),
            "at_search_boundary": bool(np.any(np.isclose(np.abs(source), BOUNDS[1], atol=0.01)))}


def execute_case(index, output, device, grid, steps):
    destination = output / case_definition(index)["case_id"]
    destination.mkdir()
    with experiment_context():
        model, xyt, basis, coefficients, observed, clean, stations, metadata = make_case(
            index, device, grid)
        save_json(destination / "input.json", metadata)
        np.savez_compressed(destination / "input.npz", xyt=xyt.cpu(), basis=basis.cpu(),
                            integration_times=model.recurrent_integration_times.cpu(),
                            wind_u=model.recurrent_u.cpu(), wind_v=model.recurrent_v.cpu(),
                            observed=observed, clean=clean, stations=stations.cpu(),
                            coefficients=coefficients)
        recovery = vp.solve_nonnegative_q(averaged_response(model, SIGMA, xyt, basis), clean)
        recovery_error = np.linalg.norm(recovery.coefficients-coefficients) / np.linalg.norm(coefficients)
        recovery_prediction_error = np.linalg.norm(recovery.prediction-clean) / np.linalg.norm(clean)
        if not recovery.success or recovery_error > 1e-3 or recovery_prediction_error > 1e-4:
            raise RuntimeError("Known-position noiseless Q recovery gate failed")
        # Both methods use exactly this common, truth-independent coarse search.
        ia.set_model_anisotropic_diffusion_norm(model, 0.05, 0.02)
        with torch.no_grad():
            model.xs.zero_()
            model.ys.zero_()
        coarse_started = time.perf_counter()
        coarse = ia.run_ia_vpdi(
            model, SIGMA, xyt, observed, BOUNDS, BOUNDS, 0.5,
            q_basis_count=5, top_k=2, diffusion_scale_seeds=(0.5, 1.0, 2.0),
            diffusion_ratio_seeds=(1.0, 3.0), outer_steps=0,
            start_strategy="spatial_diverse", start_min_distance=0.4)
        common_seconds = time.perf_counter() - coarse_started
        starts = [run.start for run in coarse.refinement_runs]
        # No generator truth is passed into start selection or refinement.
        profile_started = time.perf_counter()
        runs = []
        for start_index, start in enumerate(starts):
            print(f"profile start={start_index + 1}/{len(starts)}, steps={steps}", flush=True)
            runs.append(ia.refine_profiled_candidate(model, SIGMA, xyt, observed, basis, start,
                BOUNDS, BOUNDS, steps=steps, lr=0.02, patience=steps + 1, min_delta=0.0))
            print(f"profile done mse={runs[-1].best.profile_score:.6g}", flush=True)
        if not all(run.success for run in runs):
            raise RuntimeError(f"Profile refinement failed: {[run.message for run in runs]}")
        best = min((run.best for run in runs), key=lambda candidate: candidate.profile_score)
        profile_seconds = time.perf_counter() - profile_started
        joint_runs = []
        for start_index, start in enumerate(starts):
            print(f"joint start={start_index + 1}/{len(starts)}", flush=True)
            joint_runs.append(joint_refine(model, xyt, basis, observed, start, steps))
        joint = min(joint_runs, key=lambda run: run["score"])
        true_q = np.asarray(basis.cpu()) @ coefficients
        result = {"case": metadata, "gate_known_position_q_relative_error": float(
                    np.linalg.norm(recovery.coefficients-coefficients) / np.linalg.norm(coefficients)),
                  "common_search_seconds": common_seconds,
                  "common_search_forward_count": coarse.forward_count,
                  "common_starts": [asdict(start) for start in starts],
                  "ia_vpdi": {**metrics([best.source_x, best.source_y], best.coefficients,
                    best.prediction, metadata["source_norm"], true_q, basis, clean, observed),
                    "source_norm": [best.source_x, best.source_y],
                    "diffusion_norm": [best.diffusion_parallel_norm, best.diffusion_perp_norm],
                    "coefficients": best.coefficients,
                    "seconds_including_common_search": common_seconds + profile_seconds,
                    "forward_count": coarse.forward_count + sum(run.forward_count for run in runs)},
                  "matched_joint_adam": {**metrics(joint["source"], joint["coefficients"],
                    joint["prediction"], metadata["source_norm"], true_q, basis, clean, observed),
                    "source_norm": joint["source"],
                    "diffusion_norm": joint["diffusion"], "coefficients": joint["coefficients"],
                    "seconds_including_common_search": common_seconds + sum(r["seconds"] for r in joint_runs),
                    "forward_count": coarse.forward_count + sum(r["forward_count"] for r in joint_runs)},
                  "limitations": ["Development pilot, not held-out evidence",
                    "Common profiled initialization is charged to both methods",
                    "Same-model synthetic data; forward mismatch remains to be evaluated",
                    "Shared startup and diagnostic costs are not deployment benchmarks"]}
        for label, source, diffusion, fitted, prediction in (
                ("profile", [best.source_x, best.source_y],
                 [best.diffusion_parallel_norm, best.diffusion_perp_norm], best.coefficients, best.prediction),
                ("joint", joint["source"], joint["diffusion"], joint["coefficients"], joint["prediction"])):
            ia.set_model_anisotropic_diffusion_norm(model, *diffusion)
            with torch.no_grad():
                model.xs.fill_(source[0])
                model.ys.fill_(source[1])
            replay = averaged_response(model, SIGMA, xyt, basis).cpu().numpy() @ fitted
            if not np.allclose(replay, prediction, rtol=2e-5, atol=1e-6):
                raise RuntimeError(f"{label} saved-state prediction replay failed")
        np.savez_compressed(destination / "predictions.npz", true_q=true_q,
            ia_q=np.asarray(basis.cpu()) @ best.coefficients,
            joint_q=np.asarray(basis.cpu()) @ joint["coefficients"],
            ia_prediction=best.prediction, joint_prediction=joint["prediction"],
            ia_coefficients=best.coefficients, joint_coefficients=joint["coefficients"])
        save_json(destination / "trajectories.json", {"profile": [r.trajectory for r in runs],
                                                     "joint": joint_runs})
        save_json(destination / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "result/paper_experiments_20261017/pilot")
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--grid", type=int, default=36)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if not 1 <= args.count <= 12 or args.steps < 0 or args.grid < 8:
        parser.error("count must be 1..12, steps nonnegative, grid >= 8")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    output = args.output.resolve()
    result_root = (ROOT / "result").resolve()
    if not output.is_relative_to(result_root):
        parser.error("All experiment artifacts must stay under result/")
    output.mkdir(parents=True, exist_ok=True)
    settings = {"steps": args.steps, "grid": args.grid, "device": args.device,
                "clock_hours": "0..6, dt=0.125 h", "hourly_mean": "4 midpoint samples/hour",
                "torch": torch.__version__, "numpy": np.__version__, "platform": platform.platform(),
                "source_bounds_norm": BOUNDS, "sigma_norm": SIGMA,
                "length_m": LENGTH_M, "seed_rule": "610170+case_index",
                "fit": "all observations, unweighted SSE, nonnegative 5-hat Q, no cap",
                "diffusion": "wind-aligned anisotropic Gaussian, floors=0.001 normalized",
                "initial_field": "zero; initial_release_fraction=0",
                "decay_per_hour": 0.1, "source_scale": float(field.RECURRENT_SOURCE_SCALE),
                "initial_diffusion_norm": [0.05, 0.02],
                "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "core_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted((ROOT / "pinn_source").rglob("*.py"))},
                "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True)}
    manifest = output / "manifest.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text(encoding="utf-8"))
        if previous != json.loads(json.dumps(settings)):
            raise ValueError("Manifest differs: use a new output directory; do not mix versions")
    else:
        save_json(manifest, settings)
    results = []
    for index in range(args.count):
        path = output / case_definition(index)["case_id"] / "result.json"
        if path.exists():
            results.append(json.loads(path.read_text(encoding="utf-8")))
            continue
        print(f"START {index + 1}/{args.count}: {case_definition(index)}", flush=True)
        try:
            results.append(execute_case(index, output, args.device, args.grid, args.steps))
        except Exception as exc:
            save_json(output / "STOPPED.json", {"case": case_definition(index),
                "type": type(exc).__name__, "message": str(exc),
                "action": "Stopped; inspect and discuss before extending experiments"})
            raise
        save_json(output / "summary.json", results)
        print(f"DONE {index + 1}: IA={results[-1]['ia_vpdi']['localization_error_m']:.1f}m, "
              f"joint={results[-1]['matched_joint_adam']['localization_error_m']:.1f}m", flush=True)
    save_json(output / "summary.json", results)


if __name__ == "__main__":
    main()
