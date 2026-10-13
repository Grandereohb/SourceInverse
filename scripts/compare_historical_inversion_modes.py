"""Paired historical runs without modifying production config or source data."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "result" / "20260916京津冀指定污染物"
CASES = (
    "20260916_230434_苯",
    "20260916_224204_苯",
    "20260916_224101_丙烷",
)
INPUTS = ("sites.xlsx", "concentration.xlsx", "wind.xlsx")


RETIRED_SETTINGS = {"MODEL_NAME", "NONDETECT_MAD_MULTIPLIER", "NONDETECT_MIN_CALIBRATION_SAMPLES"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def worker(case, mode, output):
    sys.stdout.reconfigure(line_buffering=True)
    import matplotlib
    matplotlib.use("Agg")
    import numpy as np
    import pandas as pd
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    sys.path.insert(0, str(ROOT / "pinn_source"))
    import pipeline
    import ia_vpdi

    original_score = ia_vpdi.score_all_data_candidate
    evaluations = 0

    def log_candidate(*args, **kwargs):
        nonlocal evaluations
        result = original_score(*args, **kwargs)
        evaluations += 1
        if evaluations % 100 == 0:
            print(f"Profiled candidates evaluated: {evaluations}", flush=True)
        return result

    ia_vpdi.score_all_data_candidate = log_candidate
    original_layout_score = ia_vpdi.score_release_layouts
    layout_evaluations = 0

    def log_layout_candidate(*args, **kwargs):
        nonlocal layout_evaluations
        result = original_layout_score(*args, **kwargs)
        layout_evaluations += 1
        if layout_evaluations % 100 == 0:
            print(f"Layout-profiled geometries evaluated: {layout_evaluations}", flush=True)
        return result

    ia_vpdi.score_release_layouts = log_layout_candidate

    original_plot = pipeline.plot_station_timeseries

    def capture_predictions(**kwargs):
        pd.DataFrame({
            "time": kwargs["times"], "station": kwargs["station_names"],
            "observed": np.asarray(kwargs["obs_values"]).reshape(-1),
            "predicted": np.asarray(kwargs["pred_values"]).reshape(-1),
        }).to_csv(output / "predictions.csv", index=False, encoding="utf-8-sig")
        return original_plot(**kwargs)

    pipeline.plot_station_timeseries = capture_predictions
    snapshot = {name: value for name, value in vars(pipeline).items()
                if name.isupper() and isinstance(value, (str, int, float, bool, list, tuple, type(None)))}
    (output / "settings.json").write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")
    started = time.perf_counter()
    result = pipeline.run(
        *(SOURCE / case / name for name in INPUTS),
        random_seed=0, output_dir=output, result_root_dir=output,
        make_plots=True, inversion_mode=mode,
    )
    elapsed = time.perf_counter() - started
    result_dir = Path(result["output_dir"])
    for name in INPUTS:
        if digest(SOURCE / case / name) != digest(result_dir / name):
            raise RuntimeError(f"input copy mismatch: {name}")
    result["elapsed_seconds"] = elapsed
    result["source_decay_integrator"] = os.environ.get("PINN_SOURCE_DECAY_INTEGRATOR", pipeline.RECURRENT_SOURCE_DECAY_INTEGRATOR)
    result["diffusion_solver"] = os.environ.get("PINN_RECURRENT_DIFFUSION_SOLVER", pipeline.RECURRENT_DIFFUSION_SOLVER)
    result["q_layout_strategy"] = os.environ.get("PINN_IA_VP_Q_LAYOUT_STRATEGY", pipeline.IA_VP_Q_LAYOUT_STRATEGY)
    quality = json.loads((result_dir / "result_quality_report.json").read_text(encoding="utf-8"))
    result["source_mode"] = quality["recurrent_pde"]["source_mode"]
    result["source_sigma_m"] = quality["recurrent_pde"]["source_sigma_m"]
    result["transport_solver"] = quality["recurrent_pde"]["transport_solver"]
    result["parcel_subdivisions"] = quality["recurrent_pde"]["parcel_subdivisions"]
    result["case"] = case
    (output / "run.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


def audit_baseline(baseline, output):
    """Verify archived paired runs and replay fitted IA points, without a search."""
    import numpy as np
    import pandas as pd
    import torch
    os.environ["PINN_DEVICE"] = "cpu"
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    from inversion_validation_common import capture_context
    from variable_projection import build_q_basis, json_safe
    from ia_vpdi import score_all_data_candidate, compute_profile_information, temporary_source_and_diffusion
    import pipeline

    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((baseline/"manifest.json").read_text(encoding="utf-8"))
    current = json.loads(json.dumps({k:v for k,v in vars(pipeline).items() if k.isupper()
        and isinstance(v, (str, int, float, bool, list, tuple, type(None)))}))
    rows = []
    for case in CASES:
        frames, runs = {}, {}
        for mode in ("joint_adam", "ia_vpdi"):
            folder = baseline/case/mode
            settings = json.loads((folder/"settings.json").read_text(encoding="utf-8"))
            if settings.get("MODEL_NAME", "pinn") != "pinn":
                raise ValueError(f"Unsupported archived model backend: {case}/{mode}")
            if any(current.get(k) != v for k,v in settings.items() if k not in RETIRED_SETTINGS):
                raise ValueError(f"Archived algorithm settings differ: {case}/{mode}")
            run = json.loads((folder/"run.json").read_text(encoding="utf-8"))
            runs[mode] = run
            for name in INPUTS:
                if digest(SOURCE/case/name) != manifest["cases"][case][name]:
                    raise ValueError(f"Source inputs changed: {case}/{name}")
                if digest(Path(run["output_dir"])/name) != manifest["cases"][case][name]:
                    raise ValueError(f"Archived input copy mismatch: {case}/{mode}/{name}")
            frame = pd.read_csv(folder/"predictions.csv", parse_dates=["time"])
            if (frame.empty or frame[["time", "station"]].isna().any().any()
                    or frame.duplicated(["time", "station"]).any()
                    or not np.isfinite(frame[["observed", "predicted"]]).all().all()):
                raise ValueError(f"Invalid observation table: {case}/{mode}")
            frames[mode] = frame.sort_values(["time", "station"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(frames["joint_adam"][["time", "station", "observed"]],
                                      frames["ia_vpdi"][["time", "station", "observed"]])
        folder = output/case
        folder.mkdir()
        context = capture_context(case, folder)
        from inversion_validation_common import load_ia_vpdi_report
        report = load_ia_vpdi_report(runs["ia_vpdi"]["output_dir"])
        basis = build_q_basis(context["model"].recurrent_integration_times,
            report["q_basis"]["count"], report["q_basis"]["type"], knots=report["q_knots"])
        np.testing.assert_allclose(basis.cpu().numpy(), report["q_basis"]["values"], atol=1e-7)
        s, d = report["source"], report["diffusion"]
        candidate = score_all_data_candidate(context["model"], context["sigma_src"],
            context["observation_xyt"], context["observations"], basis, s["x_norm"], s["y_norm"],
            d["parallel_norm"], d["perp_norm"], weights=context["weights"],
            q_upper_bound=context["q_upper_bound"])
        if not candidate.solver_success or not np.isclose(candidate.profile_score,
                report["best_profiled_score"], rtol=2e-5, atol=1e-7):
            raise ValueError(f"Current fixed-source score does not reproduce archived result: {case}")
        with temporary_source_and_diffusion(context['model'], s['x_norm'], s['y_norm'],
                d['parallel_norm'], d['perp_norm']):
            information = compute_profile_information(context['model'], context['sigma_src'],
                context['observation_xyt'], context['observations'], basis, candidate.coefficients,
                weights=context['weights'], active_q_tol=pipeline.IA_VP_ACTIVE_Q_TOL,
                rcond=pipeline.IA_VP_INFORMATION_RCOND, condition_max=pipeline.IA_VP_INFORMATION_CONDITION_MAX,
                length_m=report['sampled_profile_stability']['length_m'],
                source_bounds=(context['source_x_bounds'], context['source_y_bounds']),
                boundary_margin_norm=pipeline.IA_VP_BOUNDARY_MARGIN_M/report['sampled_profile_stability']['length_m'],
                q_upper_bound=context['q_upper_bound'],
                localization_radius_limit_m=pipeline.IA_VP_LOCALIZATION_RADIUS_LIMIT_M)
        from dataclasses import asdict
        (folder/'replayed_information.json').write_text(
            json.dumps(json_safe(asdict(information)), indent=2, allow_nan=False), encoding='utf-8')
        joint, ia = frames["joint_adam"], frames["ia_vpdi"]
        peak = int(ia.observed.idxmax())
        dominant = ia.loc[peak, "station"]
        record = {"case": case, "observations": len(ia), "station_count": ia.station.nunique(),
            "observed_peak": float(ia.loc[peak, "observed"]),
            "ia_peak_prediction": float(ia.loc[peak, "predicted"]),
            "joint_peak_prediction": float(joint.loc[peak, "predicted"]),
            "archived_score": report["best_profiled_score"], "replayed_score": candidate.profile_score,
            "replay_forward_count": candidate.forward_count,
            "source_boundary_status": (report.get("profile_information") or {}).get("status"),
            "replayed_information_status": information.status,
            "replayed_structural_status": information.structural_status,
            "precision_assessment": information.precision_assessment,
            "ellipse_major_m": information.ellipse_major_m,
            "localization_reference_m": information.localization_reference_m,
            "ellipse_to_reference_ratio": information.ellipse_to_reference_ratio,
            "input_hashes": manifest["cases"][case]}
        for mode, frame in frames.items():
            error = frame.predicted-frame.observed
            mask = frame.station != dominant
            record.update({f"{mode}_rmse": float(np.sqrt(np.mean(error**2))),
                f"{mode}_non_dominant_rmse": float(np.sqrt(np.mean(error[mask]**2))),
                f"{mode}_worst_absolute_error": float(np.abs(error).max()),
                f"{mode}_end_to_end_seconds": runs[mode]["elapsed_seconds"]})
        record["rmse_reduction_percent"] = 100*(1-record["ia_vpdi_rmse"]/record["joint_adam_rmse"])
        record["runtime_ratio"] = record["ia_vpdi_end_to_end_seconds"]/record["joint_adam_end_to_end_seconds"]
        rows.append(record)
    payload = {"baseline": str(baseline), "cases": rows,
        "scope": "Archived complete paired runs with current fixed-source IA replays, not fresh optimization runs.",
        "limitations": ["Different fitting objectives and Q parameterizations; not an optimizer-only ablation.",
            "No known true source; better fit does not prove more accurate location.",
            "Q magnitudes are model units, not calibrated physical emission rates.",
            "Current replay checks the fitted score, not global search reproducibility or seed stability."]}
    (output/"baseline_audit.json").write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    pd.DataFrame(rows).drop(columns="input_hashes").to_csv(output/"baseline_audit.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(json_safe(payload), indent=2, ensure_ascii=False), flush=True)


def summarize(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    rows, station_rows = [], []
    for case in CASES:
        frames = {}
        for mode in ("joint_adam", "ia_vpdi"):
            folder = output / case / mode
            result = json.loads((folder / "run.json").read_text(encoding="utf-8"))
            report = json.loads((Path(result["output_dir"]) / "result_quality_report.json").read_text(encoding="utf-8"))
            frame = pd.read_csv(folder / "predictions.csv", parse_dates=["time"])
            frames[mode] = frame
            dominant = report["quality_diagnostics"]["dominant_station"]["station"]
            group = frame[frame.station == dominant]
            peak = group.loc[group.observed.idxmax()]
            residual = frame.predicted - frame.observed
            info = (report.get("ia_vpdi") or {}).get("profile_information") or {}
            q_series = pd.read_excel(Path(result["output_dir"]) / "diagnostics.xlsx", sheet_name="q_time_series")
            rows.append({
                "case": case, "mode": mode, "n_observations": len(frame),
                "rmse_raw": float(np.sqrt(np.mean(residual ** 2))),
                "mae_raw": float(np.mean(np.abs(residual))),
                "dominant_station": dominant,
                "dominant_rmse_raw": float(np.sqrt(np.mean((group.predicted - group.observed) ** 2))),
                "observed_peak": float(peak.observed), "predicted_at_observed_peak": float(peak.predicted),
                "peak_ratio_raw": float(peak.predicted / peak.observed),
                "source_lon": result["pred_lon"], "source_lat": result["pred_lat"],
                "elapsed_seconds": result["elapsed_seconds"], "best_epoch": result["best_epoch"],
                "information_status": info.get("status"), "ellipse_major_m": info.get("ellipse_major_m"),
                "q_max_model_units": float(q_series.Q.max()), "q_mean_model_units": float(q_series.Q.mean()),
                "d_parallel_m2s": report["recurrent_pde"].get("d_parallel_m2s"),
                "d_perp_m2s": report["recurrent_pde"].get("d_perp_m2s"),
                "result_dir": result["output_dir"],
                "train_on_residual": report["model"].get("train_on_residual"),
                "baseline_mode": report["model"].get("baseline_mode"),
                "c_scale": report["model"].get("c_scale"),
                "background_raw_rmse_floor": report["model"].get("background_error_floor", {}).get("raw_rmse_lower_bound"),
                "source_mode": report["recurrent_pde"].get("source_mode"),
                "source_sigma_m": report["recurrent_pde"].get("source_sigma_m"),
                "transport_solver": report["recurrent_pde"].get("transport_solver", "grid"),
                "parcel_subdivisions": report["recurrent_pde"].get("parcel_subdivisions"),
            })
            for station, g in frame.groupby("station", sort=False):
                p = g.loc[g.observed.idxmax()]
                station_rows.append({
                    "case": case, "mode": mode, "station": station, "n": len(g),
                    "rmse_raw": float(np.sqrt(np.mean((g.predicted - g.observed) ** 2))),
                    "observed_peak": float(p.observed), "predicted_at_observed_peak": float(p.predicted),
                    "predicted_peak": float(g.predicted.max()),
                })
        a = frames["joint_adam"].sort_values(["time", "station"]).reset_index(drop=True)
        b = frames["ia_vpdi"].sort_values(["time", "station"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(a[["time", "station", "observed"]], b[["time", "station", "observed"]])
        stations = a.station.unique()
        fig, axes = plt.subplots(len(stations), 1, figsize=(11, 2.2 * len(stations)), squeeze=False)
        for ax, station in zip(axes[:, 0], stations):
            for mode, color in (("joint_adam", "#c43c39"), ("ia_vpdi", "#23854d")):
                g = frames[mode][frames[mode].station == station].sort_values("time")
                ax.plot(g.time, g.predicted, color=color, label=mode)
            ax.plot(g.time, g.observed, color="#205ca0", marker=".", label="Observed")
            ax.set_title(station, fontsize=10)
            ax.grid(alpha=.2)
            ax.legend(loc="upper left", fontsize=8)
        fig.suptitle(case)
        fig.autofmt_xdate()
        fig.tight_layout(rect=(0, 0, 1, .98))
        fig.savefig(output / case / "comparison.png", dpi=130)
        plt.close(fig)
    pd.DataFrame(rows).to_csv(output / "comparison_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(station_rows).to_csv(output / "station_metrics.csv", index=False, encoding="utf-8-sig")
    (output / "comparison_summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(rows, indent=2, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", nargs=2, metavar=("CASE", "MODE"), help=argparse.SUPPRESS)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--summarize-only", action="store_true")
    modes.add_argument("--audit-baseline", type=Path,
                       help="Replay paired archives and recompute information without a search")
    args = parser.parse_args()
    if args.worker:
        if args.summarize_only or args.audit_baseline:
            parser.error("--worker cannot be combined with summary or audit")
        case, mode = args.worker
        if case not in CASES or mode not in ("joint_adam", "ia_vpdi"):
            parser.error("Unknown worker case or inversion mode")
        worker(case, mode, args.output)
        return
    if args.audit_baseline:
        audit_baseline(args.audit_baseline, args.output)
        return
    if args.summarize_only:
        summarize(args.output)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    overrides = {
        "source_decay_integrator": "PINN_SOURCE_DECAY_INTEGRATOR",
        "diffusion_solver": "PINN_RECURRENT_DIFFUSION_SOLVER",
        "q_layout_strategy": "PINN_IA_VP_Q_LAYOUT_STRATEGY",
        "train_on_residual_override": "PINN_TRAIN_ON_RESIDUAL",
        "baseline_mode_override": "PINN_BASELINE_MODE",
        "source_mode_override": "PINN_RECURRENT_SOURCE_MODE",
        "source_sigma_m_override": "PINN_SIGMA_SRC_M",
        "transport_solver_override": "PINN_TRANSPORT_SOLVER",
        "parcel_subdivisions_override": "PINN_PARCEL_SUBDIVISIONS",
        "start_strategy_override": "PINN_IA_VP_START_STRATEGY",
        "top_k_override": "PINN_IA_VP_TOP_K",
    }
    manifest = {name: os.environ.get(variable) for name, variable in overrides.items()}
    manifest.update(seed=0, device="cpu", threads=4, cases={
        case: {name: digest(SOURCE / case / name) for name in INPUTS} for case in CASES
    }, note="End-to-end comparison; different objectives/Q/diffusion, not an optimizer-only ablation.")
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    for case in CASES:
        for mode in ("joint_adam", "ia_vpdi"):
            folder = args.output / case / mode
            folder.mkdir(parents=True)
            env = dict(os.environ, PINN_DEVICE="cpu", PINN_INVERSION_MODE=mode,
                       PINN_AUTO_CLOSE_PLOTS="1", MPLBACKEND="Agg", PYTHONIOENCODING="utf-8")
            print(f"START {case} {mode}", flush=True)
            with (folder / "stdout.log").open("w", encoding="utf-8") as log:
                process = subprocess.run(
                    [sys.executable, str(Path(__file__).resolve()), "--output", str(folder),
                     "--worker", case, mode], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            if process.returncode:
                raise RuntimeError(f"run failed; see {folder / 'stdout.log'}")
            print(f"DONE {case} {mode}", flush=True)
    for case in CASES:
        for name in INPUTS:
            if digest(SOURCE / case / name) != manifest["cases"][case][name]:
                raise RuntimeError(f"source inputs changed during experiment: {case}/{name}")
    summarize(args.output)


if __name__ == "__main__":
    main()
