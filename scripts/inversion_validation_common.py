"""Shared offline validation setup; no fitted parameters or experiment drivers."""

import copy
import inspect
import json
from contextlib import ExitStack
from contextlib import nullcontext, redirect_stdout
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pinn_source"))
import field
import pipeline

SOURCE = ROOT / "result" / "20260916\u4eac\u6d25\u5180\u6307\u5b9a\u6c61\u67d3\u7269"


class ContextReady(Exception):
    pass


def load_ia_vpdi_report(case_dir):
    """Read consolidated diagnostics, falling back only for historical cases."""
    case_dir = Path(case_dir)
    consolidated = case_dir / "result_quality_report.json"
    if consolidated.exists():
        report = json.loads(consolidated.read_text(encoding="utf-8"))
        if report.get("ia_vpdi") is not None:
            return report["ia_vpdi"]
    return json.loads((case_dir / "ia_vpdi_report.json").read_text(encoding="utf-8"))


def response_amplitude_diagnostics(response, coefficients):
    """Absolute response scale; not a penalty or a physical emission estimate."""
    response = np.asarray(response, dtype=float)
    coefficients = np.asarray(coefficients, dtype=float)
    if (response.ndim != 2 or response.shape[0] == 0 or coefficients.ndim != 1
            or response.shape[1] != len(coefficients) or len(coefficients) == 0
            or not np.isfinite(response).all() or not np.isfinite(coefficients).all()
            or np.any(coefficients < 0)):
        raise ValueError("Finite response and matching nonnegative coefficients required")
    norms = np.linalg.norm(response, axis=0)
    active = coefficients > 0
    q_norm = float(np.linalg.norm(coefficients))
    prediction_norm = float(np.linalg.norm(response @ coefficients))
    return {"response_column_norm_min": float(norms.min()),
            "response_column_norm_max": float(norms.max()),
            "active_response_column_norm_min": float(norms[active].min()) if active.any() else None,
            "coefficient_l2_model_units": q_norm,
            "prediction_l2_normalized": prediction_norm,
            "effective_response_gain": prediction_norm/q_norm if q_norm > 0 else None}


def summarize_near_equivalent_candidates(records, length_m, tolerances=(.01, .05, .1)):
    """Descriptive sampled score sets, never calibrated confidence intervals."""
    if not records or not np.isfinite(length_m) or length_m <= 0:
        raise ValueError("Candidates and positive finite physical length required")
    values = np.array([[r[k] for k in ("x_norm", "y_norm", "normalized_mse", "q_max_model_units")]
                       for r in records], dtype=float)
    if not np.isfinite(values).all() or np.any(values[:, 2:] < 0):
        raise ValueError("Finite candidate coordinates and nonnegative scores/Q required")
    if not np.isfinite(tolerances).all() or any(t < 0 for t in tolerances):
        raise ValueError("Finite nonnegative relative score tolerances required")
    best = float(values[:, 2].min())
    summaries = []
    for tolerance in tolerances:
        mask = values[:, 2] <= best*(1+tolerance)
        selected = values[mask]
        delta = selected[:, None, :2]-selected[None, :, :2]
        summaries.append({"relative_mse_tolerance": tolerance, "score_limit": best*(1+tolerance),
            "candidate_count": int(mask.sum()), "x_span_m": float(np.ptp(selected[:, 0])*length_m),
            "y_span_m": float(np.ptp(selected[:, 1])*length_m),
            "sampled_diameter_m": float(np.linalg.norm(delta, axis=2).max()*length_m),
            "q_peak_min_model_units": float(selected[:, 3].min()),
            "q_peak_max_model_units": float(selected[:, 3].max()),
            "q_peak_ratio": float(selected[:, 3].max()/selected[:, 3].min()) if selected[:, 3].min() > 0 else None,
            "touches_allowed_boundary": any(r["at_allowed_boundary"] for r,m in zip(records, mask) if m),
            "touches_candidate_window_edge": any(r["at_candidate_window_edge"] for r,m in zip(records, mask) if m),
            "touches_unconstrained_window_edge": any(r.get("at_unconstrained_window_edge", False)
                for r,m in zip(records, mask) if m)})
    return summaries


def extend_boundary_candidates(candidates, center, x_bounds, y_bounds, length_m):
    """Bounded north-edge/east-window diagnostic, not a global search."""
    points = set(map(tuple, candidates))
    if not points or length_m <= 0:
        raise ValueError("Nonempty candidates and positive physical length required")
    xs = sorted({x for x, _ in points} | {center[0]+d/length_m for d in (2500., 3000.)})
    ys = sorted({y for _, y in points} | {y_bounds[1]})
    points.update((x, y) for x in xs for y in ys
                  if x_bounds[0] <= x <= x_bounds[1] and y_bounds[0] <= y <= y_bounds[1])
    return sorted(points)


def trace_recurrent_mass(model, sigma_src, q, disabled_operator=None):
    """Trace existing substep operators; disabled variants are diagnostic only."""
    if disabled_operator not in (None, "advection", "diffusion", "transport"):
        raise ValueError("Unknown operator control")
    cell_area = float((model.recurrent_x_grid[1]-model.recurrent_x_grid[0]) *
                      (model.recurrent_y_grid[1]-model.recurrent_y_grid[0]))
    mass = lambda value: float(value.detach().double().sum())*cell_area
    rows = []
    original_step = field._advance_recurrent_step
    signature = inspect.signature(original_step)

    def step(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        p = bound.arguments
        dt = float(p["dt"])
        row = {"substep": len(rows), "mass_before": mass(p["field"]),
               "source_mass": mass(p["source_start"]), "dt_norm": dt}
        exponential = p["source_decay_integrator"] == "exponential_linear"
        attenuation, left, right = (p["source_decay_plan"] or field.source_decay_weights(dt, max(float(p["decay"]), 0.)))
        row["decay_order"] = "before_injection" if exponential else "after_diffusion"
        row["mass_after_first_injection"] = (row["mass_before"]*attenuation if exponential else row["mass_before"]) + float(p["source_scale"]*p["q_start"])*row["source_mass"]*(left if exponential else dt*.5)
        rows.append(row)
        result = original_step(*args, **kwargs)
        row["mass_after_decay"] = (row["mass_before"]*attenuation if exponential else row["mass_after_diffusion"]*attenuation)
        row["mass_after_second_injection"] = mass(result)
        expected = (row["mass_after_diffusion"] if exponential else row["mass_after_decay"]) + float(p["source_scale"]*p["q_end"])*mass(p["source_end"])*(right if exponential else dt*.5)
        row["injection_balance_error"] = mass(result)-expected
        return result

    def wrapped(original, stage, disabled):
        def apply(value, *args, **kwargs):
            result = value if disabled else original(value, *args, **kwargs)
            rows[-1][stage] = mass(result)
            return result
        return apply

    with ExitStack() as stack, torch.no_grad():
        stack.enter_context(patch.object(field, "_advance_recurrent_step", side_effect=step))
        for name, stage, control in (("_advect_field", "mass_after_advection", "advection"),
                ("_diffuse_field", "mass_after_diffusion", "diffusion"),
                ("_diffuse_field_anisotropic", "mass_after_diffusion", "diffusion"),
                ("_diffuse_field_implicit", "mass_after_diffusion", "diffusion"),
                ("_diffuse_field_tensor", "mass_after_diffusion", "diffusion")):
            original = getattr(field, name)
            stack.enter_context(patch.object(field, name, side_effect=wrapped(
                original, stage, disabled_operator in (control, "transport"))))
        fields = field.recurrent_plume_fields_from_q(model, sigma_src, q)
    return fields, rows


def local_source_candidates(center, x_bounds, y_bounds, length_m,
                            offsets_m=(-1000., -500., 0., 500., 1000.)):
    """Bounded offset grid; snap float32 center roundoff, discard outside offsets."""
    values = np.asarray([*center, *x_bounds, *y_bounds, length_m, *offsets_m], dtype=float)
    if not np.isfinite(values).all() or length_m <= 0 or len(offsets_m) == 0:
        raise ValueError("Finite bounds/offsets and positive length required")
    tolerance = np.finfo(np.float32).eps * max(1., np.max(np.abs(values[:6])))
    if (x_bounds[0] > x_bounds[1] or y_bounds[0] > y_bounds[1]
            or not x_bounds[0]-tolerance <= center[0] <= x_bounds[1]+tolerance
            or not y_bounds[0]-tolerance <= center[1] <= y_bounds[1]+tolerance):
        raise ValueError("Center must be inside ordered source bounds")
    # Trained float32 boundary coordinates may round outside float64 bounds.
    center = (np.clip(center[0], *x_bounds), np.clip(center[1], *y_bounds))
    return sorted({(float(center[0]+dx/length_m), float(center[1]+dy/length_m))
                   for dx in offsets_m for dy in offsets_m
                   if x_bounds[0] <= center[0]+dx/length_m <= x_bounds[1]
                   and y_bounds[0] <= center[1]+dy/length_m <= y_bounds[1]})


def capture_context(case, output):
    captured = {}

    def capture(**kwargs):
        captured.update(kwargs)
        raise ContextReady()

    with tempfile.TemporaryDirectory(prefix="peak_q_context_") as temporary:
        with (output / "preprocessing.log").open("w", encoding="utf-8") as log:
            with patch.object(pipeline, "run_ia_vpdi", side_effect=capture), redirect_stdout(log):
                try:
                    pipeline.run(*(SOURCE / case / name for name in
                                   ("sites.xlsx", "concentration.xlsx", "wind.xlsx")),
                                 output_dir=temporary, result_root_dir=temporary,
                                 make_plots=False, inversion_mode="ia_vpdi", random_seed=0)
                except ContextReady:
                    pass
    if not captured:
        raise RuntimeError("failed to capture the initialized transport context")
    return captured


def configured_variant(prototype, n, substeps=None, dense=False, release_breakpoints=None,
                       preserve_diffusion_mode=False):
    model = copy.deepcopy(prototype)
    original_t = prototype.recurrent_times.detach().cpu().numpy()
    times = np.linspace(original_t[0], original_t[-1], (len(original_t) - 1) * 4 + 1) if dense else original_t
    indices = np.clip(np.searchsorted(original_t, times + 1e-7, side="right") - 1, 0, len(original_t) - 1)
    changes = {} if substeps is None else {"RECURRENT_ADAPTIVE_SUBSTEPS": False,
        "RECURRENT_SUBSTEPS": substeps, "RECURRENT_MAX_SUBSTEPS": substeps}
    with (patch.multiple(field, **changes) if changes else nullcontext()):
        field.configure_recurrent_context(model,
            float(prototype.recurrent_x_grid[0]), float(prototype.recurrent_x_grid[-1]),
            float(prototype.recurrent_y_grid[0]), float(prototype.recurrent_y_grid[-1]),
            times, prototype.recurrent_u[indices], prototype.recurrent_v[indices],
            d_min_norm=prototype.recurrent_d_min_norm, d_scale_norm=prototype.recurrent_d_scale_norm,
            anisotropic_diffusion=(prototype.recurrent_anisotropic_diffusion if preserve_diffusion_mode else False),
            d_parallel_min_norm=(prototype.recurrent_d_parallel_min_norm if preserve_diffusion_mode else None),
            d_perp_min_norm=(prototype.recurrent_d_perp_min_norm if preserve_diffusion_mode else None),
            decay_norm=prototype.recurrent_decay_norm,
            source_mode=getattr(prototype, "recurrent_source_mode", "gaussian"),
            source_sigma_norm=getattr(prototype, "recurrent_source_sigma_norm", None),
            diffusion_solver=getattr(prototype, "recurrent_diffusion_solver", "gaussian"),
            source_decay_integrator=getattr(prototype, "recurrent_source_decay_integrator", "legacy"),
            transport_solver=getattr(prototype, "recurrent_transport_solver", "grid"),
            parcel_subdivisions=getattr(prototype, "recurrent_parcel_subdivisions", 64),
            parcel_q_breakpoints=getattr(prototype, "recurrent_parcel_q_breakpoints", ()),
            release_breakpoints=release_breakpoints,
            initial_release_dt=prototype.recurrent_initial_release_dt, nx=n, ny=n)
    return model


def fixed_q_transport_control(model, sigma_src, xyt, knots, coefficients, disabled_operator=None):
    """One fixed piecewise-linear release; no amplitude fitting or source search."""
    from variable_projection import build_q_basis
    knots = np.asarray(knots, dtype=float)
    coefficients = np.asarray(coefficients, dtype=float)
    if (knots.ndim != 1 or len(knots) < 2 or coefficients.shape != knots.shape
            or not np.isfinite(knots).all() or not np.isfinite(coefficients).all()
            or np.any(np.diff(knots) <= 0) or np.any(coefficients < 0)):
        raise ValueError("Ordered finite knots and matching nonnegative coefficients required")
    clock = model.recurrent_integration_times
    if float(clock[0]) < knots[0]-1e-6 or float(clock[-1]) > knots[-1]+1e-6:
        raise ValueError("Fixed release must cover the integration clock")
    basis = build_q_basis(clock, len(knots), knots=knots)
    q = basis @ torch.as_tensor(coefficients, device=clock.device, dtype=clock.dtype)
    fields, mass_rows = trace_recurrent_mass(model, sigma_src, q, disabled_operator)
    with torch.no_grad():
        query = torch.as_tensor(xyt, device=fields.device, dtype=fields.dtype)
        predictions = field._sample_recurrent_fields(model, fields, query).cpu().numpy().reshape(-1)
        dx = model.recurrent_x_grid[1]-model.recurrent_x_grid[0]
        dy = model.recurrent_y_grid[1]-model.recurrent_y_grid[0]
        mass = fields.double().sum(dim=(1, 2))*float(dx*dy)
        weighted = fields.double()/mass[:, None, None].clamp_min(1e-300)*float(dx*dy)
        x = model.recurrent_x_mesh.double()
        y = model.recurrent_y_mesh.double()
        cx = (weighted*x).sum(dim=(1, 2))
        cy = (weighted*y).sum(dim=(1, 2))
        moments = [{"time_norm": float(t), "mass_norm": float(m),
            "peak_normalized": float(f.max()),
            "center_x_norm": float(a) if m > 0 else None,
            "center_y_norm": float(b) if m > 0 else None,
            "variance_x_norm2": float((p*(x-a).square()).sum()) if m > 0 else None,
            "variance_y_norm2": float((p*(y-b).square()).sum()) if m > 0 else None}
            for t,m,f,p,a,b in zip(model.recurrent_times, mass, fields, weighted, cx, cy)]
    if not np.isfinite(predictions).all() or not torch.isfinite(fields).all():
        raise RuntimeError("Nonfinite fixed-Q transport output")
    return predictions, moments, mass_rows


def exact_release_decay_mass(model, knots, coefficients):
    """Scalar unit-mass source reference with advection/diffusion disabled only."""
    from scipy.integrate import quad
    knots = np.asarray(knots, dtype=float)
    coefficients = np.asarray(coefficients, dtype=float)
    times = model.recurrent_times.detach().cpu().numpy().astype(float)
    if (knots.ndim != 1 or knots.shape != coefficients.shape or len(knots) < 2
            or not np.isfinite(knots).all() or not np.isfinite(coefficients).all()
            or np.any(np.diff(knots) <= 0) or np.any(coefficients < 0)
            or knots[0] > times[0]+1e-6 or knots[-1] < times[-1]-1e-6):
        raise ValueError("Finite nonnegative release covering the model clock required")
    decay = max(float(model.recurrent_decay_norm), 0.)
    initial = (field.RECURRENT_INITIAL_RELEASE_FRACTION*model.recurrent_initial_release_dt
               *float(np.interp(times[0], knots, coefficients)))
    result = []
    for end in times:
        mass = initial*np.exp(-decay*(end-times[0]))
        for start, stop in zip(knots[:-1], knots[1:]):
            left, right = max(start, times[0]), min(stop, end)
            if right > left:
                mass += quad(lambda t: float(np.interp(t, knots, coefficients))
                             *np.exp(-decay*(end-t)), left, right, epsabs=1e-12)[0]
        result.append(field.RECURRENT_SOURCE_SCALE*mass)
    return np.asarray(result)

def set_anisotropic_diffusion(model, parallel, perpendicular):
    """Diagnostic physical coefficients through the existing parameterization."""
    import torch
    from ia_vpdi import softplus_inverse
    scale = float(model.recurrent_d_scale_norm)
    floor_parallel = float(model.recurrent_d_parallel_min_norm)/scale
    floor_perp = float(model.recurrent_d_perp_min_norm)/scale
    base = perpendicular-floor_perp
    delta = parallel-floor_parallel-base
    if base <= 1e-6 or delta <= 0:
        raise ValueError("Requested coefficients violate the existing parameterization")
    with torch.no_grad():
        model.rawD_perp.copy_(softplus_inverse(model.rawD_perp.new_tensor(base-1e-6)))
        model.rawD_parallel_delta.copy_(softplus_inverse(model.rawD_parallel_delta.new_tensor(delta)))


def diffusion_impulse_moments(dx_m, dy_m, dt_seconds, parallel, perpendicular,
                              u, v, solver="anisotropic", size=65):
    """Measure the actual discrete diffusion operator; boundaries remain explicit."""
    if min(dx_m, dy_m, dt_seconds, parallel, perpendicular) <= 0 or size % 2 != 1:
        raise ValueError("Positive physical inputs and an odd impulse grid are required")
    import torch
    grid = torch.arange(size, dtype=torch.float64) - size//2
    x, y = grid*dx_m, grid*dy_m
    impulse = torch.zeros((size, size), dtype=torch.float64)
    impulse[size//2, size//2] = 1.
    tensor = lambda value: torch.tensor(value, dtype=torch.float64)
    if solver == "anisotropic":
        result = field._diffuse_field_anisotropic(impulse, x, y, tensor(parallel),
            tensor(perpendicular), tensor(u), tensor(v), tensor(dt_seconds))
    elif solver in ("gaussian", "implicit"):
        if parallel != perpendicular:
            raise ValueError("Scalar controls require equal diffusion coefficients")
        operation = field._diffuse_field if solver == "gaussian" else field._diffuse_field_implicit
        result = operation(impulse, x, y, tensor(parallel), tensor(dt_seconds))
    else:
        raise ValueError("Unknown diffusion operator")
    mass = float(result.sum())
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    speed = np.hypot(u, v)
    c, s = (u/speed, v/speed) if speed > 1e-12 else (1., 0.)
    along, across = xx*c+yy*s, -xx*s+yy*c
    probability = result/mass
    along -= (along*probability).sum()
    across -= (across*probability).sum()
    effective_parallel = float((along.square()*probability).sum())/(2*dt_seconds)
    effective_perp = float((across.square()*probability).sum())/(2*dt_seconds)
    return {"retained_mass":mass, "effective_parallel_m2s":effective_parallel,
        "effective_perp_m2s":effective_perp,
        "cross_covariance_m2":float((along*across*probability).sum()),
        "parallel_variance_ratio":effective_parallel/parallel,
        "perp_variance_ratio":effective_perp/perpendicular,
        "perp_sigma_over_max_grid":float(np.sqrt(2*perpendicular*dt_seconds)/max(dx_m,dy_m)),
        "calm_isotropic_fallback":bool(speed <= 1e-12 and solver == "anisotropic")}


def analytic_release_predictions(model, xyt, source, diffusion, knots, coefficients,
                                 sigma, subdivisions):
    """Full-space Gaussian reference isolating release quadrature, not a grid replacement.

    Piecewise-constant interval winds and tensors are integrated exactly for
    each released parcel. Only the source-time integral uses a trapezoidal rule.
    """
    if subdivisions < 1 or int(subdivisions) != subdivisions or sigma <= 0:
        raise ValueError("Positive integer subdivisions and source width required")
    times = model.recurrent_times.detach().cpu().numpy().astype(float)
    wind = np.column_stack([model.recurrent_u.cpu().numpy(), model.recurrent_v.cpu().numpy()]).astype(float)
    parallel, perp = diffusion
    if parallel < perp or perp <= 0:
        raise ValueError("Require parallel >= perpendicular > 0")
    tensors = []
    for velocity in wind:
        speed = np.linalg.norm(velocity)
        direction = velocity/speed if speed > 1e-12 else np.array([1., 0.])
        tensor = perp*np.eye(2)+(parallel-perp)*np.outer(direction, direction)
        tensors.append(tensor if speed > 1e-12 else .5*(parallel+perp)*np.eye(2))
    tensors = np.asarray(tensors)
    predictions = []
    for x, y, end in np.asarray(xyt):
        releases, weights = [], []
        for start, stop in zip(times[:-1], times[1:]):
            stop = min(stop, end)
            if stop <= start:
                continue
            nodes = np.linspace(start, stop, subdivisions+1)
            weight = np.full(subdivisions+1, (stop-start)/subdivisions)
            weight[[0, -1]] *= .5
            releases.extend(nodes)
            weights.extend(weight*np.interp(nodes, knots, coefficients))
        initial_weight = (field.RECURRENT_INITIAL_RELEASE_FRACTION *
                          model.recurrent_initial_release_dt * np.interp(times[0], knots, coefficients))
        releases.append(times[0])
        weights.append(initial_weight)
        release = np.asarray(releases)
        residence = np.maximum(0., np.minimum(times[1:, None], end)-
                               np.maximum(times[:-1, None], release[None, :]))
        displacement = residence.T @ wind[:-1]
        covariance = sigma**2*np.eye(2)[None, :, :] + 2*np.einsum("ir,ijk->rjk", residence, tensors[:-1])
        delta = np.array([x, y])-np.asarray(source)-displacement
        exponent = np.einsum("ri,rij,rj->r", delta, np.linalg.inv(covariance), delta)
        kernel = np.exp(-.5*exponent)/(2*np.pi*np.sqrt(np.linalg.det(covariance)))
        attenuation = np.exp(-model.recurrent_decay_norm*(end-release))
        predictions.append(field.RECURRENT_SOURCE_SCALE*np.dot(weights, kernel*attenuation))
    return np.asarray(predictions)


def isolated_operator_moments(model, diffusion, subdivisions=(1, 2, 4), size=129):
    """Same grid spacing, extended domain, source-free impulse operator checks."""
    if size < 3 or size % 2 != 1:
        raise ValueError("Odd impulse grid size >= 3 required")
    dx = float(model.recurrent_x_grid[1]-model.recurrent_x_grid[0])
    dy = float(model.recurrent_y_grid[1]-model.recurrent_y_grid[0])
    coordinates = torch.arange(size, dtype=torch.float64)-size//2
    xgrid, ygrid = coordinates*dx, coordinates*dy
    yy, xx = torch.meshgrid(ygrid, xgrid, indexing="ij")
    impulse = torch.zeros((size, size), dtype=torch.float64)
    impulse[size//2, size//2] = 1.
    times = model.recurrent_times.detach().cpu().numpy()
    rows = []
    def moments(values):
        mass = float(values.sum())
        probability = values/max(mass, 1e-300)
        cx, cy = float((probability*xx).sum()), float((probability*yy).sum())
        return {"retained_mass": mass, "center_x_norm": cx, "center_y_norm": cy,
                "variance_x_norm2": float((probability*(xx-cx)**2).sum()),
                "variance_y_norm2": float((probability*(yy-cy)**2).sum())}
    for interval in range(len(times)-1):
        duration = float(times[interval+1]-times[interval])
        u, v = float(model.recurrent_u[interval]), float(model.recurrent_v[interval])
        speed = np.hypot(u, v)
        direction = np.array([u, v])/speed if speed > 1e-12 else np.array([1., 0.])
        tensor = diffusion[1]*np.eye(2)+(diffusion[0]-diffusion[1])*np.outer(direction, direction)
        if speed <= 1e-12:
            tensor = .5*sum(diffusion)*np.eye(2)
        for steps in subdivisions:
            if steps < 1 or int(steps) != steps:
                raise ValueError("Positive integer subdivisions required")
            dt = torch.tensor(duration/steps, dtype=torch.float64)
            advected, diffused = impulse.clone(), impulse.clone()
            implicit = impulse.clone() if np.isclose(*diffusion, rtol=1e-5, atol=0.) else None
            for _ in range(steps):
                advected = field._advect_field(advected, xgrid, ygrid, xx.reshape(-1), yy.reshape(-1),
                    torch.tensor(u), torch.tensor(v), dt)
                diffused = field._diffuse_field_anisotropic(diffused, xgrid, ygrid,
                    torch.tensor(diffusion[0]), torch.tensor(diffusion[1]), torch.tensor(u), torch.tensor(v), dt)
                if implicit is not None:
                    implicit = field._diffuse_field_implicit(implicit, xgrid, ygrid,
                        torch.tensor(.5*sum(diffusion)), dt)
            variants = [("advection_only", advected), ("diffusion_only", diffused)]
            if implicit is not None:
                variants.append(("implicit_isotropic_control", implicit))
            for label, values in variants:
                rows.append({"operator": label, "interval": interval, "substeps": steps,
                    "dx_norm": dx, "dy_norm": dy, "duration_norm": duration,
                    "expected_center_x_norm": u*duration if label == "advection_only" else 0.,
                    "expected_center_y_norm": v*duration if label == "advection_only" else 0.,
                    "expected_variance_x_norm2": 0. if label == "advection_only" else 2*duration*tensor[0, 0],
                    "expected_variance_y_norm2": 0. if label == "advection_only" else 2*duration*tensor[1, 1],
                    **moments(values)})
    return rows
