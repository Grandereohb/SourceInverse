"""Identifiability-aware variable-projection differentiable inversion."""

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from functools import wraps
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from variable_projection import (
    build_q_basis,
    build_release_layout_family,
    build_response_matrix,
    generate_candidate_grid,
    json_safe,
    resolve_q_basis_count,
    solve_nonnegative_q,
    inversion_timing,
    timing_stage,
)
from config import IA_VP_JACOBIAN_BATCH_SIZE, IA_VP_RESPONSE_BATCH_SIZE


@dataclass(frozen=True)
class IAVPDICandidate:
    source_x: float
    source_y: float
    diffusion_parallel_norm: float
    diffusion_perp_norm: float
    coefficients: np.ndarray
    q_integration: np.ndarray
    prediction: np.ndarray
    profile_score: float
    weighted_sse: float
    response_rank: int
    response_condition: float
    solver_success: bool
    solver_status: int
    solver_message: str
    seed_label: str = ""
    forward_count: int = 0
    layout_id: str = "uniform_base"
    q_knots: tuple = ()


@dataclass
class IAVPDIRefinementRun:
    run_id: int
    start: IAVPDICandidate
    best: IAVPDICandidate | None
    success: bool
    steps: int
    message: str
    trajectory: list = field(default_factory=list)
    forward_count: int = 0
    inner_solve_count: int = 0


@dataclass(frozen=True)
class ProfileInformationDiagnostic:
    status: str
    theta_names: tuple
    active_q_indices: tuple
    full_information: np.ndarray
    profiled_theta_information: np.ndarray
    profiled_source_information: np.ndarray
    covariance_source_norm: np.ndarray | None
    covariance_source_m2: np.ndarray | None
    rank: int
    condition_number: float
    sigma2: float
    ellipse_major_m: float | None
    ellipse_minor_m: float | None
    ellipse_angle_deg: float | None
    ellipse_area_m2: float | None
    weakest_direction: tuple | None
    boundary_sensitive: bool
    warnings: tuple
    structural_status: str = "unavailable"
    localization_reference_m: float | None = None
    ellipse_to_reference_ratio: float | None = None
    precision_assessment: str = "unavailable"


@dataclass(frozen=True)
class SensorRecommendation:
    source_x: float
    source_y: float
    ellipse_area_m2: float | None
    area_reduction_fraction: float | None
    logdet_gain: float | None
    label: str = "information-only recommendation"


@dataclass
class IAVPDIResult:
    best: IAVPDICandidate
    coarse_candidates: tuple
    refinement_runs: tuple
    q_basis: np.ndarray
    q_basis_type: str
    information: ProfileInformationDiagnostic | None
    sensor_recommendations: tuple
    elapsed_seconds: float
    forward_count: int
    exports: dict = field(default_factory=dict)
    q_upper_bound: float = float("inf")
    q_knots: tuple = ()
    start_strategy: str = "score"
    start_min_distance: float = 0.0
    q_layout_strategy: str = "uniform"
    layout_definitions: tuple = ()
    inner_solve_count: int = 0
    final_layout_candidates: tuple = ()
    timing: dict = field(default_factory=dict)


def _profile_inversion(function):
    @wraps(function)
    def call(*args, **kwargs):
        with inversion_timing() as record:
            result = function(*args, **kwargs)
        result.timing = record
        result.timing['note'] = 'Phase totals include component times; overlapping stages must not be added together.'
        for name, stage in record['stages'].items():
            print(f"IA-VPDI timing: {name}={stage['seconds']:.3f}s, calls={stage['calls']}", flush=True)
        return result
    return call


def softplus_inverse(x):
    """Stable inverse of softplus for strictly positive scalars/tensors."""
    value = torch.as_tensor(x)
    tiny = torch.finfo(value.dtype).tiny if torch.is_floating_point(value) else 1e-12
    value = torch.clamp(value, min=tiny)
    return value + torch.log(-torch.expm1(-value))


def model_anisotropic_diffusion_norm(model):
    parallel_multiplier, perp_multiplier = model.anisotropic_D()
    scale = float(getattr(model, "recurrent_d_scale_norm", 1.0))
    d_perp = float(getattr(model, "recurrent_d_perp_min_norm", 0.0)) + scale * perp_multiplier
    d_parallel_candidate = (
        float(getattr(model, "recurrent_d_parallel_min_norm", 0.0))
        + scale * parallel_multiplier
    )
    return torch.maximum(d_parallel_candidate, d_perp), d_perp


def set_model_anisotropic_diffusion_norm(model, d_parallel_norm, d_perp_norm):
    """Set normalized anisotropic diffusivities through the constrained raw parameters."""
    scale = float(getattr(model, "recurrent_d_scale_norm", 1.0))
    if scale <= 0.0:
        raise ValueError("recurrent_d_scale_norm must be positive.")
    min_parallel = float(getattr(model, "recurrent_d_parallel_min_norm", 0.0))
    min_perp = float(getattr(model, "recurrent_d_perp_min_norm", 0.0))
    d_perp = float(d_perp_norm)
    d_parallel = float(d_parallel_norm)
    if not np.isfinite([d_parallel, d_perp]).all() or d_perp <= min_perp:
        raise ValueError("D_perp must be finite and above its configured minimum.")
    if d_parallel < d_perp:
        raise ValueError("D_parallel must be greater than or equal to D_perp.")
    perp_multiplier = (d_perp - min_perp) / scale
    delta_multiplier = (d_parallel - min_parallel) / scale - perp_multiplier
    if delta_multiplier < -1e-12:
        raise ValueError("Requested diffusivities are incompatible with configured minima.")
    dtype, device = model.rawD_perp.dtype, model.rawD_perp.device
    eps = 1.0e-6
    with torch.no_grad():
        model.rawD_perp.copy_(
            softplus_inverse(torch.tensor(max(perp_multiplier - eps, 1e-12), dtype=dtype, device=device))
        )
        model.rawD_parallel_delta.copy_(
            softplus_inverse(torch.tensor(max(delta_multiplier, 1e-12), dtype=dtype, device=device))
        )
    return tuple(float(v.detach().cpu()) for v in model_anisotropic_diffusion_norm(model))


@contextmanager
def temporary_source_and_diffusion(model, source_x, source_y, d_parallel_norm, d_perp_norm):
    saved = {
        "xs": model.xs.detach().clone(),
        "ys": model.ys.detach().clone(),
        "rawD_perp": model.rawD_perp.detach().clone(),
        "rawD_parallel_delta": model.rawD_parallel_delta.detach().clone(),
    }
    try:
        with torch.no_grad():
            model.xs.fill_(float(source_x))
            model.ys.fill_(float(source_y))
        set_model_anisotropic_diffusion_norm(model, d_parallel_norm, d_perp_norm)
        yield
    finally:
        with torch.no_grad():
            for name, value in saved.items():
                getattr(model, name).copy_(value)


def _weights_numpy(weights, n):
    if weights is None:
        return np.ones(int(n), dtype=np.float64)
    value = np.asarray(
        weights.detach().cpu().numpy() if isinstance(weights, torch.Tensor) else weights,
        dtype=np.float64,
    ).reshape(-1)
    if value.size != int(n):
        raise ValueError("weights must match observations.")
    return value


def score_all_data_candidate(
    model,
    sigma_src,
    observation_xyt,
    observations,
    q_basis,
    source_x,
    source_y,
    diffusion_parallel_norm,
    diffusion_perp_norm,
    weights=None,
    seed_label="",
    q_upper_bound=np.inf,
):
    """Profile nonnegative Q using every supplied valid observation."""
    with temporary_source_and_diffusion(
        model, source_x, source_y, diffusion_parallel_norm, diffusion_perp_norm
    ):
        response = build_response_matrix(model, sigma_src, observation_xyt, q_basis)
    solution = solve_nonnegative_q(response, observations, weights=weights,
                                   coefficient_upper_bound=q_upper_bound)
    return _candidate_from_solution(solution, q_basis, source_x, source_y,
        diffusion_parallel_norm, diffusion_perp_norm, seed_label)


def _candidate_from_solution(solution, q_basis, source_x, source_y,
                             diffusion_parallel_norm, diffusion_perp_norm,
                             seed_label="", *, forward_count=None, layout_id="uniform_base",
                             q_knots=()):
    basis = np.asarray(
        q_basis.detach().cpu().numpy() if isinstance(q_basis, torch.Tensor) else q_basis,
        dtype=np.float64,
    )
    return IAVPDICandidate(
        source_x=float(source_x),
        source_y=float(source_y),
        diffusion_parallel_norm=float(diffusion_parallel_norm),
        diffusion_perp_norm=float(diffusion_perp_norm),
        coefficients=solution.coefficients,
        q_integration=basis @ solution.coefficients,
        prediction=solution.prediction,
        profile_score=solution.weighted_sse / max(solution.residual.size, 1),
        weighted_sse=solution.weighted_sse,
        response_rank=solution.response_rank,
        response_condition=solution.response_condition,
        solver_success=solution.success,
        solver_status=solution.status,
        solver_message=solution.message,
        seed_label=str(seed_label),
        forward_count=int(basis.shape[1] if forward_count is None else forward_count),
        layout_id=layout_id,
        q_knots=tuple(q_knots),
    )


def score_release_layouts(model, sigma_src, observation_xyt, observations, family,
                         source_x, source_y, diffusion_parallel_norm, diffusion_perp_norm,
                         weights=None, seed_label="", q_upper_bound=np.inf):
    """Profile every clock-only layout using one common linear transport span."""
    with temporary_source_and_diffusion(model, source_x, source_y,
                                        diffusion_parallel_norm, diffusion_perp_norm):
        response = build_response_matrix(model, sigma_src, observation_xyt, family.response_basis)
    candidates = []
    for index, layout in enumerate(family.layouts):
        mapped = response @ layout.mapping.to(device=response.device, dtype=response.dtype)
        solution = solve_nonnegative_q(mapped, observations, weights=weights,
                                        coefficient_upper_bound=q_upper_bound)
        candidates.append(_candidate_from_solution(solution, layout.basis, source_x, source_y,
            diffusion_parallel_norm, diffusion_perp_norm, seed_label,
            forward_count=family.response_basis.shape[1] if index == 0 else 0,
            layout_id=layout.layout_id, q_knots=layout.knots))
    return tuple(candidates)


def build_diffusion_seeds(model, scale_seeds=(0.5, 1.0, 2.0), ratio_seeds=(1.0, 3.0)):
    d_parallel, d_perp = model_anisotropic_diffusion_norm(model)
    base = max(0.5 * float((d_parallel + d_perp).detach().cpu()), 1e-8)
    min_perp = float(getattr(model, "recurrent_d_perp_min_norm", 0.0))
    min_parallel = float(getattr(model, "recurrent_d_parallel_min_norm", 0.0))
    scale = float(getattr(model, "recurrent_d_scale_norm", 1.0))
    seeds = []
    for scale_seed in scale_seeds:
        for ratio in ratio_seeds:
            perp = max(base * float(scale_seed), min_perp + 1e-5 * scale)
            parallel = max(perp * float(ratio), min_parallel + (perp - min_perp) + 1e-5 * scale)
            seeds.append((f"scale_{scale_seed:g}_ratio_{ratio:g}", parallel, perp))
    return tuple(seeds)


def refine_profiled_candidate(
    model,
    sigma_src,
    observation_xyt,
    observations,
    q_basis,
    start,
    source_x_bounds,
    source_y_bounds,
    weights=None,
    steps=200,
    lr=1e-2,
    patience=30,
    min_delta=1e-8,
    max_grad_norm=10.0,
    run_id=0,
    q_upper_bound=np.inf,
    optimize_diffusion=True,
):
    """Refine theta with the envelope gradient while re-solving NNLS each step.

    The gradient is valid locally when the NNLS optimum is unique and its active
    set is unchanged. Active-set switches make the profiled objective piecewise
    smooth; SciPy itself is intentionally outside autograd.
    """
    set_model_anisotropic_diffusion_norm(
        model, start.diffusion_parallel_norm, start.diffusion_perp_norm
    )
    with torch.no_grad():
        model.xs.fill_(start.source_x)
        model.ys.fill_(start.source_y)
    params = [model.xs, model.ys]
    if optimize_diffusion:
        params += [model.rawD_perp, model.rawD_parallel_delta]
    previous_requires = [param.requires_grad for param in params]
    for param in params:
        param.requires_grad_(True)
    optimizer = torch.optim.Adam(params, lr=float(lr))
    obs_t = torch.as_tensor(observations, dtype=model.xs.dtype, device=model.xs.device).reshape(-1)
    weight_t = torch.as_tensor(
        _weights_numpy(weights, obs_t.numel()), dtype=model.xs.dtype, device=model.xs.device
    )
    basis_np = np.asarray(
        q_basis.detach().cpu().numpy() if isinstance(q_basis, torch.Tensor) else q_basis,
        dtype=np.float64,
    )
    best = start if np.isinf(q_upper_bound) else None
    wait = 0
    trajectory = []
    forward_count = 0
    inner_solve_count = 0
    message = "maximum steps reached"
    try:
        for step_index in range(1, max(int(steps), 0) + 1):
            optimizer.zero_grad()
            response = build_response_matrix(
                model, sigma_src, observation_xyt, q_basis, track_gradients=True
            )
            forward_count += basis_np.shape[1]
            inner_solve_count += 1
            solution = solve_nonnegative_q(response.detach(), observations, weights=weights,
                                           coefficient_upper_bound=q_upper_bound)
            if not solution.success:
                raise RuntimeError(f"inner Q solve failed: {solution.message}")
            coefficients = torch.as_tensor(
                solution.coefficients, dtype=response.dtype, device=response.device
            )
            prediction = response @ coefficients
            loss = torch.mean((weight_t * (prediction - obs_t)) ** 2)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite profiled objective")
            loss.backward()
            if max_grad_norm is not None and max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(params, float(max_grad_norm))
            # Snapshot the evaluated state before Adam changes its parameters.
            d_parallel, d_perp = model_anisotropic_diffusion_norm(model)
            score = solution.weighted_sse / obs_t.numel()
            row = {
                "run_id": int(run_id),
                "step": step_index,
                "profile_score": score,
                "source_x": float(model.xs.detach().cpu()),
                "source_y": float(model.ys.detach().cpu()),
                "d_parallel_norm": float(d_parallel.detach().cpu()),
                "d_perp_norm": float(d_perp.detach().cpu()),
                "q_max": float(np.max(basis_np @ solution.coefficients)),
                "coefficients_at_cap": int(np.sum(solution.coefficients >= q_upper_bound * (1. - 1e-6)))
                if np.isfinite(q_upper_bound) else 0,
                "gradient_norm": float(
                    np.sqrt(sum(float(p.grad.detach().norm().cpu()) ** 2 for p in params if p.grad is not None))
                ),
            }
            trajectory.append(row)
            if best is None or score < best.profile_score - float(min_delta):
                best = IAVPDICandidate(
                    source_x=row["source_x"], source_y=row["source_y"],
                    diffusion_parallel_norm=row["d_parallel_norm"],
                    diffusion_perp_norm=row["d_perp_norm"],
                    coefficients=solution.coefficients.copy(),
                    q_integration=basis_np @ solution.coefficients,
                    prediction=solution.prediction.copy(),
                    profile_score=score,
                    weighted_sse=score * obs_t.numel(),
                    response_rank=solution.response_rank,
                    response_condition=solution.response_condition,
                    solver_success=solution.success,
                    solver_status=solution.status,
                    solver_message=solution.message,
                    seed_label=start.seed_label,
                    forward_count=basis_np.shape[1],
                    layout_id=start.layout_id,
                    q_knots=start.q_knots,
                )
                wait = 0
            else:
                wait += 1
            if step_index >= 2 and wait >= int(patience):
                message = "profile objective stabilized"
                break
            optimizer.step()
            with torch.no_grad():
                model.xs.clamp_(*map(float, source_x_bounds))
                model.ys.clamp_(*map(float, source_y_bounds))
            if not all(bool(torch.all(torch.isfinite(param))) for param in params):
                raise FloatingPointError("non-finite outer parameter")
        # The final update has not been scored inside the loop.
        d_parallel, d_perp = model_anisotropic_diffusion_norm(model)
        inner_solve_count += 1
        final = score_all_data_candidate(
            model, sigma_src, observation_xyt, observations, q_basis,
            float(model.xs.detach().cpu()), float(model.ys.detach().cpu()),
            float(d_parallel.detach().cpu()), float(d_perp.detach().cpu()),
            weights, start.seed_label, q_upper_bound,
        )
        forward_count += basis_np.shape[1]
        if not final.solver_success:
            raise RuntimeError(f"final Q solve failed: {final.solver_message}")
        if best is None or final.profile_score < best.profile_score:
            best = replace(final, layout_id=start.layout_id, q_knots=start.q_knots)
        set_model_anisotropic_diffusion_norm(
            model, best.diffusion_parallel_norm, best.diffusion_perp_norm
        )
        with torch.no_grad():
            model.xs.fill_(best.source_x)
            model.ys.fill_(best.source_y)
        inner_solve_count += 1
        best = score_all_data_candidate(
            model, sigma_src, observation_xyt, observations, q_basis,
            best.source_x, best.source_y, best.diffusion_parallel_norm,
            best.diffusion_perp_norm, weights, start.seed_label, q_upper_bound,
        )
        best = replace(best, layout_id=start.layout_id, q_knots=start.q_knots)
        forward_count += basis_np.shape[1]
        return IAVPDIRefinementRun(
            int(run_id), start, best, True, len(trajectory), message, trajectory, forward_count,
            inner_solve_count,
        )
    except Exception as exc:
        return IAVPDIRefinementRun(
            int(run_id), start, best, False, len(trajectory), str(exc), trajectory, forward_count,
            inner_solve_count,
        )
    finally:
        for param, requires_grad in zip(params, previous_requires):
            param.requires_grad_(requires_grad)


def _prediction_jacobian_raw(model, prediction, batch_size=None):
    params = [model.xs, model.ys, model.rawD_perp, model.rawD_parallel_delta]
    if batch_size is None:
        batch_size = int(os.environ.get('PINN_IA_VP_JACOBIAN_BATCH_SIZE', IA_VP_JACOBIAN_BATCH_SIZE))
    if int(batch_size) != batch_size or batch_size < 1:
        raise ValueError('Jacobian batch_size must be a positive integer')
    if batch_size > 1:
        rows = []
        for start in range(0, prediction.numel(), batch_size):
            indices = torch.arange(start, min(start+batch_size, prediction.numel()), device=prediction.device)
            signals = prediction.new_zeros((indices.numel(), prediction.numel()))
            signals[torch.arange(indices.numel(), device=prediction.device), indices] = 1.
            grads = torch.autograd.grad(prediction.reshape(-1), params, grad_outputs=signals,
                retain_graph=True, allow_unused=False, is_grads_batched=True)
            rows.append(torch.stack([grad.reshape(indices.numel()) for grad in grads], dim=1))
        return torch.cat(rows, dim=0)
    rows = []
    for index in range(prediction.numel()):
        grads = torch.autograd.grad(
            prediction[index], params, retain_graph=True, allow_unused=False
        )
        rows.append(torch.stack([grad.reshape(()) for grad in grads]))
    return torch.stack(rows, dim=0)


def _raw_to_physical_jacobian(model):
    raw = torch.stack([model.rawD_perp, model.rawD_parallel_delta])

    def mapping(values):
        perp_multiplier = torch.nn.functional.softplus(values[0]) + 1e-6
        parallel_multiplier = perp_multiplier + torch.nn.functional.softplus(values[1])
        scale = float(getattr(model, "recurrent_d_scale_norm", 1.0))
        d_perp = float(getattr(model, "recurrent_d_perp_min_norm", 0.0)) + scale * perp_multiplier
        candidate = float(getattr(model, "recurrent_d_parallel_min_norm", 0.0)) + scale * parallel_multiplier
        return torch.stack([torch.maximum(candidate, d_perp), d_perp])

    return torch.autograd.functional.jacobian(mapping, raw)


def _numpy_double(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _column_space(matrix, rcond):
    """Use scaled columns so nuisance units do not determine numerical rank."""
    norms = np.linalg.norm(matrix, axis=0)
    scaled = matrix[:, norms > 0] / norms[norms > 0]
    if not scaled.size:
        return np.zeros((matrix.shape[0], 0), dtype=np.float64)
    u, singular, _ = np.linalg.svd(scaled, full_matrices=False)
    return u[:, singular > float(rcond) * singular[0]]


def _project_out(jacobian, nuisance, rcond):
    space = _column_space(nuisance, rcond)
    return jacobian - space @ (space.T @ jacobian)


def _theta_jacobian_numpy(model, prediction, rcond):
    raw = _numpy_double(_prediction_jacobian_raw(model, prediction))
    transform = _numpy_double(_raw_to_physical_jacobian(model))
    return np.concatenate([
        raw[:, :2], raw[:, 2:] @ np.linalg.pinv(transform, rcond=float(rcond)),
    ], axis=1)


def _source_information_from_full(information, rcond):
    """Recover a Jacobian factor from a scaled PSD Gram matrix for projection."""
    scales = np.sqrt(np.maximum(np.diag(information), 0.))
    scales[scales == 0.] = 1.
    correlation = information / scales[:, None] / scales[None, :]
    eigenvalues, eigenvectors = np.linalg.eigh(.5 * (correlation + correlation.T))
    tolerance = np.finfo(float).eps * max(information.shape) * max(np.max(np.abs(eigenvalues)), 1.)
    if np.min(eigenvalues) < -tolerance:
        raise FloatingPointError("augmented information is not positive semidefinite")
    factor = (np.sqrt(np.maximum(eigenvalues, 0.))[:, None] * eigenvectors.T) * scales
    projected = _project_out(factor[:, :2], factor[:, 2:], rcond)
    return projected.T @ projected


def compute_profile_information(
    model,
    sigma_src,
    observation_xyt,
    observations,
    q_basis,
    coefficients,
    weights=None,
    active_q_tol=1e-8,
    rcond=1e-10,
    condition_max=1e8,
    length_m=1.0,
    source_bounds=None,
    boundary_margin_norm=0.0,
    q_upper_bound=np.inf,
    localization_radius_limit_m=None,
):
    if not np.isfinite(length_m) or length_m <= 0:
        raise ValueError("length_m must be finite and positive")
    if localization_radius_limit_m is not None and (
        not np.isfinite(localization_radius_limit_m) or localization_radius_limit_m <= 0
    ):
        raise ValueError("localization_radius_limit_m must be finite and positive")
    reference_m = localization_radius_limit_m
    if source_bounds is not None:
        bounds = np.asarray(source_bounds, dtype=float)
        if bounds.shape != (2, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 1] <= bounds[:, 0]):
            raise ValueError("source_bounds must contain two finite increasing intervals")
        if reference_m is None:
            reference_m = float(np.linalg.norm(bounds[:, 1] - bounds[:, 0]) * length_m)
    response = build_response_matrix(
        model, sigma_src, observation_xyt, q_basis, track_gradients=True
    )
    coefficient_t = torch.as_tensor(coefficients, dtype=response.dtype, device=response.device)
    prediction = response @ coefficient_t
    j_theta = _theta_jacobian_numpy(model, prediction, rcond)
    response_np = response.detach().cpu().numpy().astype(np.float64)
    active = np.flatnonzero(_numpy_double(coefficients) > float(active_q_tol))
    j_q = response_np[:, active]
    observed = _numpy_double(observations).reshape(-1)
    weight = _weights_numpy(weights, observed.size)
    jw_theta = j_theta * weight[:, None]
    jw_q = j_q * weight[:, None]
    f_tt = jw_theta.T @ jw_theta
    f_tq = jw_theta.T @ jw_q
    f_qq = jw_q.T @ jw_q
    # Project Jacobians, not normal matrices: avoid cancellation in Schur subtraction.
    projected_theta = _project_out(jw_theta, jw_q, rcond)
    profiled_theta = projected_theta.T @ projected_theta
    projected_source = _project_out(
        jw_theta[:, :2], np.concatenate([jw_q, jw_theta[:, 2:]], axis=1), rcond,
    )
    profiled_source = projected_source.T @ projected_source
    singular = np.linalg.svd(projected_source, compute_uv=False)
    reference = np.linalg.norm(jw_theta[:, :2], ord=2)
    # The reference also detects roundoff residuals after a full nuisance projection.
    rank = int(np.count_nonzero(singular > float(rcond) * reference)) if reference > 0 else 0
    condition = float(np.linalg.cond(profiled_source)) if rank == 2 else float("inf")
    residual = prediction.detach().cpu().numpy() - observed
    effective = _column_space(np.concatenate([jw_theta, jw_q], axis=1), rcond).shape[1]
    dof = int(np.count_nonzero(weight) - effective)
    sigma2 = float(np.sum((weight * residual) ** 2) / max(dof, 1))
    warnings = []
    covariance_norm = covariance_m2 = None
    major = minor = angle = area = None
    weakest = None
    if dof <= 0:
        warnings.append("insufficient residual degrees of freedom for noise estimation")
    zero_noise = sigma2 <= np.finfo(float).eps * max(float(np.mean((weight * observed) ** 2)), np.finfo(float).tiny)
    if zero_noise:
        warnings.append("zero estimated noise: a zero-area ellipse is not evidence of exact localization")
    warnings.append("ellipse is a local linear approximation conditional on the active Q set and noise model")
    if np.isfinite(q_upper_bound) and np.any(_numpy_double(coefficients) >= q_upper_bound * (1. - 1e-6)):
        warnings.append("Q upper bound is active; unconstrained-Q ellipse is not calibrated for bounded inference")
    if rank < 2:
        status = "rank_deficient"
        warnings.append("profiled source information is rank deficient")
    elif dof <= 0:
        status = "weakly_identifiable"
    else:
        covariance_norm = sigma2 * np.linalg.inv(profiled_source)
        covariance_norm = 0.5 * (covariance_norm + covariance_norm.T)
        covariance_m2 = covariance_norm * float(length_m) ** 2
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_m2)
        if np.min(eigenvalues) < 0:
            raise FloatingPointError("source covariance is not positive semidefinite")
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
        major, minor = np.sqrt(5.991 * eigenvalues)
        direction = eigenvectors[:, 0]
        angle = float(np.degrees(np.arctan2(direction[1], direction[0])))
        area = float(np.pi * major * minor)
        weakest = tuple(float(v) for v in direction)
        status = "identifiable" if np.isfinite(condition) and condition <= condition_max else "weakly_identifiable"
        if status != "identifiable":
            warnings.append("profiled source information is ill-conditioned")
    structural_status = status
    precision = "unavailable"
    ellipse_ratio = None
    if major is not None and reference_m is not None:
        ellipse_ratio = float(major / reference_m)
        precision = "within_reference_scale" if ellipse_ratio <= 1. else "exceeds_reference_scale"
        if ellipse_ratio > 1.:
            if status == "identifiable":
                status = "weakly_identifiable"
            warnings.append("local uncertainty radius exceeds the localization reference scale; full rank does not establish practical precision")
    if zero_noise:
        precision = "noise_not_established"
        if status == "identifiable":
            status = "weakly_identifiable"
    boundary_sensitive = False
    if source_bounds is not None:
        x_bounds, y_bounds = source_bounds
        x, y = float(model.xs.detach().cpu()), float(model.ys.detach().cpu())
        boundary_sensitive = min(x - x_bounds[0], x_bounds[1] - x, y - y_bounds[0], y_bounds[1] - y) <= float(boundary_margin_norm)
        if boundary_sensitive:
            status = "boundary_sensitive"
            warnings.append("best source is close to the search boundary")
    if np.isfinite(q_upper_bound) and np.any(_numpy_double(coefficients) >= q_upper_bound * (1. - 1e-6)):
        covariance_norm = covariance_m2 = major = minor = angle = area = None
        precision = "amplitude_bound_active"
        ellipse_ratio = None
        if status == "identifiable":
            status = "weakly_identifiable"
    full = np.block([[f_tt, f_tq], [f_tq.T, f_qq]])
    return ProfileInformationDiagnostic(
        status=status,
        theta_names=("source_x", "source_y", "D_parallel", "D_perp"),
        active_q_indices=tuple(int(v) for v in active),
        full_information=0.5 * (full + full.T),
        profiled_theta_information=profiled_theta,
        profiled_source_information=profiled_source,
        covariance_source_norm=covariance_norm,
        covariance_source_m2=covariance_m2,
        rank=rank,
        condition_number=condition,
        sigma2=sigma2,
        ellipse_major_m=None if major is None else float(major),
        ellipse_minor_m=None if minor is None else float(minor),
        ellipse_angle_deg=angle,
        ellipse_area_m2=area,
        weakest_direction=weakest,
        boundary_sensitive=boundary_sensitive,
        warnings=tuple(warnings),
        structural_status=structural_status,
        localization_reference_m=None if reference_m is None else float(reference_m),
        ellipse_to_reference_ratio=ellipse_ratio,
        precision_assessment=precision,
    )


def recommend_sensor_locations(
    model,
    sigma_src,
    candidate_locations,
    sensor_times,
    q_basis,
    coefficients,
    base_information,
    existing_station_xy=(),
    min_distance_norm=0.0,
    top_k=5,
    length_m=1.0,
    rcond=1e-10,
):
    if base_information.ellipse_area_m2 is None:
        return ()
    recommendations = []
    times = torch.as_tensor(sensor_times, dtype=model.xs.dtype, device=model.xs.device).reshape(-1)
    for x, y in candidate_locations:
        if existing_station_xy and min(
            np.hypot(float(x) - float(ex), float(y) - float(ey)) for ex, ey in existing_station_xy
        ) < float(min_distance_norm):
            continue
        xyt = torch.stack([
            torch.full_like(times, float(x)), torch.full_like(times, float(y)), times
        ], dim=1)
        response = build_response_matrix(model, sigma_src, xyt, q_basis, track_gradients=True)
        prediction = response @ torch.as_tensor(coefficients, dtype=response.dtype, device=response.device)
        j_theta = _theta_jacobian_numpy(model, prediction, rcond)
        active = np.asarray(base_information.active_q_indices, dtype=int)
        j_q = response.detach().cpu().numpy()[:, active]
        j_all = np.concatenate([j_theta, j_q], axis=1)
        augmented = base_information.full_information + j_all.T @ j_all
        f_source = _source_information_from_full(augmented, rcond)
        if np.min(np.linalg.eigvalsh(f_source)) <= 0.:
            continue
        covariance = base_information.sigma2 * np.linalg.inv(f_source) * float(length_m) ** 2
        eig = np.linalg.eigvalsh(0.5 * (covariance + covariance.T))
        if np.min(eig) < 0.:
            raise FloatingPointError("sensor covariance is not positive semidefinite")
        area = float(np.pi * 5.991 * np.sqrt(np.prod(eig)))
        if base_information.ellipse_area_m2 <= np.finfo(float).eps:
            area = 0.0
            reduction = 0.0
        else:
            reduction = max(0.0, 1.0 - area / base_information.ellipse_area_m2)
        sign0, log0 = np.linalg.slogdet(base_information.profiled_source_information)
        sign1, log1 = np.linalg.slogdet(f_source)
        gain = float(log1 - log0) if sign0 > 0 and sign1 > 0 else None
        recommendations.append(SensorRecommendation(float(x), float(y), area, reduction, gain))
    recommendations.sort(key=lambda row: (
        -(row.logdet_gain if row.logdet_gain is not None else -np.inf),
        row.source_x, row.source_y,
    ))
    return tuple(recommendations[: max(int(top_k), 0)])


def select_refinement_starts(candidates, top_k=3, strategy="score", min_distance=0.0):
    """Keep the best fit, then explore separate source neighborhoods at fixed budget.

    If too few neighborhoods exist, relax spacing to distinct coordinates first,
    then use remaining diffusion seeds. Selection never runs the transport model.
    """
    if strategy not in ("score", "spatial_diverse"):
        raise ValueError("Unknown refinement start strategy.")
    if int(top_k) != top_k or top_k < 1:
        raise ValueError("top_k must be a positive integer.")
    if not np.isfinite(min_distance) or min_distance < 0:
        raise ValueError("min_distance must be finite and nonnegative.")
    ranked = sorted((c for c in candidates if c.solver_success and
                     np.all(np.isfinite([c.profile_score, c.source_x, c.source_y]))),
                    key=lambda c: c.profile_score)
    count = min(int(top_k), len(ranked))
    if strategy == "score" or not count:
        return ranked[:count]
    selected = [0]
    def distance(i):
        return min(np.hypot(ranked[i].source_x - ranked[j].source_x,
                            ranked[i].source_y - ranked[j].source_y) for j in selected)
    for threshold in (min_distance, 0.0, None):
        for i in range(1, len(ranked)):
            if len(selected) == count:
                return [ranked[j] for j in selected]
            if i not in selected and (threshold is None or
                                     (distance(i) > 0 and distance(i) >= threshold)):
                selected.append(i)
    return [ranked[j] for j in selected]


@_profile_inversion
def run_ia_vpdi(
    model,
    sigma_src,
    observation_xyt,
    observations,
    source_x_bounds,
    source_y_bounds,
    search_step,
    q_basis_count=0,
    q_basis_type="hat",
    top_k=3,
    diffusion_scale_seeds=(0.5, 1.0, 2.0),
    diffusion_ratio_seeds=(1.0, 3.0),
    weights=None,
    outer_steps=200,
    outer_lr=1e-2,
    outer_patience=30,
    outer_min_delta=1e-8,
    max_grad_norm=10.0,
    q_upper_bound=np.inf,
    q_knots=None,
    start_strategy="score",
    start_min_distance=0.0,
    q_layout_strategy="uniform",
):
    started = time.perf_counter()
    if np.isnan(q_upper_bound) or q_upper_bound <= 0:
        raise ValueError("q_upper_bound must be positive or infinity.")
    select_refinement_starts((), top_k, start_strategy, start_min_distance)
    basis_count = (len(q_knots) if q_knots is not None else
                   resolve_q_basis_count(model.recurrent_integration_times.numel(), q_basis_count))
    q_basis = build_q_basis(model.recurrent_integration_times, basis_count, q_basis_type, knots=q_knots)
    if q_layout_strategy not in ("uniform", "local_templates"):
        raise ValueError("Unknown Q layout strategy.")
    family = None
    if q_layout_strategy == "local_templates":
        if q_knots is not None or q_basis_type != "hat":
            raise ValueError("Local templates require hat basis without custom knots.")
        family = build_release_layout_family(model.recurrent_integration_times,
                                            model.recurrent_times, basis_count)
    layouts = {} if family is None else {layout.layout_id: layout for layout in family.layouts}
    grid = generate_candidate_grid(source_x_bounds, source_y_bounds, search_step)
    seeds = build_diffusion_seeds(model, diffusion_scale_seeds, diffusion_ratio_seeds)
    coarse = []
    selection_pool = []
    evaluated, total = 0, len(grid)*len(seeds)
    coarse_started = time.perf_counter()
    with timing_stage('coarse_search', model.xs.device):
        for label, d_parallel, d_perp in seeds:
            for source_x, source_y in grid:
                if family is None:
                    candidates = (score_all_data_candidate(
                        model, sigma_src, observation_xyt, observations, q_basis,
                        source_x, source_y, d_parallel, d_perp, weights, label, q_upper_bound,
                    ),)
                else:
                    candidates = score_release_layouts(model, sigma_src, observation_xyt,
                        observations, family, source_x, source_y, d_parallel, d_perp,
                        weights, label, q_upper_bound)
                coarse.extend(candidates)
                evaluated += 1
                if evaluated % 25 == 0 or evaluated == total:
                    print(f"IA-VPDI coarse: {evaluated}/{total}, elapsed={time.perf_counter()-coarse_started:.1f}s", flush=True)
                valid_here = [c for c in candidates if c.solver_success and np.isfinite(c.profile_score)]
                if valid_here:
                    selection_pool.append(min(valid_here, key=lambda item: item.profile_score))
    coarse.sort(key=lambda item: item.profile_score)
    valid = [item for item in coarse if item.solver_success and np.isfinite(item.profile_score)]
    if not valid:
        raise RuntimeError("No valid profiled candidates.")
    selection_pool.sort(key=lambda item: item.profile_score)
    starts = select_refinement_starts(selection_pool, top_k, start_strategy, start_min_distance)
    runs = []
    with timing_stage('refinement', model.xs.device):
        for run_id, start in enumerate(starts, 1):
            print(f"IA-VPDI refine: {run_id}/{len(starts)}", flush=True)
            runs.append(refine_profiled_candidate(
                model, sigma_src, observation_xyt, observations,
                q_basis if family is None else layouts[start.layout_id].basis, start,
                source_x_bounds, source_y_bounds, weights, outer_steps, outer_lr,
                outer_patience, outer_min_delta, max_grad_norm, run_id, q_upper_bound,
            ))
            print(f"IA-VPDI refine done: steps={runs[-1].steps}, success={runs[-1].success}", flush=True)
    successful = [run.best for run in runs if run.success and run.best is not None]
    best = min(successful or starts, key=lambda item: item.profile_score)
    additional_forwards = 0
    additional_solves = 0
    final_layout_candidates = []
    if family is not None:
        # Re-profile all layouts at each refined geometry before final selection.
        final_pool = list(starts)
        for candidate in successful:
            rescored = score_release_layouts(model, sigma_src, observation_xyt, observations,
                family, candidate.source_x, candidate.source_y,
                candidate.diffusion_parallel_norm, candidate.diffusion_perp_norm,
                weights, candidate.seed_label, q_upper_bound)
            additional_forwards += sum(c.forward_count for c in rescored)
            additional_solves += len(rescored)
            final_layout_candidates.extend(rescored)
            final_pool.extend(c for c in rescored if c.solver_success and np.isfinite(c.profile_score))
        best = min(final_pool, key=lambda item: item.profile_score)
        selected = layouts[best.layout_id]
        q_basis = selected.basis
        q_knots = selected.knots
        # Direct replay avoids exporting a common-span roundoff discrepancy.
        best = replace(score_all_data_candidate(model, sigma_src, observation_xyt, observations,
            q_basis, best.source_x, best.source_y, best.diffusion_parallel_norm,
            best.diffusion_perp_norm, weights, best.seed_label, q_upper_bound),
            layout_id=selected.layout_id, q_knots=selected.knots)
        additional_forwards += q_basis.shape[1]
        additional_solves += 1
        basis_count = q_basis.shape[1]
    set_model_anisotropic_diffusion_norm(model, best.diffusion_parallel_norm, best.diffusion_perp_norm)
    with torch.no_grad():
        model.xs.fill_(best.source_x)
        model.ys.fill_(best.source_y)
    return IAVPDIResult(
        best=best,
        coarse_candidates=tuple(coarse),
        refinement_runs=tuple(runs),
        q_basis=np.asarray(q_basis.detach().cpu().numpy() if isinstance(q_basis, torch.Tensor) else q_basis),
        q_basis_type=str(q_basis_type),
        information=None,
        sensor_recommendations=(),
        elapsed_seconds=time.perf_counter() - started,
        forward_count=additional_forwards + sum(item.forward_count for item in coarse) + sum(
            run.forward_count for run in runs
        ),
        q_upper_bound=float(q_upper_bound),
        start_strategy=start_strategy,
        start_min_distance=float(start_min_distance),
        q_layout_strategy=q_layout_strategy,
        layout_definitions=tuple({"layout_id": layout.layout_id, "knots": list(layout.knots)}
                                 for layout in (() if family is None else family.layouts)),
        inner_solve_count=len(coarse) + additional_solves + sum(run.inner_solve_count for run in runs),
        final_layout_candidates=tuple(final_layout_candidates),
        q_knots=tuple(np.linspace(float(model.recurrent_integration_times[0]),
                                 float(model.recurrent_integration_times[-1]), basis_count)
                      if q_knots is None else np.asarray(q_knots, dtype=float)),
    )


def _information_dict(info):
    if info is None:
        return None
    payload = asdict(info)
    payload.pop("full_information", None)
    payload["profiled_theta_information"] = info.profiled_theta_information.tolist()
    payload["profiled_source_information"] = info.profiled_source_information.tolist()
    payload["covariance_source_norm"] = None if info.covariance_source_norm is None else info.covariance_source_norm.tolist()
    payload["covariance_source_m2"] = None if info.covariance_source_m2 is None else info.covariance_source_m2.tolist()
    return json_safe(payload)


def sampled_profile_stability(result, *, length_m=None, source_bounds=None,
                              score_tolerances=(.01, .05, .1), boundary_margin_m=250.):
    """Summarize already evaluated candidates without changing source selection."""
    tolerances = np.asarray(score_tolerances, dtype=float)
    if tolerances.ndim != 1 or not np.isfinite(tolerances).all() or np.any(tolerances < 0):
        raise ValueError("Score tolerances must be finite and nonnegative")
    if length_m is not None and (not np.isfinite(length_m) or length_m <= 0):
        raise ValueError("Physical length must be positive and finite")
    if not np.isfinite(boundary_margin_m) or boundary_margin_m < 0:
        raise ValueError("Boundary margin must be finite and nonnegative")
    bounds = None if source_bounds is None else np.asarray(source_bounds, dtype=float)
    if bounds is not None and (bounds.shape != (2, 2) or not np.isfinite(bounds).all()
                               or np.any(bounds[:, 0] > bounds[:, 1])):
        raise ValueError("Source bounds must be finite ordered x/y pairs")

    def boundary(candidate):
        if bounds is None:
            return {"distance_norm": None, "distance_m": None, "at_boundary": None, "near_boundary": None}
        xy = np.array([candidate.source_x, candidate.source_y])
        distance = float(np.min(np.r_[xy-bounds[:, 0], bounds[:, 1]-xy]))
        distance_m = None if length_m is None else distance*length_m
        return {"distance_norm": distance, "distance_m": distance_m,
            "at_boundary": bool(distance <= 4*np.finfo(np.float32).eps),
            "near_boundary": None if distance_m is None else bool(distance_m <= boundary_margin_m)}

    pool = list(result.coarse_candidates) + [result.best]
    pool.extend(run.best for run in result.refinement_runs if run.success and run.best is not None)
    pool.extend(result.final_layout_candidates)
    pool = list({id(candidate): candidate for candidate in pool}.values())
    valid, excluded = [], 0
    for candidate in pool:
        values = [candidate.source_x, candidate.source_y, candidate.profile_score,
                  candidate.diffusion_parallel_norm, candidate.diffusion_perp_norm]
        if (not candidate.solver_success or not np.isfinite(values).all() or candidate.profile_score < 0
                or candidate.q_integration.size == 0 or not np.isfinite(candidate.q_integration).all()
                or np.any(candidate.q_integration < 0)):
            excluded += 1
        else:
            valid.append(candidate)
    if not valid or not any(c is result.best for c in valid):
        raise ValueError("Valid selected solution required for sampled stability report")
    locations = {}
    for candidate in valid:
        key = (candidate.source_x, candidate.source_y)
        if key not in locations or candidate.profile_score < locations[key].profile_score:
            locations[key] = candidate
    reference_score = float(result.best.profile_score)
    sets = []
    for tolerance in tolerances:
        selected = [c for c in locations.values() if c.profile_score <= reference_score*(1+float(tolerance))]
        selected_fits = [c for c in valid if c.profile_score <= reference_score*(1+float(tolerance))]
        xy = np.array([[c.source_x, c.source_y] for c in selected])
        q = np.array([float(c.q_integration.max()) for c in selected_fits])
        span = np.ptp(xy, axis=0)
        sets.append({"relative_mse_tolerance": float(tolerance), "score_limit": reference_score*(1+float(tolerance)),
            "candidate_location_count": len(selected), "evaluated_fit_count": len(selected_fits),
            "x_min_norm": float(xy[:, 0].min()),
            "x_max_norm": float(xy[:, 0].max()), "y_min_norm": float(xy[:, 1].min()), "y_max_norm": float(xy[:, 1].max()),
            "x_span_norm": float(span[0]), "y_span_norm": float(span[1]),
            "x_span_m": None if length_m is None else float(span[0]*length_m),
            "y_span_m": None if length_m is None else float(span[1]*length_m),
            "q_peak_min_model_units": float(q.min()), "q_peak_max_model_units": float(q.max()),
            "q_peak_ratio": float(q.max()/q.min()) if q.min() > 0 else None,
            "contains_zero_q_peak": bool(q.min() == 0),
            "touches_allowed_boundary": None if bounds is None else any(boundary(c)["at_boundary"] for c in selected),
            "near_allowed_boundary": None if bounds is None or length_m is None else any(boundary(c)["near_boundary"] for c in selected)})
    warnings = ["Sampled score sets are not calibrated confidence regions; a single candidate does not establish unique localization.",
                "Candidates cover only the existing coarse search and successful refinement optima, not a dense fixed-physics landscape.",
                "Q ranges are model-unit sensitivity summaries, not validated physical emission uncertainty."]
    best_boundary = boundary(result.best)
    if best_boundary["near_boundary"] or best_boundary["at_boundary"]:
        warnings.append("Selected source is near an allowed boundary; score sets may be truncated by the configured source domain.")
    lowest = min(c.profile_score for c in valid)
    if lowest < reference_score-1e-12*max(1., abs(reference_score)):
        warnings.append("An already evaluated candidate has a lower score than the selected solution; selection was not changed by reporting.")
    return {"reference_selected_score": reference_score, "lowest_evaluated_score": float(lowest),
        "evaluated_record_count": len(pool), "excluded_record_count": excluded,
        "unique_location_count": len(locations), "length_m": length_m,
        "source_bounds_norm": None if bounds is None else bounds.tolist(),
        "boundary_margin_m": boundary_margin_m, "selected_source_boundary": best_boundary,
        "near_equivalent_sets": sets, "warnings": warnings,
        "extra_forward_count": 0, "changes_source_selection": False,
        "nuisance_note": "Location sets retain the lowest evaluated score per location across sampled diffusion settings and temporal layouts. Q ranges include all evaluated fits within the score threshold. Not a continuous profile over all transport parameters; not a calibrated confidence region."}


def build_ia_vpdi_diagnostics(result, *, length_m=None, source_bounds=None,
                               score_tolerances=(.01, .05, .1), boundary_margin_m=250.):
    """Collect tables/report for the pipeline's consolidated XLSX and JSON."""
    stability = sampled_profile_stability(result, length_m=length_m, source_bounds=source_bounds,
        score_tolerances=score_tolerances, boundary_margin_m=boundary_margin_m)
    candidate_rows = []
    for rank, candidate in enumerate((*result.coarse_candidates, *result.final_layout_candidates), 1):
        candidate_rows.append({
            "stage": "coarse" if rank <= len(result.coarse_candidates) else "final_layout_reprofile",
            "rank": rank, "source_x": candidate.source_x, "source_y": candidate.source_y,
            "d_parallel_norm": candidate.diffusion_parallel_norm,
            "d_perp_norm": candidate.diffusion_perp_norm,
            "profile_score": candidate.profile_score, "weighted_sse": candidate.weighted_sse,
            "response_rank": candidate.response_rank,
            "response_condition": candidate.response_condition,
            "solver_success": candidate.solver_success, "seed_label": candidate.seed_label,
            "layout_id": candidate.layout_id,
            "q_basis_count": len(candidate.coefficients),
            "q_knots": json.dumps(list(candidate.q_knots)),
            "q_max": float(candidate.q_integration.max()),
            "relative_score_excess": (candidate.profile_score/result.best.profile_score-1)
            if result.best.profile_score > 0 else None,
            "effective_response_gain": float(np.linalg.norm(candidate.prediction)/np.linalg.norm(candidate.coefficients))
            if np.linalg.norm(candidate.coefficients) > 0 else None,
            "coefficients_at_cap": int(np.sum(candidate.coefficients >= result.q_upper_bound * (1. - 1e-6)))
            if np.isfinite(result.q_upper_bound) else 0,
        })
    run_rows, trajectory_rows = [], []
    for run in result.refinement_runs:
        run_rows.append({
            "run_id": run.run_id, "success": run.success, "steps": run.steps,
            "forward_count": run.forward_count,
            "message": run.message, "start_score": run.start.profile_score,
            "start_source_x": run.start.source_x, "start_source_y": run.start.source_y,
            "start_seed_label": run.start.seed_label,
            "layout_id": run.start.layout_id,
            "best_score": None if run.best is None else run.best.profile_score,
            "best_source_x": None if run.best is None else run.best.source_x,
            "best_source_y": None if run.best is None else run.best.source_y,
        })
        trajectory_rows.extend(run.trajectory)
    candidates_df, runs_df, trajectory_df = map(pd.DataFrame, (candidate_rows, run_rows, trajectory_rows))
    candidates_df = candidates_df.replace([np.inf, -np.inf], np.nan)
    runs_df = runs_df.replace([np.inf, -np.inf], np.nan)
    trajectory_df = trajectory_df.replace([np.inf, -np.inf], np.nan)
    sensors_df = pd.DataFrame([asdict(item) for item in result.sensor_recommendations])
    sensors_df = sensors_df.replace([np.inf, -np.inf], np.nan)
    best = result.best
    report = {
        "mode": "ia_vpdi",
        "best_profiled_score": best.profile_score,
        "q_basis": {"type": result.q_basis_type, "count": int(result.q_basis.shape[1]), "values": result.q_basis.tolist()},
        "q_coefficients": best.coefficients.tolist(),
        "q_upper_model_units": None if np.isinf(result.q_upper_bound) else result.q_upper_bound,
        "q_knots": list(result.q_knots),
        "q_layout_strategy": result.q_layout_strategy,
        "selected_layout_id": best.layout_id,
        "layout_definitions": list(result.layout_definitions),
        "inner_solve_count": result.inner_solve_count,
        "layout_uncertainty_note": "Local information is conditional on the selected temporal layout; it does not include discrete layout-selection uncertainty.",
        "q_coefficients_at_cap": int(np.sum(best.coefficients >= result.q_upper_bound * (1. - 1e-6)))
        if np.isfinite(result.q_upper_bound) else 0,
        "q_bound_note": "Model-unit sensitivity bound, not a validated physical emission limit.",
        "source": {"x_norm": best.source_x, "y_norm": best.source_y},
        "diffusion": {"parallel_norm": best.diffusion_parallel_norm, "perp_norm": best.diffusion_perp_norm},
        "multi_start_stability": {
            "start_strategy": result.start_strategy,
            "requested_min_distance_norm": result.start_min_distance,
            "initial_sources": [[run.start.source_x, run.start.source_y] for run in result.refinement_runs],
            "run_count": len(result.refinement_runs),
            "successful_count": sum(run.success for run in result.refinement_runs),
            "final_sources": [[run.best.source_x, run.best.source_y] for run in result.refinement_runs if run.best is not None],
        },
        "profile_information": _information_dict(result.information),
        "sampled_profile_stability": stability,
        "sensor_recommendations": [asdict(item) for item in result.sensor_recommendations],
        "runtime_seconds": result.elapsed_seconds,
        "timing": result.timing,
        "response_batch_size": int(os.environ.get('PINN_IA_VP_RESPONSE_BATCH_SIZE', IA_VP_RESPONSE_BATCH_SIZE)),
        "jacobian_batch_size": int(os.environ.get('PINN_IA_VP_JACOBIAN_BATCH_SIZE', IA_VP_JACOBIAN_BATCH_SIZE)),
        "forward_count": result.forward_count,
    }
    return {
        "tables": {
            "ia_vpdi_candidates": candidates_df,
            "ia_vpdi_runs": runs_df,
            "ia_vpdi_trajectory": trajectory_df,
            "ia_vpdi_information": pd.DataFrame([_information_dict(result.information) or {}]),
            "ia_vpdi_sensors": sensors_df,
            "ia_vpdi_stability": pd.DataFrame(stability["near_equivalent_sets"]),
        },
        "report": json_safe(report),
    }
