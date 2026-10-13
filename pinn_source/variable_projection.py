"""Reusable low-dimensional variable-projection utilities."""

from contextlib import nullcontext, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
import os
import time

import numpy as np
import torch

from config import IA_VP_RESPONSE_BATCH_SIZE, IA_VP_TIMING_ENABLED
from field import (recurrent_plume_value_from_q, _build_recurrent_tensor_plans,
                   _build_recurrent_gaussian_plans, recurrent_plume_response_batch)


_ACTIVE_TIMING = ContextVar('inversion_timing', default=None)


@contextmanager
def inversion_timing():
    enabled = os.environ.get('PINN_IA_VP_TIMING', str(int(IA_VP_TIMING_ENABLED))) == '1'
    record = {'enabled': enabled, 'measurement': 'synchronized_wall_seconds', 'stages': {}}
    token = _ACTIVE_TIMING.set(record if enabled else None)
    try:
        yield record
    finally:
        _ACTIVE_TIMING.reset(token)


@contextmanager
def timing_stage(name, device=None):
    record = _ACTIVE_TIMING.get()
    if record is None:
        yield
        return
    def sync():
        if device is not None and torch.device(device).type == 'cuda':
            torch.cuda.synchronize(device)
    sync()
    started = time.perf_counter()
    try:
        yield
    finally:
        sync()
        entry = record['stages'].setdefault(name, {'seconds': 0., 'calls': 0})
        entry['seconds'] += time.perf_counter() - started
        entry['calls'] += 1


def timed_component(name):
    def decorate(function):
        @wraps(function)
        def call(*args, **kwargs):
            first = args[0] if args else kwargs.get('model', kwargs.get('response_matrix'))
            device = first.device if isinstance(first, torch.Tensor) else getattr(getattr(first, 'xs', None), 'device', None)
            with timing_stage(name, device):
                return function(*args, **kwargs)
        return call
    return decorate


@dataclass(frozen=True)
class NonnegativeQSolution:
    coefficients: np.ndarray
    prediction: np.ndarray
    residual: np.ndarray
    weighted_sse: float
    success: bool
    status: int
    message: str
    response_rank: int
    response_condition: float


def resolve_q_basis_count(n_times, requested=0):
    n_times = int(n_times)
    if n_times < 1:
        raise ValueError("n_times must be positive.")
    if requested is not None and int(requested) > 0:
        return min(int(requested), n_times)
    return min(n_times, max(3, min(5, int(np.ceil(n_times / 6.0)) + 2)))


def build_q_basis(t_values, n_basis, basis_type="hat", knots=None):
    """Build partition-of-unity hats with uniform or explicit ordered knots."""
    if str(basis_type).lower() != "hat":
        raise ValueError(f"Unsupported Q basis type: {basis_type}")
    is_tensor = isinstance(t_values, torch.Tensor)
    if is_tensor:
        t = t_values.reshape(-1)
        if not torch.is_floating_point(t):
            t = t.to(dtype=torch.float32)
    else:
        values = np.asarray(t_values)
        dtype = torch.float64 if values.dtype == np.float64 else torch.float32
        t = torch.as_tensor(values, dtype=dtype).reshape(-1)
    if t.numel() == 0:
        raise ValueError("t_values must not be empty.")
    n_basis = int(n_basis)
    if n_basis < 1:
        raise ValueError("n_basis must be positive.")
    if knots is not None:
        knots = torch.as_tensor(knots, dtype=t.dtype, device=t.device).reshape(-1)
        if (knots.numel() != n_basis or n_basis < 2
                or not bool(torch.isfinite(knots).all())
                or not bool(torch.all(knots[1:] > knots[:-1]))):
            raise ValueError("knots must be finite, strictly increasing and match n_basis >= 2.")
        if bool(knots[0] > t.min()) or bool(knots[-1] < t.max()):
            raise ValueError("knots must cover the complete time range.")
    if n_basis == 1:
        basis = torch.ones((t.numel(), 1), dtype=t.dtype, device=t.device)
        return basis if is_tensor else basis.cpu().numpy()
    t_min, t_max = torch.min(t), torch.max(t)
    span = t_max - t_min
    if knots is None and float(torch.abs(span).detach().cpu()) <= torch.finfo(t.dtype).eps:
        basis = torch.zeros((t.numel(), n_basis), dtype=t.dtype, device=t.device)
        basis[:, 0] = 1.0
        return basis if is_tensor else basis.cpu().numpy()
    if knots is None:
        knots = torch.linspace(t_min, t_max, n_basis, dtype=t.dtype, device=t.device)
    t_clamped = torch.clamp(t, min=t_min, max=t_max)
    idx_hi = torch.clamp(torch.bucketize(t_clamped.contiguous(), knots), 1, n_basis - 1)
    idx_lo = idx_hi - 1
    alpha = (t_clamped - knots[idx_lo]) / torch.clamp(
        knots[idx_hi] - knots[idx_lo], min=torch.finfo(t.dtype).eps
    )
    basis = torch.zeros((t.numel(), n_basis), dtype=t.dtype, device=t.device)
    basis.scatter_(1, idx_lo[:, None], (1.0 - alpha)[:, None])
    basis.scatter_add_(1, idx_hi[:, None], alpha[:, None])
    return basis if is_tensor else basis.cpu().numpy()


def build_local_release_templates(times):
    """Data-independent six-hat layouts with four adjacent interior time nodes.

    Endpoints remain free, so these are locally dense release layouts, not a
    strict zero-outside-support single-pulse model. No Q or observations input.
    """
    times = np.asarray(times, dtype=float)
    if (times.ndim != 1 or times.size < 6 or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0)):
        raise ValueError("At least six finite strictly increasing times required")
    return [{"template_id": int(i), "retained_indices": [0, i-1, i, i+1, i+2, times.size-1],
             "knots": times[[0, i-1, i, i+1, i+2, times.size-1]].tolist()}
            for i in range(2, times.size-3)]


@dataclass(frozen=True)
class ReleaseLayout:
    layout_id: str
    knots: tuple
    basis: torch.Tensor
    mapping: torch.Tensor


@dataclass(frozen=True)
class ReleaseLayoutFamily:
    response_basis: torch.Tensor
    layouts: tuple
    response_knots: tuple


def build_release_layout_family(integration_times, event_times, n_uniform=5):
    """Clock-only low-dimensional layouts sharing one linear response basis.

    Include the original uniform basis, a uniform-six capacity control, and
    local-six layouts. Use the smaller of the integration-node identity and
    union-knot hats. The latter also represents off-hour uniform knots exactly
    when adaptive integration adds substeps. No observations or fitted Q enter
    this construction. Returned tensors keep the integration dtype/device.
    """
    clock = (integration_times if isinstance(integration_times, torch.Tensor)
             else torch.as_tensor(integration_times, dtype=torch.float64)).reshape(-1)
    if not torch.is_floating_point(clock):
        clock = clock.to(torch.float64)
    event = torch.as_tensor(event_times, device=clock.device, dtype=clock.dtype).reshape(-1).clone()
    for values, minimum in ((clock, 2), (event, 6)):
        if (values.numel() < minimum or not bool(torch.isfinite(values).all())
                or not bool((values[1:] > values[:-1]).all())):
            raise ValueError("Ordered finite integration times and at least six event times required.")
    if (int(n_uniform) != n_uniform or not 2 <= n_uniform <= 6):
        raise ValueError("Uniform release count must be an integer between two and six.")
    if not bool(torch.isclose(clock[[0, -1]], event[[0, -1]], rtol=0., atol=1e-6).all()):
        raise ValueError("Event and integration clocks must have the same endpoints.")
    event[0], event[-1] = clock[0], clock[-1]
    definitions = [("uniform_base", torch.linspace(clock[0], clock[-1], int(n_uniform),
                                                   device=clock.device, dtype=clock.dtype))]
    if n_uniform != 6:
        definitions.append(("uniform_six", torch.linspace(clock[0], clock[-1], 6,
                                                        device=clock.device, dtype=clock.dtype)))
    templates = build_local_release_templates(event.detach().cpu().double().numpy())
    definitions.extend((f"local_{r['template_id']}",
                        torch.as_tensor(r['knots'], device=clock.device, dtype=clock.dtype))
                       for r in templates)
    union = torch.unique(torch.cat([knots for _, knots in definitions]), sorted=True)
    if clock.numel() <= union.numel():
        response_knots = clock
        response_basis = torch.eye(clock.numel(), device=clock.device, dtype=clock.dtype)
    else:
        response_knots = union
        response_basis = build_q_basis(clock, union.numel(), knots=union)
    layouts = tuple(ReleaseLayout(name, tuple(knots.detach().cpu().double().tolist()),
        build_q_basis(clock, knots.numel(), knots=knots),
        build_q_basis(response_knots, knots.numel(), knots=knots)) for name, knots in definitions)
    return ReleaseLayoutFamily(response_basis, layouts,
                               tuple(response_knots.detach().cpu().double().tolist()))


@timed_component('response_matrix')
def build_response_matrix(model, sigma_src, observation_xyt, q_basis, track_gradients=False,
                          *, transport_solver=None, parcel_subdivisions=None, batch_size=None):
    """Evaluate one recurrent transport response per temporal basis column."""
    transport_solver = getattr(model, "recurrent_transport_solver", "grid") if transport_solver is None else transport_solver
    parcel_subdivisions = (getattr(model, "recurrent_parcel_subdivisions", 64)
        if getattr(model, "recurrent_transport_solver", "grid") == "gaussian_parcels" else 16) if parcel_subdivisions is None else parcel_subdivisions
    if transport_solver not in ("grid", "gaussian_parcels"):
        raise ValueError("Unknown response transport solver")
    if transport_solver == "grid" and getattr(model, "recurrent_transport_solver", "grid") != "grid":
        raise ValueError("A parcel Q clock cannot be used with grid transport; configure a separate grid context")
    xyt = torch.as_tensor(observation_xyt, dtype=model.xs.dtype, device=model.xs.device)
    basis = torch.as_tensor(q_basis, dtype=model.xs.dtype, device=model.xs.device)
    if basis.ndim != 2:
        raise ValueError("q_basis must be a two-dimensional matrix.")
    if basis.shape[1] == 0:
        raise ValueError("q_basis must contain at least one column")
    if batch_size is None:
        batch_size = int(os.environ.get("PINN_IA_VP_RESPONSE_BATCH_SIZE", IA_VP_RESPONSE_BATCH_SIZE))
    if int(batch_size) != batch_size or batch_size < 1:
        raise ValueError("response batch_size must be a positive integer")
    expected = int(model.recurrent_integration_times.numel())
    if basis.shape[0] != expected:
        raise ValueError(
            "q_basis rows must match recurrent integration times "
            f"(expected {expected}, got {basis.shape[0]})."
        )
    context = nullcontext() if track_gradients else torch.no_grad()
    with context:
        if transport_solver == "gaussian_parcels":
            from gaussian_parcels import gaussian_parcel_response
            return gaussian_parcel_response(model, sigma_src, xyt,
                model.recurrent_integration_times, basis, subdivisions=parcel_subdivisions)
        plans = _build_recurrent_tensor_plans(model)
        if batch_size > 1:
            gaussian_plans = _build_recurrent_gaussian_plans(model)
            return torch.cat([
                recurrent_plume_response_batch(model, xyt, sigma_src, basis[:, start:start+batch_size],
                    tensor_plans=plans, gaussian_plans=gaussian_plans)
                for start in range(0, basis.shape[1], batch_size)
            ], dim=1)
        shared = {} if plans is None else {"_tensor_plans": plans}
        columns = [
            recurrent_plume_value_from_q(model, xyt, sigma_src, basis[:, j], **shared).reshape(-1)
            for j in range(basis.shape[1])
        ]
    return torch.stack(columns, dim=1)


@timed_component('inner_q_solve')
def solve_nonnegative_q(response_matrix, observations, weights=None, coefficient_upper_bound=np.inf):
    """Solve min ||W(Aq-y)||^2 with 0 <= q <= upper; weights scale residuals."""
    try:
        from scipy.optimize import lsq_linear
    except ImportError as exc:
        raise ImportError("Variable projection requires SciPy.") from exc
    response = np.asarray(
        response_matrix.detach().cpu().numpy()
        if isinstance(response_matrix, torch.Tensor)
        else response_matrix,
        dtype=np.float64,
    )
    observed = np.asarray(
        observations.detach().cpu().numpy()
        if isinstance(observations, torch.Tensor)
        else observations,
        dtype=np.float64,
    ).reshape(-1)
    if response.ndim != 2 or response.shape[0] != observed.size:
        raise ValueError("response_matrix and observations have incompatible shapes.")
    if not np.all(np.isfinite(response)) or not np.all(np.isfinite(observed)):
        raise ValueError("response_matrix and observations must be finite.")
    if weights is None:
        weight = np.ones(observed.size, dtype=np.float64)
    else:
        weight = np.asarray(
            weights.detach().cpu().numpy() if isinstance(weights, torch.Tensor) else weights,
            dtype=np.float64,
        ).reshape(-1)
        if weight.size != observed.size or np.any(weight < 0.0) or not np.all(np.isfinite(weight)):
            raise ValueError("weights must be finite, nonnegative, and match observations.")
    weighted_response = response * weight[:, None]
    upper = float(coefficient_upper_bound)
    if np.isnan(upper) or upper <= 0.:
        raise ValueError("coefficient_upper_bound must be positive or infinity.")
    result = lsq_linear(weighted_response, observed * weight, bounds=(0.0, upper))
    coefficients = np.asarray(result.x, dtype=np.float64)
    prediction = response @ coefficients
    residual = prediction - observed
    singular_values = np.linalg.svd(weighted_response, compute_uv=False)
    condition = (
        float("inf")
        if singular_values.size == 0 or singular_values[-1] <= np.finfo(float).eps
        else float(singular_values[0] / singular_values[-1])
    )
    return NonnegativeQSolution(
        coefficients=coefficients,
        prediction=prediction,
        residual=residual,
        weighted_sse=float(np.sum((weight * residual) ** 2)),
        success=bool(result.success),
        status=int(result.status),
        message=str(result.message),
        response_rank=int(np.linalg.matrix_rank(weighted_response)),
        response_condition=condition,
    )


def generate_candidate_grid(x_bounds, y_bounds, step):
    x_min, x_max = map(float, x_bounds)
    y_min, y_max = map(float, y_bounds)
    step = float(step)
    if x_min > x_max or y_min > y_max or step <= 0.0:
        raise ValueError("Candidate bounds must be ordered and step must be positive.")

    def axis_values(lower, upper):
        count = max(1, int(np.floor((upper - lower) / step)) + 1)
        values = lower + step * np.arange(count, dtype=np.float64)
        if values[-1] < upper - step * 1.0e-9:
            values = np.append(values, upper)
        else:
            values[-1] = upper
        return values

    return tuple(
        (float(x), float(y))
        for y in axis_values(y_min, y_max)
        for x in axis_values(x_min, x_max)
    )


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
