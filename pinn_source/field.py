import math

import torch
import torch.nn.functional as F

from config import (
    ENABLE_ANISOTROPIC_DIFFUSION,
    RECURRENT_ADAPTIVE_SUBSTEPS,
    RECURRENT_GRID_NX,
    RECURRENT_GRID_NY,
    RECURRENT_INITIAL_RELEASE_FRACTION,
    RECURRENT_SOURCE_DECAY_INTEGRATOR,
    RECURRENT_MAX_ADVECTION_CELLS,
    RECURRENT_MAX_SUBSTEPS,
    RECURRENT_SOURCE_SCALE,
    RECURRENT_SOURCE_MODE,
    RECURRENT_DIFFUSION_SOLVER,
    RECURRENT_TRANSPORT_SOLVER,
    RECURRENT_PARCEL_SUBDIVISIONS,
    RECURRENT_SUBSTEPS,
)


FIELD_MODE = "recurrent_pde"


def _match_column(tensor):
    if tensor.dim() == 1:
        return tensor.view(-1, 1)
    return tensor


def _build_adaptive_substep_plan(
    t_values,
    u_values,
    v_values,
    dx,
    dy,
    minimum_substeps=1,
    max_advection_cells=1.0,
    maximum_substeps=16,
    enabled=True,
):
    t_cpu = torch.as_tensor(t_values, dtype=torch.float64).detach().cpu().view(-1)
    u_cpu = torch.as_tensor(u_values, dtype=torch.float64).detach().cpu().view(-1)
    v_cpu = torch.as_tensor(v_values, dtype=torch.float64).detach().cpu().view(-1)
    interval_count = max(int(t_cpu.numel()) - 1, 0)
    if interval_count == 0:
        return (), (), ()
    if u_cpu.numel() < interval_count or v_cpu.numel() < interval_count:
        raise ValueError("Wind history must cover every recurrent time interval.")

    dx = max(abs(float(dx)), 1e-12)
    dy = max(abs(float(dy)), 1e-12)
    minimum_substeps = max(1, int(minimum_substeps))
    maximum_substeps = max(minimum_substeps, int(maximum_substeps))
    max_advection_cells = max(float(max_advection_cells), 1e-6)

    selected = []
    required = []
    displacement_cells = []
    for i in range(interval_count):
        dt = max(float(t_cpu[i + 1] - t_cpu[i]), 0.0)
        cell_distance = math.hypot(
            float(u_cpu[i]) * dt / dx,
            float(v_cpu[i]) * dt / dy,
        )
        requested = max(
            minimum_substeps,
            int(math.ceil(cell_distance / max_advection_cells - 1e-12)),
        )
        chosen = min(requested, maximum_substeps) if enabled else minimum_substeps
        selected.append(chosen)
        required.append(requested)
        displacement_cells.append(cell_distance)
    return tuple(selected), tuple(required), tuple(displacement_cells)


def configure_recurrent_context(
    model,
    x_min,
    x_max,
    y_min,
    y_max,
    t_values,
    u_values,
    v_values,
    d_min_norm=0.0,
    d_scale_norm=1.0,
    anisotropic_diffusion=None,
    d_parallel_min_norm=None,
    d_perp_min_norm=None,
    decay_norm=0.0,
    initial_release_dt=None,
    nx=None,
    ny=None,
    source_mode=None,
    source_sigma_norm=None,
    diffusion_solver=None,
    release_breakpoints=None,
    source_decay_integrator=None,
    transport_solver=None,
    parcel_subdivisions=None,
    parcel_q_breakpoints=None,
):
    """Optional release knots refine output times as well as integration times.

    With no knots the legacy schedule is unchanged. Knots use normalized model
    time; external Q must be evaluated on the resulting integration times.
    """
    nx = max(8, int(RECURRENT_GRID_NX if nx is None else nx))
    ny = max(8, int(RECURRENT_GRID_NY if ny is None else ny))
    dtype = torch.float32
    device = next(model.parameters()).device
    transport_solver = str(RECURRENT_TRANSPORT_SOLVER if transport_solver is None else transport_solver)
    if transport_solver not in {"grid", "gaussian_parcels"}:
        raise ValueError("transport_solver must be grid or gaussian_parcels")
    parcel_subdivisions = RECURRENT_PARCEL_SUBDIVISIONS if parcel_subdivisions is None else parcel_subdivisions
    if isinstance(parcel_subdivisions, bool) or int(parcel_subdivisions) != parcel_subdivisions or parcel_subdivisions < 1:
        raise ValueError("parcel_subdivisions must be a positive integer")
    model.recurrent_transport_solver = transport_solver
    model.recurrent_parcel_subdivisions = int(parcel_subdivisions)
    model.recurrent_parcel_q_breakpoints = tuple(() if parcel_q_breakpoints is None else parcel_q_breakpoints)
    source_decay_integrator = str(RECURRENT_SOURCE_DECAY_INTEGRATOR
        if source_decay_integrator is None else source_decay_integrator).lower()
    if source_decay_integrator not in {"legacy", "exponential_linear"}:
        raise ValueError("source_decay_integrator must be legacy or exponential_linear.")
    model.recurrent_source_decay_integrator = source_decay_integrator

    model.recurrent_d_min_norm = float(d_min_norm)
    model.recurrent_d_scale_norm = float(d_scale_norm)
    model.recurrent_anisotropic_diffusion = bool(
        ENABLE_ANISOTROPIC_DIFFUSION
        if anisotropic_diffusion is None
        else anisotropic_diffusion
    )
    source_mode = str(RECURRENT_SOURCE_MODE if source_mode is None else source_mode).lower()
    diffusion_solver = str(RECURRENT_DIFFUSION_SOLVER if diffusion_solver is None else diffusion_solver).lower()
    if source_mode not in {"gaussian", "cell_gaussian", "point"}:
        raise ValueError("source_mode must be gaussian, cell_gaussian or point.")
    if diffusion_solver not in {"gaussian", "implicit", "tensor_semigroup"}:
        raise ValueError("diffusion_solver must be gaussian, implicit or tensor_semigroup.")
    if diffusion_solver == "implicit" and model.recurrent_anisotropic_diffusion:
        raise ValueError("Implicit diffusion supports scalar D only; use gaussian or tensor_semigroup for anisotropy.")
    if source_sigma_norm is not None and (not math.isfinite(float(source_sigma_norm)) or source_sigma_norm <= 0):
        raise ValueError("source_sigma_norm must be finite and positive.")
    if source_mode == "point" and source_sigma_norm is not None:
        raise ValueError("Point deposition does not use a Gaussian source width.")
    if transport_solver == "gaussian_parcels" and (source_mode == "point" or getattr(model, "source_position_mode", "single") != "single"):
        raise ValueError("Gaussian parcels require a finite-width Gaussian and a single fixed source")
    model.recurrent_source_mode = source_mode
    model.recurrent_source_sigma_norm = None if source_sigma_norm is None else float(source_sigma_norm)
    model.recurrent_diffusion_solver = diffusion_solver
    model.recurrent_d_parallel_min_norm = float(
        d_min_norm if d_parallel_min_norm is None else d_parallel_min_norm
    )
    model.recurrent_d_perp_min_norm = float(
        d_min_norm if d_perp_min_norm is None else d_perp_min_norm
    )
    model.recurrent_decay_norm = float(decay_norm)
    model.recurrent_x_grid = torch.linspace(
        float(x_min), float(x_max), steps=nx, dtype=dtype, device=device
    )
    model.recurrent_y_grid = torch.linspace(
        float(y_min), float(y_max), steps=ny, dtype=dtype, device=device
    )
    yy, xx = torch.meshgrid(
        model.recurrent_y_grid,
        model.recurrent_x_grid,
        indexing="ij",
    )
    model.recurrent_x_mesh = xx
    model.recurrent_y_mesh = yy
    model.recurrent_x_mesh_flat = xx.reshape(-1)
    model.recurrent_y_mesh_flat = yy.reshape(-1)
    model.recurrent_times = torch.as_tensor(
        t_values, dtype=dtype, device=device
    ).view(-1)
    model.recurrent_u = torch.as_tensor(u_values, dtype=dtype, device=device).view(-1)
    model.recurrent_v = torch.as_tensor(v_values, dtype=dtype, device=device).view(-1)
    if initial_release_dt is None:
        if model.recurrent_times.numel() > 1:
            initial_release_dt = float(
                model.recurrent_times[1] - model.recurrent_times[0]
            )
        else:
            initial_release_dt = 1.0
    model.recurrent_initial_release_dt = max(float(initial_release_dt), 1e-6)
    (
        model.recurrent_substeps_per_interval,
        model.recurrent_required_substeps_per_interval,
        model.recurrent_advection_cells_per_interval,
    ) = _build_adaptive_substep_plan(
        model.recurrent_times,
        model.recurrent_u,
        model.recurrent_v,
        dx=float(model.recurrent_x_grid[1] - model.recurrent_x_grid[0]),
        dy=float(model.recurrent_y_grid[1] - model.recurrent_y_grid[0]),
        minimum_substeps=1 if transport_solver == "gaussian_parcels" else RECURRENT_SUBSTEPS,
        max_advection_cells=RECURRENT_MAX_ADVECTION_CELLS,
        maximum_substeps=1 if transport_solver == "gaussian_parcels" else RECURRENT_MAX_SUBSTEPS,
        enabled=RECURRENT_ADAPTIVE_SUBSTEPS and transport_solver == "grid",
    )
    if release_breakpoints is not None:
        from release_pulses import build_release_aware_schedule

        refined, wind_indices = build_release_aware_schedule(
            model.recurrent_times, release_breakpoints, model.recurrent_substeps_per_interval
        )
        if refined.numel() > 1+sum(model.recurrent_substeps_per_interval):
            model.recurrent_times = refined
            model.recurrent_u = model.recurrent_u[wind_indices.clamp(max=model.recurrent_u.numel()-1)]
            model.recurrent_v = model.recurrent_v[wind_indices.clamp(max=model.recurrent_v.numel()-1)]
            # Original substep nodes become explicit output boundaries only
            # when a knot is missing. Avoid extra outputs for aligned schedules.
            (model.recurrent_substeps_per_interval,
             model.recurrent_required_substeps_per_interval,
             model.recurrent_advection_cells_per_interval) = _build_adaptive_substep_plan(
                refined, model.recurrent_u, model.recurrent_v,
                dx=float(model.recurrent_x_grid[1]-model.recurrent_x_grid[0]),
                dy=float(model.recurrent_y_grid[1]-model.recurrent_y_grid[0]),
                minimum_substeps=1, max_advection_cells=RECURRENT_MAX_ADVECTION_CELLS,
                maximum_substeps=1, enabled=False,
            )
    integration_times = [model.recurrent_times[0]]
    interval_q_offsets = []
    advection_plans = []
    source_decay_plans = []
    for i, substeps in enumerate(model.recurrent_substeps_per_interval):
        t_i = model.recurrent_times[i]
        dt_total = torch.clamp(model.recurrent_times[i + 1] - t_i, min=1e-6)
        dt = dt_total / substeps
        source_decay_plans.append(source_decay_weights(float(dt), max(model.recurrent_decay_norm, 0.)))
        interval_q_offsets.append(len(integration_times) - 1)
        integration_times.extend(
            t_i + dt * step for step in range(1, substeps + 1)
        )
        x_back = model.recurrent_x_mesh_flat - model.recurrent_u[i] * dt
        y_back = model.recurrent_y_mesh_flat - model.recurrent_v[i] * dt
        advection_plans.append(
            _build_bilinear_sample_plan(x_back, y_back, model.recurrent_x_grid, model.recurrent_y_grid)
        )
    model.recurrent_integration_times = torch.stack(integration_times)
    model.recurrent_interval_q_offsets = tuple(interval_q_offsets)
    model.recurrent_advection_plans = tuple(advection_plans)
    model.recurrent_source_decay_plans = tuple(source_decay_plans)
    if transport_solver == "gaussian_parcels":
        # Physical Q knots do not depend on raster size or grid CFL schedules.
        clock = model.recurrent_times
        extra = torch.as_tensor(model.recurrent_parcel_q_breakpoints, dtype=clock.dtype, device=device)
        if extra.numel() and (not bool(torch.isfinite(extra).all()) or bool((extra < clock[0]).any()) or bool((extra > clock[-1]).any())):
            raise ValueError("Parcel Q breakpoints must be finite and within the event")
        model.recurrent_integration_times = torch.unique(torch.cat((clock, extra,
            torch.linspace(clock[0], clock[-1], 5, device=device),
            torch.linspace(clock[0], clock[-1], 6, device=device))), sorted=True)


def _has_recurrent_context(model):
    return (
        hasattr(model, "recurrent_x_grid")
        and hasattr(model, "recurrent_y_grid")
        and hasattr(model, "recurrent_times")
        and model.recurrent_times.numel() > 0
    )


def _build_bilinear_sample_plan(x_query, y_query, x_grid, y_grid):
    x_query = _match_column(x_query).view(-1)
    y_query = _match_column(y_query).view(-1)
    nx = int(x_grid.numel())
    ny = int(y_grid.numel())
    x_min = x_grid[0]
    x_max = x_grid[-1]
    y_min = y_grid[0]
    y_max = y_grid[-1]
    inside = (
        (x_query >= x_min)
        & (x_query <= x_max)
        & (y_query >= y_min)
        & (y_query <= y_max)
    )
    gx = (x_query - x_min) / torch.clamp(x_max - x_min, min=1e-8) * (nx - 1)
    gy = (y_query - y_min) / torch.clamp(y_max - y_min, min=1e-8) * (ny - 1)
    gx = torch.clamp(gx, 0.0, float(nx - 1))
    gy = torch.clamp(gy, 0.0, float(ny - 1))

    x0 = torch.floor(gx).long()
    y0 = torch.floor(gy).long()
    x1 = torch.clamp(x0 + 1, max=nx - 1)
    y1 = torch.clamp(y0 + 1, max=ny - 1)
    wx = (gx - x0.to(gx.dtype)).view(-1, 1)
    wy = (gy - y0.to(gy.dtype)).view(-1, 1)

    inside_weight = inside.to(gx.dtype).view(-1, 1)
    return (
        y0 * nx + x0,
        y0 * nx + x1,
        y1 * nx + x0,
        y1 * nx + x1,
        (1.0 - wx) * (1.0 - wy) * inside_weight,
        wx * (1.0 - wy) * inside_weight,
        (1.0 - wx) * wy * inside_weight,
        wx * wy * inside_weight,
    )


def _sample_grid_bilinear_from_plan(field, plan):
    idx00, idx10, idx01, idx11, w00, w10, w01, w11 = plan
    if idx00.device != field.device or w00.dtype != field.dtype:
        idx00, idx10, idx01, idx11 = (index.to(device=field.device) for index in plan[:4])
        w00, w10, w01, w11 = (weight.to(device=field.device, dtype=field.dtype) for weight in plan[4:])
    if field.ndim > 2:
        flat = field.reshape(-1, field.shape[-2] * field.shape[-1])
        return sum(weight.reshape(1, -1) * flat[:, index]
                   for weight, index in zip((w00, w10, w01, w11), (idx00, idx10, idx01, idx11)))
    flat = field.reshape(-1)
    return (
        w00 * flat[idx00].view(-1, 1)
        + w10 * flat[idx10].view(-1, 1)
        + w01 * flat[idx01].view(-1, 1)
        + w11 * flat[idx11].view(-1, 1)
    )


def _sample_grid_bilinear(field, x_query, y_query, x_grid, y_grid):
    plan = _build_bilinear_sample_plan(x_query, y_query, x_grid, y_grid)
    return _sample_grid_bilinear_from_plan(field, plan)


def _advect_field(
    field,
    x_grid,
    y_grid,
    x_mesh_flat,
    y_mesh_flat,
    u,
    v,
    dt,
    sample_plan=None,
):
    if sample_plan is None:
        x_back = x_mesh_flat - u * dt
        y_back = y_mesh_flat - v * dt
        sample_plan = _build_bilinear_sample_plan(
            x_back, y_back, x_grid, y_grid
        )
    return _sample_grid_bilinear_from_plan(field, sample_plan).view_as(field)


def _build_scalar_gaussian_plan(field, x_grid, y_grid, diffusion, dt):
    dx = torch.clamp(x_grid[1] - x_grid[0], min=1e-8)
    dy = torch.clamp(y_grid[1] - y_grid[0], min=1e-8)
    diffusion_eff = torch.clamp(diffusion, min=1e-12)
    sigma_x = torch.sqrt(2.0 * diffusion_eff * dt) / dx
    sigma_y = torch.sqrt(2.0 * diffusion_eff * dt) / dy
    sigma_max = torch.maximum(sigma_x, sigma_y)
    radius = max(1, int(torch.ceil(4.0 * sigma_max.detach()).item()))
    radius = min(radius, max(1, min(field.shape[-2:]) - 1))

    coords = torch.arange(
        -radius,
        radius + 1,
        dtype=field.dtype,
        device=field.device,
    )
    kernel_x = torch.exp(-0.5 * (coords / torch.clamp(sigma_x, min=1e-4)) ** 2)
    kernel_y = torch.exp(-0.5 * (coords / torch.clamp(sigma_y, min=1e-4)) ** 2)
    kernel_x = (kernel_x / torch.clamp(kernel_x.sum(), min=1e-12)).view(1, 1, 1, -1)
    kernel_y = (kernel_y / torch.clamp(kernel_y.sum(), min=1e-12)).view(1, 1, -1, 1)

    return ("separable", kernel_x, kernel_y, radius)


def _apply_gaussian_plan(field, plan):
    value = field.reshape(-1, 1, *field.shape[-2:])
    if plan[0] == "separable":
        _, kernel_x, kernel_y, radius = plan
        value = F.conv2d(F.pad(value, (radius, radius, 0, 0)), kernel_x)
        value = F.conv2d(F.pad(value, (0, 0, radius, radius)), kernel_y)
    else:
        _, kernel, radius_x, radius_y = plan
        value = F.conv2d(F.pad(value, (radius_x, radius_x, radius_y, radius_y)),
                         kernel.view(1, 1, *kernel.shape))
    return torch.clamp(value.reshape(field.shape), min=0.0)


def _diffuse_field(field, x_grid, y_grid, diffusion, dt):
    return _apply_gaussian_plan(field, _build_scalar_gaussian_plan(field, x_grid, y_grid, diffusion, dt))


def _implicit_diffusion_factors(field, x_grid, y_grid, diffusion, dt):
    """Factorized backward Euler with absorbing zero ghost cells, not tensor diffusion."""
    factors = []
    for grid in (x_grid, y_grid):
        spacing = (grid[1] - grid[0]).to(dtype=field.dtype, device=field.device)
        ratio = torch.clamp(diffusion, min=0.) * dt / spacing.square()
        n = grid.numel()
        laplacian = 2. * torch.eye(n, dtype=field.dtype, device=field.device)
        diagonal = torch.ones(n - 1, dtype=field.dtype, device=field.device)
        laplacian = laplacian - torch.diag(diagonal, 1) - torch.diag(diagonal, -1)
        matrix = torch.eye(n, dtype=field.dtype, device=field.device) + ratio * laplacian
        factors.append(torch.linalg.cholesky(matrix))
    return tuple(factors)


def _diffuse_field_implicit(field, x_grid, y_grid, diffusion, dt, factors=None):
    """Positive scalar diffusion retaining small-step variance instead of sampling a tiny Gaussian."""
    if factors is None:
        factors = _implicit_diffusion_factors(field, x_grid, y_grid, diffusion, dt)
    x_factor, y_factor = factors
    after_x = torch.cholesky_solve(field.transpose(-1, -2).contiguous(), x_factor).transpose(-1, -2)
    return torch.cholesky_solve(after_x.contiguous(), y_factor)


def _build_anisotropic_gaussian_plan(
    field,
    x_grid,
    y_grid,
    diffusion_parallel,
    diffusion_perp,
    u,
    v,
    dt,
):
    """Apply a wind-aligned rotated Gaussian diffusion tensor."""
    dx = torch.clamp(x_grid[1] - x_grid[0], min=1e-8)
    dy = torch.clamp(y_grid[1] - y_grid[0], min=1e-8)
    d_parallel = torch.clamp(diffusion_parallel, min=1e-12)
    d_perp = torch.clamp(diffusion_perp, min=1e-12)
    speed = torch.sqrt(u**2 + v**2)
    if float(speed.detach().cpu()) <= 1e-12:
        return _build_scalar_gaussian_plan(
            field,
            x_grid,
            y_grid,
            0.5 * (d_parallel + d_perp),
            dt,
        )

    direction_x = u / speed
    direction_y = v / speed
    variance_x = 2.0 * dt * (
        d_parallel * direction_x**2 + d_perp * direction_y**2
    )
    variance_y = 2.0 * dt * (
        d_parallel * direction_y**2 + d_perp * direction_x**2
    )
    radius_x = max(
        1,
        int(torch.ceil(4.0 * torch.sqrt(variance_x) / dx).detach().cpu()),
    )
    radius_y = max(
        1,
        int(torch.ceil(4.0 * torch.sqrt(variance_y) / dy).detach().cpu()),
    )
    radius_x = min(radius_x, max(1, field.shape[-1] - 1))
    radius_y = min(radius_y, max(1, field.shape[-2] - 1))
    offsets_x = torch.arange(
        -radius_x,
        radius_x + 1,
        dtype=field.dtype,
        device=field.device,
    ) * dx
    offsets_y = torch.arange(
        -radius_y,
        radius_y + 1,
        dtype=field.dtype,
        device=field.device,
    ) * dy
    offset_y, offset_x = torch.meshgrid(offsets_y, offsets_x, indexing="ij")
    distance_parallel = offset_x * direction_x + offset_y * direction_y
    distance_perp = -offset_x * direction_y + offset_y * direction_x
    exponent = -(
        distance_parallel**2 / torch.clamp(4.0 * d_parallel * dt, min=1e-12)
        + distance_perp**2 / torch.clamp(4.0 * d_perp * dt, min=1e-12)
    )
    kernel = torch.exp(exponent)
    kernel = kernel / torch.clamp(kernel.sum(), min=1e-12)
    return ("rotated", kernel, radius_x, radius_y)


def _diffuse_field_anisotropic(field, x_grid, y_grid, diffusion_parallel,
                              diffusion_perp, u, v, dt):
    return _apply_gaussian_plan(field, _build_anisotropic_gaussian_plan(
        field, x_grid, y_grid, diffusion_parallel, diffusion_perp, u, v, dt))


def _diffuse_field_tensor(field, x_grid, y_grid, diffusion_parallel, diffusion_perp,
                          u, v, dt, plan=None):
    from tensor_diffusion import build_diffusion_plan, apply_diffusion_plan
    if plan is None:
        plan = build_diffusion_plan(x_grid, y_grid, diffusion_parallel, diffusion_perp, u, v, dt)
    return apply_diffusion_plan(field, plan)


def _source_grid(model, t_value, x_grid, y_grid, x_mesh, y_mesh, sigma_src):
    xs, ys = model.source_xy(t_value.view(1, 1))
    dx = torch.clamp(x_grid[1] - x_grid[0], min=1e-8)
    dy = torch.clamp(y_grid[1] - y_grid[0], min=1e-8)
    source_mode = getattr(model, "recurrent_source_mode", "gaussian")
    if source_mode == "point":
        if not bool(((xs >= x_grid[0]) & (xs <= x_grid[-1]) &
                     (ys >= y_grid[0]) & (ys <= y_grid[-1])).all()):
            raise ValueError("Point source lies outside the computational grid; enlarge its domain first.")
        plan = _build_bilinear_sample_plan(xs, ys, x_grid, y_grid)
        values = torch.zeros_like(x_mesh).reshape(-1)
        for index, weight in zip(plan[:4], plan[4:]):
            values = values.scatter_add(0, index, weight.reshape(-1))
        return values.view_as(x_mesh) / (dx * dy)
    sigma_override = getattr(model, "recurrent_source_sigma_norm", None)
    # Preserve the legacy floor, but respect an explicitly configured physical width.
    sigma = max(float(sigma_src), 1e-4) if sigma_override is None else float(sigma_override)
    if source_mode == "cell_gaussian":
        # Integrate the full-space Gaussian over grid-centered cells. Outside
        # emission remains outside: unlike the legacy source, do not renormalize.
        def cell_mass(grid, center, spacing):
            scale = math.sqrt(2.) * sigma
            lo = (grid - .5 * spacing - center.reshape(())) / scale
            hi = (grid + .5 * spacing - center.reshape(())) / scale
            return .5 * (torch.erf(hi) - torch.erf(lo))
        x_mass = cell_mass(x_grid, xs, dx)
        y_mass = cell_mass(y_grid, ys, dy)
        return y_mass[:, None] * x_mass[None, :] / (dx * dy)
    src = torch.exp(-((x_mesh - xs) ** 2 + (y_mesh - ys) ** 2) / (2.0 * sigma**2))
    mass = torch.clamp(torch.sum(src) * dx * dy, min=1e-8)
    return src / mass


def source_decay_weights(dt, decay):
    """Integrals of endpoint linear hats times exp(-decay*(dt-t)); no transport."""
    if not math.isfinite(dt) or not math.isfinite(decay) or dt <= 0 or decay < 0:
        raise ValueError("Positive finite dt and nonnegative finite decay required")
    z = decay*dt
    attenuation = math.exp(-z)
    if z < 1e-3:
        left = .5-z/3+z*z/8-z**3/30+z**4/144
        right = .5-z/6+z*z/24-z**3/120+z**4/720
    else:
        integral = -math.expm1(-z)/z
        left = (integral-attenuation)/z
        right = (1-integral)/z
    return attenuation, dt*left, dt*right


def _advance_recurrent_step(
    field,
    source_start,
    q_start,
    source_end,
    q_end,
    x_grid,
    y_grid,
    x_mesh_flat,
    y_mesh_flat,
    u_value,
    v_value,
    diffusion,
    decay,
    source_scale,
    dt,
    advection_plan=None,
    diffusion_parallel=None,
    diffusion_perp=None,
    diffusion_solver="gaussian",
    implicit_factors=None,
    tensor_diffusion_plan=None,
    source_decay_integrator="legacy",
    source_decay_plan=None,
    gaussian_plan=None,
):
    half_dt = 0.5 * dt
    if source_decay_integrator == "exponential_linear":
        attenuation, left_weight, right_weight = (source_decay_plan if source_decay_plan is not None
            else source_decay_weights(float(dt), max(float(decay), 0.)))
        # Scalar decay commutes with linear transport. Endpoint source weights
        # are exact for linear Q with decay alone, not for spatial transport.
        field = field*attenuation + source_scale*q_start*source_start*left_weight
    elif source_decay_integrator == "legacy":
        field = field + source_scale * q_start * source_start * half_dt
        right_weight = half_dt
    else:
        raise ValueError("Unknown source/decay integrator")
    field = _advect_field(
        field, x_grid, y_grid, x_mesh_flat, y_mesh_flat,
        u_value, v_value, dt, sample_plan=advection_plan,
    )
    if diffusion_solver == "tensor_semigroup":
        field = _diffuse_field_tensor(field, x_grid, y_grid,
            diffusion if diffusion_parallel is None else diffusion_parallel,
            diffusion if diffusion_perp is None else diffusion_perp,
            u_value, v_value, dt, tensor_diffusion_plan)
    elif diffusion_solver == "implicit":
        if diffusion_parallel is not None or diffusion_perp is not None:
            raise ValueError("Scalar implicit diffusion cannot represent rotated anisotropy.")
        field = _diffuse_field_implicit(field, x_grid, y_grid, diffusion, dt, implicit_factors)
    elif gaussian_plan is not None:
        field = _apply_gaussian_plan(field, gaussian_plan)
    elif diffusion_parallel is not None and diffusion_perp is not None:
        field = _diffuse_field_anisotropic(
            field,
            x_grid,
            y_grid,
            diffusion_parallel,
            diffusion_perp,
            u_value,
            v_value,
            dt,
        )
    else:
        field = _diffuse_field(field, x_grid, y_grid, diffusion, dt)
    if decay > 0.0 and source_decay_integrator == "legacy":
        field = field * torch.exp(-dt * decay)
    field = field + source_scale * q_end * source_end * right_weight
    return field


def _coerce_external_q(model, q_integration):
    q_values = torch.as_tensor(
        q_integration,
        dtype=model.xs.dtype,
        device=model.xs.device,
    ).view(-1)
    expected = int(model.recurrent_integration_times.numel())
    if q_values.numel() != expected:
        raise ValueError(
            "q_integration must contain one value per recurrent integration "
            f"time (expected {expected}, got {q_values.numel()})."
        )
    return q_values


def _recurrent_diffusion_values(model):
    if getattr(model, "recurrent_anisotropic_diffusion", False):
        if not hasattr(model, "anisotropic_D"):
            raise AttributeError("Anisotropic diffusion requires model.anisotropic_D().")
        parallel, perp = model.anisotropic_D()
        scale = float(getattr(model, "recurrent_d_scale_norm", 1.0))
        parallel = float(getattr(model, "recurrent_d_parallel_min_norm", 0.0)) + parallel*scale
        perp = float(getattr(model, "recurrent_d_perp_min_norm", 0.0)) + perp*scale
        return None, torch.maximum(parallel, perp), perp
    diffusion = (float(getattr(model, "recurrent_d_min_norm", 0.0))
                 + model.D()*float(getattr(model, "recurrent_d_scale_norm", 1.0)))
    return diffusion, None, None


def _build_recurrent_tensor_plans(model):
    """Fresh plans for a single response matrix; never stored on the model."""
    if getattr(model, "recurrent_diffusion_solver", "gaussian") != "tensor_semigroup":
        return None
    from tensor_diffusion import build_diffusion_plan
    diffusion, parallel, perp = _recurrent_diffusion_values(model)
    if parallel is None:
        parallel = perp = diffusion
    convert = lambda value: value.to(device=model.xs.device, dtype=model.xs.dtype)
    times = convert(model.recurrent_times)
    x_grid, y_grid = convert(model.recurrent_x_grid), convert(model.recurrent_y_grid)
    u_values, v_values = convert(model.recurrent_u), convert(model.recurrent_v)
    substeps = getattr(model, "recurrent_substeps_per_interval", ())
    return tuple(build_diffusion_plan(x_grid, y_grid,
        parallel, perp, u_values[i], v_values[i],
        torch.clamp(times[i+1]-times[i], min=1e-6)/(
            int(substeps[i]) if i < len(substeps) else max(1, int(RECURRENT_SUBSTEPS))))
        for i in range(times.numel()-1))


def _build_recurrent_gaussian_plans(model):
    """Fresh differentiable kernels shared only within one response matrix."""
    if getattr(model, "recurrent_diffusion_solver", "gaussian") != "gaussian":
        return None
    diffusion, parallel, perp = _recurrent_diffusion_values(model)
    convert = lambda value: value.to(device=model.xs.device, dtype=model.xs.dtype)
    times = convert(model.recurrent_times)
    x_grid, y_grid = convert(model.recurrent_x_grid), convert(model.recurrent_y_grid)
    u_values, v_values = convert(model.recurrent_u), convert(model.recurrent_v)
    reference = model.xs.new_empty((y_grid.numel(), x_grid.numel()))
    counts = getattr(model, "recurrent_substeps_per_interval", ())
    plans = []
    for i in range(times.numel()-1):
        dt = torch.clamp(times[i+1]-times[i], min=1e-6)/(
            int(counts[i]) if i < len(counts) else max(1, int(RECURRENT_SUBSTEPS)))
        plans.append(_build_scalar_gaussian_plan(reference, x_grid, y_grid, diffusion, dt)
                     if parallel is None else _build_anisotropic_gaussian_plan(
                         reference, x_grid, y_grid, parallel, perp, u_values[i], v_values[i], dt))
    return tuple(plans)


def _recurrent_plume_fields_core(model, sigma_src, q_integration, *, initial_field=None,
                                 initial_release_fraction=None, _tensor_plans=None,
                                 _gaussian_plans=None, _batch_q=False):
    if not _has_recurrent_context(model):
        raise RuntimeError("Recurrent plume context has not been configured.")

    if _batch_q:
        q_integration = torch.as_tensor(q_integration, dtype=model.xs.dtype, device=model.xs.device)
        if q_integration.ndim != 2 or q_integration.shape[0] != model.recurrent_integration_times.numel():
            raise ValueError("Batched Q must have shape (integration times, basis count)")
        q_integration = q_integration[:, :, None, None]
    else:
        q_integration = _coerce_external_q(model, q_integration)

    x_grid = model.recurrent_x_grid.to(device=model.xs.device, dtype=model.xs.dtype)
    y_grid = model.recurrent_y_grid.to(device=model.xs.device, dtype=model.xs.dtype)
    x_mesh = model.recurrent_x_mesh.to(device=model.xs.device, dtype=model.xs.dtype)
    y_mesh = model.recurrent_y_mesh.to(device=model.xs.device, dtype=model.xs.dtype)
    x_mesh_flat = model.recurrent_x_mesh_flat.to(
        device=model.xs.device, dtype=model.xs.dtype
    )
    y_mesh_flat = model.recurrent_y_mesh_flat.to(
        device=model.xs.device, dtype=model.xs.dtype
    )
    t_values = model.recurrent_times.to(device=model.xs.device, dtype=model.xs.dtype)
    u_values = model.recurrent_u.to(device=model.xs.device, dtype=model.xs.dtype)
    v_values = model.recurrent_v.to(device=model.xs.device, dtype=model.xs.dtype)

    field = torch.zeros(
        ((q_integration.shape[1],) if _batch_q else ()) + (y_grid.numel(), x_grid.numel()),
        dtype=model.xs.dtype, device=model.xs.device
    )
    if initial_field is not None:
        initial_field = torch.as_tensor(initial_field, dtype=field.dtype, device=field.device)
        if (initial_field.shape != field.shape or not bool(torch.isfinite(initial_field).all())
                or bool((initial_field < 0).any())):
            raise ValueError("initial_field must be finite, nonnegative and match the transport grid.")
        field = initial_field
    fields = []
    use_anisotropic_diffusion = bool(
        getattr(model, "recurrent_anisotropic_diffusion", False)
    )
    diffusion_solver = getattr(model, "recurrent_diffusion_solver", "gaussian")
    if diffusion_solver == "implicit" and use_anisotropic_diffusion:
        raise ValueError("Implicit diffusion currently supports scalar D only.")
    diffusion, diffusion_parallel, diffusion_perp = _recurrent_diffusion_values(model)
    if _tensor_plans is not None and (diffusion_solver != "tensor_semigroup"
                                    or len(_tensor_plans) != t_values.numel()-1):
        raise ValueError("Internal tensor plans must match this recurrent context.")
    decay = max(float(getattr(model, "recurrent_decay_norm", 0.0)), 0.0)
    source_scale = float(RECURRENT_SOURCE_SCALE)

    source = _source_grid(
        model, t_values[0], x_grid, y_grid, x_mesh, y_mesh, sigma_src
    )
    if t_values.numel() == 1:
        field = field + source_scale * q_integration[0] * source
        fields.append(field)
        return torch.stack(fields, dim=0)

    initial_release_fraction = max(float(RECURRENT_INITIAL_RELEASE_FRACTION
        if initial_release_fraction is None else initial_release_fraction), 0.0)
    if initial_release_fraction > 0.0:
        first_dt_total = torch.as_tensor(
            getattr(
                model,
                "recurrent_initial_release_dt",
                float(t_values[1] - t_values[0]),
            ),
            dtype=model.xs.dtype,
            device=model.xs.device,
        )
        initial_q = q_integration[0]
        field = (
            field
            + source_scale
            * initial_q
            * source
            * first_dt_total
            * initial_release_fraction
        )

    fields.append(field)
    for i in range(t_values.numel() - 1):
        t_i = t_values[i]
        dt_total = torch.clamp(t_values[i + 1] - t_i, min=1e-6)
        substep_plan = getattr(model, "recurrent_substeps_per_interval", ())
        substeps = (
            int(substep_plan[i])
            if i < len(substep_plan)
            else max(1, int(RECURRENT_SUBSTEPS))
        )
        dt = dt_total / substeps
        q_offset = model.recurrent_interval_q_offsets[i]
        advection_plan = model.recurrent_advection_plans[i]
        implicit_factors = (_implicit_diffusion_factors(field, x_grid, y_grid, diffusion, dt)
                            if diffusion_solver == "implicit" else None)
        tensor_plan = None
        if _tensor_plans is not None:
            tensor_plan = _tensor_plans[i]
        elif diffusion_solver == "tensor_semigroup":
            from tensor_diffusion import build_diffusion_plan
            tensor_plan = build_diffusion_plan(x_grid, y_grid,
                diffusion_parallel if use_anisotropic_diffusion else diffusion,
                diffusion_perp if use_anisotropic_diffusion else diffusion,
                u_values[i], v_values[i], dt)
        for substep in range(substeps):
            q_left = q_integration[q_offset + substep]
            q_right = q_integration[q_offset + substep + 1]
            field = _advance_recurrent_step(
                field,
                source,
                q_left,
                source,
                q_right,
                x_grid,
                y_grid,
                x_mesh_flat,
                y_mesh_flat,
                u_values[i],
                v_values[i],
                diffusion,
                decay,
                source_scale,
                dt,
                advection_plan=advection_plan,
                diffusion_parallel=diffusion_parallel,
                diffusion_perp=diffusion_perp,
                diffusion_solver=diffusion_solver,
                source_decay_integrator=getattr(model, "recurrent_source_decay_integrator", "legacy"),
                source_decay_plan=model.recurrent_source_decay_plans[i],
                implicit_factors=implicit_factors,
                tensor_diffusion_plan=tensor_plan,
                gaussian_plan=None if _gaussian_plans is None else _gaussian_plans[i],
            )
        fields.append(field)

    return torch.stack(fields, dim=0)


def recurrent_plume_response_batch(model, xyt, sigma_src, q_basis, *,
                                    tensor_plans=None, gaussian_plans=None):
    """Propagate independent Q columns together; retain the scalar transport core."""
    fields = _recurrent_plume_fields_core(model, sigma_src, q_basis, _batch_q=True,
        _tensor_plans=tensor_plans, _gaussian_plans=gaussian_plans)
    query = torch.as_tensor(xyt, device=fields.device, dtype=fields.dtype)
    times = model.recurrent_times.to(device=fields.device, dtype=fields.dtype)
    plan = _build_bilinear_sample_plan(query[:, 0], query[:, 1],
        model.recurrent_x_grid.to(fields), model.recurrent_y_grid.to(fields))
    cells = fields.shape[-2] * fields.shape[-1]
    flat = fields.permute(1, 0, 2, 3).reshape(fields.shape[1], -1)
    if times.numel() == 1:
        low = high = torch.zeros(query.shape[0], device=fields.device, dtype=torch.long)
        alpha = query.new_zeros(query.shape[0])
    else:
        exact_index = torch.bucketize(query[:, 2].contiguous(), times)
        valid = exact_index < times.numel()
        exact = valid & ((query[:, 2]-times[exact_index.clamp(max=times.numel()-1)]).abs() < 1e-6)
        if bool(exact.all()):
            low = high = exact_index
            alpha = query.new_zeros(query.shape[0])
        else:
            high = exact_index.clamp(1, times.numel()-1)
            low = high-1
            alpha = ((query[:, 2]-times[low])/(times[high]-times[low]).clamp(min=1e-6)).clamp(0., 1.)
    def sample(layer):
        return sum(weight.reshape(1, -1)*flat[:, layer*cells+index]
                   for index, weight in zip(plan[:4], plan[4:]))
    return ((1.-alpha)*sample(low)+alpha*sample(high)).T


def _parcel_values(model, xyt, sigma_src, q_integration, initial_release_fraction=None):
    from gaussian_parcels import gaussian_parcel_value_from_q
    q = _coerce_external_q(model, q_integration)
    if not bool(torch.isfinite(q).all()) or bool((q < 0).any()):
        raise ValueError("Parcel source strengths must be finite and nonnegative")
    query = torch.as_tensor(xyt, dtype=model.xs.dtype, device=model.xs.device)
    return torch.cat([gaussian_parcel_value_from_q(model, chunk, sigma_src,
        model.recurrent_integration_times, q,
        subdivisions=model.recurrent_parcel_subdivisions,
        initial_release_fraction=initial_release_fraction) for chunk in query.split(4096)], dim=0)


def _parcel_fields(model, sigma_src, q, times, x_values=None, y_values=None,
                   initial_release_fraction=None):
    cast = lambda v: torch.as_tensor(v, dtype=model.xs.dtype, device=model.xs.device).reshape(-1)
    xs = cast(model.recurrent_x_grid if x_values is None else x_values)
    ys = cast(model.recurrent_y_grid if y_values is None else y_values)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    xy = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)
    return torch.stack([_parcel_values(model, torch.cat((xy, t.expand(len(xy), 1)), dim=1),
        sigma_src, q, initial_release_fraction).reshape(len(ys), len(xs)) for t in cast(times)])


def recurrent_plume_fields_from_q(model, sigma_src, q_integration, *, initial_field=None,
                                  initial_release_fraction=None, _tensor_plans=None):
    """Run recurrent transport with externally supplied source strengths.

    ``q_integration`` is ordered on ``model.recurrent_integration_times`` and is
    used directly, so it is not registered as a trainable model parameter.
    Optional ``initial_field`` is an already transported nonnegative grid;
    it is added before the existing first-node release. An explicit zero
    ``initial_release_fraction`` disables that release for history-only runs.
    Omitted keywords preserve the production initialization exactly.
    """
    if getattr(model, "recurrent_transport_solver", "grid") == "gaussian_parcels":
        if initial_field is not None:
            raise ValueError("Parcel transport cannot interpret an initial grid field")
        return _parcel_fields(model, sigma_src, q_integration, model.recurrent_times,
            initial_release_fraction=initial_release_fraction)
    return _recurrent_plume_fields_core(model, sigma_src, q_integration,
        initial_field=initial_field, initial_release_fraction=initial_release_fraction,
        _tensor_plans=_tensor_plans)


def recurrent_plume_fields(model, sigma_src):
    if not _has_recurrent_context(model):
        raise RuntimeError("Recurrent plume context has not been configured.")
    integration_times = model.recurrent_integration_times.to(
        device=model.xs.device, dtype=model.xs.dtype
    )
    q_integration = model.Q(integration_times.view(-1, 1)).view(-1)
    return recurrent_plume_fields_from_q(model, sigma_src, q_integration)


_RECURRENT_CONTEXT_ATTRIBUTES = (
    "recurrent_transport_solver",
    "recurrent_parcel_subdivisions",
    "recurrent_parcel_q_breakpoints",
    "recurrent_source_decay_integrator",
    "recurrent_source_decay_plans",
    "recurrent_source_mode",
    "recurrent_source_sigma_norm",
    "recurrent_diffusion_solver",
    "recurrent_d_min_norm",
    "recurrent_d_scale_norm",
    "recurrent_anisotropic_diffusion",
    "recurrent_d_parallel_min_norm",
    "recurrent_d_perp_min_norm",
    "recurrent_decay_norm",
    "recurrent_x_grid",
    "recurrent_y_grid",
    "recurrent_x_mesh",
    "recurrent_y_mesh",
    "recurrent_x_mesh_flat",
    "recurrent_y_mesh_flat",
    "recurrent_times",
    "recurrent_u",
    "recurrent_v",
    "recurrent_initial_release_dt",
    "recurrent_substeps_per_interval",
    "recurrent_required_substeps_per_interval",
    "recurrent_advection_cells_per_interval",
    "recurrent_integration_times",
    "recurrent_interval_q_offsets",
    "recurrent_advection_plans",
)


def recurrent_plume_fields_at_times(model, sigma_src, t_values, u_values, v_values, *, x_values=None, y_values=None):
    if not _has_recurrent_context(model):
        raise RuntimeError("Recurrent plume context has not been configured.")
    if getattr(model, "recurrent_transport_solver", "grid") == "gaussian_parcels":
        clock = model.recurrent_integration_times.to(device=model.xs.device, dtype=model.xs.dtype)
        return _parcel_fields(model, sigma_src, model.Q(clock.view(-1, 1)).view(-1),
            t_values, x_values, y_values)
    if x_values is not None or y_values is not None:
        raise ValueError("Custom output coordinates are only supported by parcel transport")

    saved_context = {
        name: getattr(model, name)
        for name in _RECURRENT_CONTEXT_ATTRIBUTES
        if hasattr(model, name)
    }
    try:
        configure_recurrent_context(
            model=model,
            x_min=float(saved_context["recurrent_x_grid"][0]),
            x_max=float(saved_context["recurrent_x_grid"][-1]),
            y_min=float(saved_context["recurrent_y_grid"][0]),
            y_max=float(saved_context["recurrent_y_grid"][-1]),
            t_values=t_values,
            u_values=u_values,
            v_values=v_values,
            d_min_norm=saved_context["recurrent_d_min_norm"],
            d_scale_norm=saved_context["recurrent_d_scale_norm"],
            anisotropic_diffusion=saved_context[
                "recurrent_anisotropic_diffusion"
            ],
            d_parallel_min_norm=saved_context[
                "recurrent_d_parallel_min_norm"
            ],
            d_perp_min_norm=saved_context["recurrent_d_perp_min_norm"],
            decay_norm=saved_context["recurrent_decay_norm"],
            initial_release_dt=saved_context["recurrent_initial_release_dt"],
            nx=int(saved_context["recurrent_x_grid"].numel()),
            ny=int(saved_context["recurrent_y_grid"].numel()),
            source_mode=saved_context.get("recurrent_source_mode", "gaussian"),
            source_sigma_norm=saved_context.get("recurrent_source_sigma_norm"),
            diffusion_solver=saved_context.get("recurrent_diffusion_solver", "gaussian"),
            source_decay_integrator=saved_context.get("recurrent_source_decay_integrator", "legacy"),
        )
        return recurrent_plume_fields(model, sigma_src)
    finally:
        for name in _RECURRENT_CONTEXT_ATTRIBUTES:
            if name in saved_context:
                setattr(model, name, saved_context[name])
            elif hasattr(model, name):
                delattr(model, name)


def _sample_recurrent_fields(model, fields, xyt):
    x_grid = model.recurrent_x_grid.to(device=xyt.device, dtype=xyt.dtype)
    y_grid = model.recurrent_y_grid.to(device=xyt.device, dtype=xyt.dtype)
    t_grid = model.recurrent_times.to(device=xyt.device, dtype=xyt.dtype)
    x_query = xyt[:, 0:1]
    y_query = xyt[:, 1:2]
    t_query = xyt[:, 2:3].view(-1)

    if t_grid.numel() == 1:
        return _sample_grid_bilinear(fields[0], x_query, y_query, x_grid, y_grid)

    idx_exact = torch.bucketize(t_query.contiguous(), t_grid)
    exact_valid = idx_exact < t_grid.numel()
    exact_match = torch.zeros_like(exact_valid, dtype=torch.bool)
    if torch.any(exact_valid):
        exact_match[exact_valid] = (
            torch.abs(t_query[exact_valid] - t_grid[idx_exact[exact_valid]]) < 1e-6
        )
    if bool(torch.all(exact_match).detach().cpu().item()):
        val = torch.empty((t_query.numel(), 1), dtype=xyt.dtype, device=xyt.device)
        for layer in torch.unique(idx_exact):
            mask = idx_exact == layer
            val[mask] = _sample_grid_bilinear(
                fields[layer], x_query[mask], y_query[mask], x_grid, y_grid
            )
        return val

    idx_hi = torch.bucketize(t_query.contiguous(), t_grid)
    idx_hi = torch.clamp(idx_hi, min=1, max=t_grid.numel() - 1)
    idx_lo = idx_hi - 1
    t_lo = t_grid[idx_lo]
    t_hi = t_grid[idx_hi]
    alpha = ((t_query - t_lo) / torch.clamp(t_hi - t_lo, min=1e-6)).view(-1, 1)
    alpha = torch.clamp(alpha, 0.0, 1.0)

    val_lo = torch.empty((t_query.numel(), 1), dtype=xyt.dtype, device=xyt.device)
    val_hi = torch.empty((t_query.numel(), 1), dtype=xyt.dtype, device=xyt.device)
    for layer in torch.unique(idx_lo):
        mask = idx_lo == layer
        val_lo[mask] = _sample_grid_bilinear(
            fields[layer], x_query[mask], y_query[mask], x_grid, y_grid
        )
    for layer in torch.unique(idx_hi):
        mask = idx_hi == layer
        val_hi[mask] = _sample_grid_bilinear(
            fields[layer], x_query[mask], y_query[mask], x_grid, y_grid
        )
    return (1.0 - alpha) * val_lo + alpha * val_hi


def recurrent_plume_value_from_q(model, xyt, sigma_src, q_integration, *, initial_field=None,
                                 initial_release_fraction=None, _tensor_plans=None):
    if getattr(model, "recurrent_transport_solver", "grid") == "gaussian_parcels":
        if initial_field is not None:
            raise ValueError("Parcel transport cannot interpret an initial grid field")
        return _parcel_values(model, xyt, sigma_src, q_integration, initial_release_fraction)
    fields = recurrent_plume_fields_from_q(model, sigma_src, q_integration,
        initial_field=initial_field, initial_release_fraction=initial_release_fraction,
        _tensor_plans=_tensor_plans)
    return _sample_recurrent_fields(model, fields, xyt)


def recurrent_plume_value(model, xyt, sigma_src):
    if getattr(model, "recurrent_transport_solver", "grid") == "gaussian_parcels":
        clock = model.recurrent_integration_times.to(device=model.xs.device, dtype=model.xs.dtype)
        return _parcel_values(model, xyt, sigma_src, model.Q(clock.view(-1, 1)).view(-1))
    fields = recurrent_plume_fields(model, sigma_src)
    return _sample_recurrent_fields(model, fields, xyt)


def field_components(model, xyt, u, v, sigma_src):
    q_val = model.Q(xyt[:, 2:3])
    plume = recurrent_plume_value(model, xyt, sigma_src)
    bg = torch.zeros_like(q_val)
    return bg, plume, q_val, plume, plume


def predict_concentration(model, xyt, u, v, sigma_src):
    return recurrent_plume_value(model, xyt, sigma_src)


def concentration_from_components(bg, plume, q_val, source_term):
    return source_term
