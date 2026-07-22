import math

import torch
import torch.nn.functional as F

from config import (
    RECURRENT_ADAPTIVE_SUBSTEPS,
    RECURRENT_GRID_NX,
    RECURRENT_GRID_NY,
    RECURRENT_INITIAL_RELEASE_FRACTION,
    RECURRENT_MAX_ADVECTION_CELLS,
    RECURRENT_MAX_SUBSTEPS,
    RECURRENT_SOURCE_SCALE,
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
    decay_norm=0.0,
    initial_release_dt=None,
    nx=None,
    ny=None,
):
    nx = max(8, int(RECURRENT_GRID_NX if nx is None else nx))
    ny = max(8, int(RECURRENT_GRID_NY if ny is None else ny))
    dtype = torch.float32
    device = next(model.parameters()).device

    model.recurrent_d_min_norm = float(d_min_norm)
    model.recurrent_d_scale_norm = float(d_scale_norm)
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
        minimum_substeps=RECURRENT_SUBSTEPS,
        max_advection_cells=RECURRENT_MAX_ADVECTION_CELLS,
        maximum_substeps=RECURRENT_MAX_SUBSTEPS,
        enabled=RECURRENT_ADAPTIVE_SUBSTEPS,
    )
    integration_times = [model.recurrent_times[0]]
    interval_q_offsets = []
    advection_plans = []
    for i, substeps in enumerate(model.recurrent_substeps_per_interval):
        t_i = model.recurrent_times[i]
        dt_total = torch.clamp(model.recurrent_times[i + 1] - t_i, min=1e-6)
        dt = dt_total / substeps
        interval_q_offsets.append(len(integration_times) - 1)
        integration_times.extend(
            t_i + dt * step for step in range(1, substeps + 1)
        )
        x_back = model.recurrent_x_mesh_flat - model.recurrent_u[i] * dt
        y_back = model.recurrent_y_mesh_flat - model.recurrent_v[i] * dt
        advection_plans.append(
            _build_bilinear_sample_plan(
                x_back,
                y_back,
                model.recurrent_x_grid,
                model.recurrent_y_grid,
            )
        )
    model.recurrent_integration_times = torch.stack(integration_times)
    model.recurrent_interval_q_offsets = tuple(interval_q_offsets)
    model.recurrent_advection_plans = tuple(advection_plans)


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


def _diffuse_field(field, x_grid, y_grid, diffusion, dt):
    dx = torch.clamp(x_grid[1] - x_grid[0], min=1e-8)
    dy = torch.clamp(y_grid[1] - y_grid[0], min=1e-8)
    diffusion_eff = torch.clamp(diffusion, min=1e-12)
    sigma_x = torch.sqrt(2.0 * diffusion_eff * dt) / dx
    sigma_y = torch.sqrt(2.0 * diffusion_eff * dt) / dy
    sigma_max = torch.maximum(sigma_x, sigma_y)
    radius = max(1, int(torch.ceil(4.0 * sigma_max.detach()).item()))
    radius = min(radius, max(1, min(field.shape) - 1))

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

    blurred = field.view(1, 1, *field.shape)
    blurred = F.pad(blurred, (radius, radius, 0, 0), mode="constant", value=0.0)
    blurred = F.conv2d(blurred, kernel_x)
    blurred = F.pad(blurred, (0, 0, radius, radius), mode="constant", value=0.0)
    blurred = F.conv2d(blurred, kernel_y)
    return torch.clamp(blurred[0, 0], min=0.0)


def _source_grid(model, t_value, x_grid, y_grid, x_mesh, y_mesh, sigma_src):
    xs, ys = model.source_xy(t_value.view(1, 1))
    sigma = max(float(sigma_src), 1e-4)
    src = torch.exp(-((x_mesh - xs) ** 2 + (y_mesh - ys) ** 2) / (2.0 * sigma**2))
    dx = torch.clamp(x_grid[1] - x_grid[0], min=1e-8)
    dy = torch.clamp(y_grid[1] - y_grid[0], min=1e-8)
    mass = torch.clamp(torch.sum(src) * dx * dy, min=1e-8)
    return src / mass


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
):
    half_dt = 0.5 * dt
    field = field + source_scale * q_start * source_start * half_dt
    field = _advect_field(
        field,
        x_grid,
        y_grid,
        x_mesh_flat,
        y_mesh_flat,
        u_value,
        v_value,
        dt,
        sample_plan=advection_plan,
    )
    field = _diffuse_field(field, x_grid, y_grid, diffusion, dt)
    if decay > 0.0:
        field = field * torch.exp(-dt * decay)
    field = field + source_scale * q_end * source_end * half_dt
    return field


def recurrent_plume_fields(model, sigma_src):
    if not _has_recurrent_context(model):
        raise RuntimeError("Recurrent plume context has not been configured.")

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
        (y_grid.numel(), x_grid.numel()), dtype=model.xs.dtype, device=model.xs.device
    )
    fields = []
    diffusion = (
        float(getattr(model, "recurrent_d_min_norm", 0.0))
        + model.D() * float(getattr(model, "recurrent_d_scale_norm", 1.0))
    )
    decay = max(float(getattr(model, "recurrent_decay_norm", 0.0)), 0.0)
    source_scale = float(RECURRENT_SOURCE_SCALE)

    source = _source_grid(
        model, t_values[0], x_grid, y_grid, x_mesh, y_mesh, sigma_src
    )
    if t_values.numel() == 1:
        field = field + source_scale * model.Q(t_values[0].view(1, 1)).view(()) * source
        fields.append(field)
        return torch.stack(fields, dim=0)

    integration_times = model.recurrent_integration_times.to(
        device=model.xs.device, dtype=model.xs.dtype
    )
    q_integration = model.Q(integration_times.view(-1, 1)).view(-1)

    initial_release_fraction = max(float(RECURRENT_INITIAL_RELEASE_FRACTION), 0.0)
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
            )
        fields.append(field)

    return torch.stack(fields, dim=0)


_RECURRENT_CONTEXT_ATTRIBUTES = (
    "recurrent_d_min_norm",
    "recurrent_d_scale_norm",
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


def recurrent_plume_fields_at_times(model, sigma_src, t_values, u_values, v_values):
    if not _has_recurrent_context(model):
        raise RuntimeError("Recurrent plume context has not been configured.")

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
            decay_norm=saved_context["recurrent_decay_norm"],
            initial_release_dt=saved_context["recurrent_initial_release_dt"],
            nx=int(saved_context["recurrent_x_grid"].numel()),
            ny=int(saved_context["recurrent_y_grid"].numel()),
        )
        return recurrent_plume_fields(model, sigma_src)
    finally:
        for name in _RECURRENT_CONTEXT_ATTRIBUTES:
            if name in saved_context:
                setattr(model, name, saved_context[name])
            elif hasattr(model, name):
                delattr(model, name)


def recurrent_plume_value(model, xyt, sigma_src):
    fields = recurrent_plume_fields(model, sigma_src)
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


def field_components(model, xyt, u, v, sigma_src):
    q_val = model.Q(xyt[:, 2:3])
    plume = recurrent_plume_value(model, xyt, sigma_src)
    bg = torch.zeros_like(q_val)
    return bg, plume, q_val, plume, plume


def predict_concentration(model, xyt, u, v, sigma_src):
    return recurrent_plume_value(model, xyt, sigma_src)


def concentration_from_components(bg, plume, q_val, source_term):
    return source_term
