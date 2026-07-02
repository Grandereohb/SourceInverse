import torch
import torch.nn.functional as F

from config import (
    RECURRENT_DECAY,
    RECURRENT_GRID_NX,
    RECURRENT_GRID_NY,
    RECURRENT_INITIAL_RELEASE_FRACTION,
    RECURRENT_SOURCE_SCALE,
    RECURRENT_SUBSTEPS,
)


FIELD_MODE = "recurrent_pde"


def _match_column(tensor):
    if tensor.dim() == 1:
        return tensor.view(-1, 1)
    return tensor


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
    nx=None,
    ny=None,
):
    nx = max(8, int(RECURRENT_GRID_NX if nx is None else nx))
    ny = max(8, int(RECURRENT_GRID_NY if ny is None else ny))
    dtype = torch.float32
    device = next(model.parameters()).device

    model.recurrent_d_min_norm = float(d_min_norm)
    model.recurrent_d_scale_norm = float(d_scale_norm)
    model.recurrent_x_grid = torch.linspace(
        float(x_min), float(x_max), steps=nx, dtype=dtype, device=device
    )
    model.recurrent_y_grid = torch.linspace(
        float(y_min), float(y_max), steps=ny, dtype=dtype, device=device
    )
    model.recurrent_times = torch.as_tensor(
        t_values, dtype=dtype, device=device
    ).view(-1)
    model.recurrent_u = torch.as_tensor(u_values, dtype=dtype, device=device).view(-1)
    model.recurrent_v = torch.as_tensor(v_values, dtype=dtype, device=device).view(-1)


def _has_recurrent_context(model):
    return (
        hasattr(model, "recurrent_x_grid")
        and hasattr(model, "recurrent_y_grid")
        and hasattr(model, "recurrent_times")
        and model.recurrent_times.numel() > 0
    )


def _sample_grid_bilinear(field, x_query, y_query, x_grid, y_grid):
    x_query = _match_column(x_query).view(-1)
    y_query = _match_column(y_query).view(-1)
    ny, nx = field.shape
    x_min = x_grid[0]
    x_max = x_grid[-1]
    y_min = y_grid[0]
    y_max = y_grid[-1]
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

    flat = field.reshape(-1)
    f00 = flat[y0 * nx + x0].view(-1, 1)
    f10 = flat[y0 * nx + x1].view(-1, 1)
    f01 = flat[y1 * nx + x0].view(-1, 1)
    f11 = flat[y1 * nx + x1].view(-1, 1)
    return (
        (1.0 - wx) * (1.0 - wy) * f00
        + wx * (1.0 - wy) * f10
        + (1.0 - wx) * wy * f01
        + wx * wy * f11
    )


def _advect_field(field, x_grid, y_grid, u, v, dt):
    yy, xx = torch.meshgrid(y_grid, x_grid, indexing="ij")
    x_back = xx.reshape(-1) - u * dt
    y_back = yy.reshape(-1) - v * dt
    return _sample_grid_bilinear(field, x_back, y_back, x_grid, y_grid).view_as(field)


def _diffuse_field(field, x_grid, y_grid, diffusion, dt):
    dx = torch.clamp(x_grid[1] - x_grid[0], min=1e-8)
    dy = torch.clamp(y_grid[1] - y_grid[0], min=1e-8)
    padded = F.pad(field.view(1, 1, *field.shape), (1, 1, 1, 1), mode="replicate")[
        0, 0
    ]
    center = padded[1:-1, 1:-1]
    lap = (
        (padded[1:-1, 2:] - 2.0 * center + padded[1:-1, :-2]) / (dx**2)
        + (padded[2:, 1:-1] - 2.0 * center + padded[:-2, 1:-1]) / (dy**2)
    )
    stable_max = 0.22 * torch.minimum(dx**2, dy**2) / torch.clamp(dt, min=1e-8)
    diffusion_eff = torch.clamp(
        diffusion, min=0.0, max=float(stable_max.detach().cpu().item())
    )
    return torch.clamp(field + diffusion_eff * dt * lap, min=0.0)


def _source_grid(model, t_value, x_grid, y_grid, sigma_src):
    yy, xx = torch.meshgrid(y_grid, x_grid, indexing="ij")
    xs, ys = model.source_xy(t_value.view(1, 1))
    sigma = max(float(sigma_src), 1e-4)
    src = torch.exp(-((xx - xs) ** 2 + (yy - ys) ** 2) / (2.0 * sigma**2))
    dx = torch.clamp(x_grid[1] - x_grid[0], min=1e-8)
    dy = torch.clamp(y_grid[1] - y_grid[0], min=1e-8)
    mass = torch.clamp(torch.sum(src) * dx * dy, min=1e-8)
    return src / mass


def _advance_recurrent_step(
    field,
    source,
    q_value,
    x_grid,
    y_grid,
    u_value,
    v_value,
    diffusion,
    decay,
    source_scale,
    dt,
):
    field = field + source_scale * q_value * source * dt
    field = _advect_field(field, x_grid, y_grid, u_value, v_value, dt)
    field = _diffuse_field(field, x_grid, y_grid, diffusion, dt)
    if decay > 0.0:
        field = field * torch.exp(-dt * decay)
    return field


def recurrent_plume_fields(model, sigma_src):
    if not _has_recurrent_context(model):
        raise RuntimeError("Recurrent plume context has not been configured.")

    x_grid = model.recurrent_x_grid.to(device=model.xs.device, dtype=model.xs.dtype)
    y_grid = model.recurrent_y_grid.to(device=model.xs.device, dtype=model.xs.dtype)
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
    substeps = max(1, int(RECURRENT_SUBSTEPS))
    decay = max(float(RECURRENT_DECAY), 0.0)
    source_scale = float(RECURRENT_SOURCE_SCALE)

    if t_values.numel() == 1:
        source = _source_grid(model, t_values[0], x_grid, y_grid, sigma_src)
        field = field + source_scale * model.Q(t_values[0].view(1, 1)).view(()) * source
        fields.append(field)
        return torch.stack(fields, dim=0)

    warmup_fraction = max(float(RECURRENT_INITIAL_RELEASE_FRACTION), 0.0)
    if warmup_fraction > 0.0:
        first_dt_total = torch.clamp(t_values[1] - t_values[0], min=1e-6)
        warmup_dt_total = first_dt_total * warmup_fraction
        warmup_steps = max(1, int(round(substeps * warmup_fraction)))
        warmup_dt = warmup_dt_total / warmup_steps
        warmup_source = _source_grid(model, t_values[0], x_grid, y_grid, sigma_src)
        warmup_q = model.Q(t_values[0].view(1, 1)).view(())
        for _ in range(warmup_steps):
            field = _advance_recurrent_step(
                field,
                warmup_source,
                warmup_q,
                x_grid,
                y_grid,
                u_values[0],
                v_values[0],
                diffusion,
                decay,
                source_scale,
                warmup_dt,
            )

    fields.append(field)
    for i in range(t_values.numel() - 1):
        t_i = t_values[i]
        dt_total = torch.clamp(t_values[i + 1] - t_i, min=1e-6)
        dt = dt_total / substeps
        source = _source_grid(model, t_i, x_grid, y_grid, sigma_src)
        q_i = model.Q(t_i.view(1, 1)).view(())
        for _ in range(substeps):
            field = _advance_recurrent_step(
                field,
                source,
                q_i,
                x_grid,
                y_grid,
                u_values[i],
                v_values[i],
                diffusion,
                decay,
                source_scale,
                dt,
            )
        fields.append(field)

    return torch.stack(fields, dim=0)


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
