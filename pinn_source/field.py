import torch

from config import (
    FIELD_MODE,
    GATE_CORE_SCALE,
    GATE_CROSS_SCALE,
    GATE_CROSS_MIN,
    GATE_STEEPNESS_SCALE,
    GATE_STEEPNESS_MIN,
    GATE_DECAY_SCALE,
    GATE_DECAY_MIN,
    GATE_FLOOR,
    GATE_DOWNWIND_BROADEN,
    ANALYTIC_PLUME_LAG_STEPS,
    ANALYTIC_PLUME_MAX_AGE,
    ANALYTIC_PLUME_MIN_AGE,
    ANALYTIC_PLUME_AGE_DECAY,
    ANALYTIC_PLUME_ALONG_SPREAD,
    ANALYTIC_PLUME_CROSS_SPREAD,
    ANALYTIC_PLUME_TRANSPORT_SCALE,
    ANALYTIC_PLUME_SOURCE_CORE_WEIGHT,
)


def _match_column(tensor):
    if tensor.dim() == 1:
        return tensor.view(-1, 1)
    return tensor


def _interp_history(t_query, t_hist, value_hist):
    t_query = _match_column(t_query)
    if t_hist.numel() == 0 or value_hist.numel() == 0:
        return torch.zeros_like(t_query)
    t_hist = t_hist.to(device=t_query.device, dtype=t_query.dtype).view(-1)
    value_hist = value_hist.to(device=t_query.device, dtype=t_query.dtype).view(-1)
    if t_hist.numel() == 1:
        return value_hist[0].expand_as(t_query)

    t_flat = t_query.reshape(-1)
    idx_hi = torch.bucketize(t_flat.contiguous(), t_hist)
    idx_hi = torch.clamp(idx_hi, min=1, max=t_hist.numel() - 1)
    idx_lo = idx_hi - 1
    t_lo = t_hist[idx_lo]
    t_hi = t_hist[idx_hi]
    v_lo = value_hist[idx_lo]
    v_hi = value_hist[idx_hi]
    alpha = (t_flat - t_lo) / torch.clamp(t_hi - t_lo, min=1e-6)
    alpha = torch.clamp(alpha, min=0.0, max=1.0)
    return (v_lo + alpha * (v_hi - v_lo)).view_as(t_query)


def _transport_wind_at(model, t_query, fallback_u, fallback_v):
    if (
        hasattr(model, "transport_times")
        and model.transport_times.numel() > 0
        and model.transport_u.numel() > 0
        and model.transport_v.numel() > 0
    ):
        u_hist = _interp_history(t_query, model.transport_times, model.transport_u)
        v_hist = _interp_history(t_query, model.transport_times, model.transport_v)
        return u_hist, v_hist
    return _match_column(fallback_u), _match_column(fallback_v)


def source_aligned_coords(xyt, u, v, model):
    u = _match_column(u)
    v = _match_column(v)

    if hasattr(model, "source_xy"):
        xs, ys = model.source_xy(xyt[:, 2:3])
    else:
        xs, ys = model.xs, model.ys
    dx = xyt[:, 0:1] - xs
    dy = xyt[:, 1:2] - ys

    speed = torch.sqrt(u**2 + v**2 + 1e-12)
    ex = u / speed
    ey = v / speed

    along = dx * ex + dy * ey
    cross = -dx * ey + dy * ex
    return along, cross, dx, dy


def source_gate(xyt, u, v, model, sigma_src):
    along, cross, dx, dy = source_aligned_coords(xyt, u, v, model)

    sigma_core = max(float(sigma_src), 1e-4)
    sigma_core_gate = max(GATE_CORE_SCALE * sigma_core, 1e-4)
    sigma_cross = max(GATE_CROSS_SCALE * sigma_core, GATE_CROSS_MIN)
    gate_steepness = max(GATE_STEEPNESS_SCALE * sigma_core, GATE_STEEPNESS_MIN)
    decay_length = max(GATE_DECAY_SCALE * sigma_core, GATE_DECAY_MIN)
    downwind_broaden = max(float(GATE_DOWNWIND_BROADEN), 1.0)
    gate_floor = min(max(float(GATE_FLOOR), 0.0), 0.95)

    source_core = torch.exp(-(dx**2 + dy**2) / (2.0 * sigma_core_gate**2))
    downwind_weight = torch.sigmoid(along / gate_steepness) ** 2
    sigma_cross_eff = sigma_cross * (1.0 + (downwind_broaden - 1.0) * downwind_weight)
    plume_tail = (
        downwind_weight
        * torch.exp(-(cross**2) / (2.0 * sigma_cross_eff**2))
        * torch.exp(-torch.relu(along) / decay_length)
    )
    gate_raw = torch.clamp(source_core + plume_tail, min=0.0, max=1.0)
    return gate_floor + (1.0 - gate_floor) * gate_raw


def analytic_plume_kernel(xyt, u, v, model, sigma_src):
    _, _, dx, dy = source_aligned_coords(xyt, u, v, model)

    sigma_core = max(float(sigma_src), 1e-4)
    sigma_core_gate = max(GATE_CORE_SCALE * sigma_core, 1e-4)
    sigma_cross = max(GATE_CROSS_SCALE * sigma_core, GATE_CROSS_MIN)
    lag_steps = max(1, int(ANALYTIC_PLUME_LAG_STEPS))
    max_age = max(float(ANALYTIC_PLUME_MAX_AGE), 0.0)
    min_age = max(float(ANALYTIC_PLUME_MIN_AGE), 0.0)
    min_age = min(min_age, max_age)
    age_decay = max(float(ANALYTIC_PLUME_AGE_DECAY), 1e-4)
    along_spread = max(float(ANALYTIC_PLUME_ALONG_SPREAD), 0.0)
    cross_spread = max(float(ANALYTIC_PLUME_CROSS_SPREAD), 0.0)
    transport_scale = max(float(ANALYTIC_PLUME_TRANSPORT_SCALE), 0.0)
    source_core_weight = min(max(float(ANALYTIC_PLUME_SOURCE_CORE_WEIGHT), 0.0), 1.0)

    if lag_steps == 1 or max_age <= 0.0:
        along, cross, _, _ = source_aligned_coords(xyt, u, v, model)
        source_core = torch.exp(-(dx**2 + dy**2) / (2.0 * sigma_core_gate**2))
        downwind_weight = torch.sigmoid(along / max(GATE_STEEPNESS_MIN, 1e-4)) ** 2
        plume_tail = (
            downwind_weight
            * torch.exp(-(cross**2) / (2.0 * sigma_cross**2))
            * torch.exp(-torch.relu(along) / max(GATE_DECAY_MIN, 1e-4))
        )
        return torch.clamp(source_core + plume_tail, min=0.0, max=1.0)

    ages = torch.linspace(
        min_age,
        max_age,
        steps=lag_steps,
        dtype=xyt.dtype,
        device=xyt.device,
    )
    t = xyt[:, 2:3]
    if hasattr(model, "source_xy"):
        xs, ys = model.source_xy(t)
    else:
        xs, ys = model.xs, model.ys

    kernel_sum = torch.zeros_like(t)
    weight_sum = torch.zeros_like(t)

    for age in ages:
        valid = (t >= age).to(xyt.dtype)
        t_emit = torch.clamp(t - age, min=0.0)
        u_emit, v_emit = _transport_wind_at(model, t_emit, u, v)
        center_x = xs + transport_scale * u_emit * age
        center_y = ys + transport_scale * v_emit * age
        dx_age = xyt[:, 0:1] - center_x
        dy_age = xyt[:, 1:2] - center_y

        speed_emit = torch.sqrt(u_emit**2 + v_emit**2 + 1e-12)
        ex = u_emit / speed_emit
        ey = v_emit / speed_emit
        along_age = dx_age * ex + dy_age * ey
        cross_age = -dx_age * ey + dy_age * ex
        age_root = torch.sqrt(age + 1e-6)
        sigma_along = sigma_core_gate + along_spread * age_root
        sigma_cross_eff = sigma_cross * (1.0 + cross_spread * age_root)
        age_weight = torch.exp(-age / age_decay)
        q_emit = model.Q(t_emit)
        puff = (
            torch.exp(-(along_age**2) / (2.0 * sigma_along**2))
            * torch.exp(-(cross_age**2) / (2.0 * sigma_cross_eff**2))
            * q_emit
            * age_weight
            * valid
        )
        kernel_sum = kernel_sum + puff
        weight_sum = weight_sum + age_weight * valid

    kernel = kernel_sum / torch.clamp(weight_sum, min=1e-6)
    if source_core_weight > 0.0:
        q_now = model.Q(t)
        source_core = torch.exp(-(dx**2 + dy**2) / (2.0 * sigma_core_gate**2))
        kernel = kernel + source_core_weight * source_core * q_now
    return torch.clamp(kernel, min=0.0)


def field_components(model, xyt, u, v, sigma_src):
    bg = model.background(xyt[:, 2:3])
    t = xyt[:, 2:3]
    q_val = model.Q(t)
    source_bias = model.source_bias()
    if FIELD_MODE == "analytic_plume":
        plume = analytic_plume_kernel(xyt, u, v, model, sigma_src)
        gate = plume
        source_term = (1.0 + source_bias) * plume
        return bg, plume, q_val, gate, source_term

    along, cross, _, _ = source_aligned_coords(xyt, u, v, model)
    plume_features = torch.cat([torch.relu(along), cross, xyt[:, 2:3]], dim=1)
    plume = model.plume_strength(plume_features)
    gate = source_gate(xyt, u, v, model, sigma_src)
    source_term = gate * (source_bias + plume) * q_val
    return bg, plume, q_val, gate, source_term


def predict_concentration(model, xyt, u, v, sigma_src):
    bg, plume, q_val, gate, source_term = field_components(model, xyt, u, v, sigma_src)
    return concentration_from_components(bg, plume, q_val, source_term)


def concentration_from_components(bg, plume, q_val, source_term):
    if FIELD_MODE == "no_gate":
        return bg + plume * q_val
    if FIELD_MODE == "no_background":
        return source_term
    if FIELD_MODE == "minimal":
        return plume * q_val
    if FIELD_MODE == "analytic_plume":
        return source_term
    return bg + source_term
