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
)


def _match_column(tensor):
    if tensor.dim() == 1:
        return tensor.view(-1, 1)
    return tensor


def source_aligned_coords(xyt, u, v, model):
    u = _match_column(u)
    v = _match_column(v)

    dx = xyt[:, 0:1] - model.xs
    dy = xyt[:, 1:2] - model.ys

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
    gate_raw = torch.clamp(source_core + plume_tail, min=0.0)
    return gate_floor + (1.0 - gate_floor) * gate_raw


def field_components(model, xyt, u, v, sigma_src):
    bg = model.background(xyt[:, 2:3])
    along, cross, _, _ = source_aligned_coords(xyt, u, v, model)
    plume_features = torch.cat([torch.relu(along), cross, xyt[:, 2:3]], dim=1)
    plume = model.plume_strength(plume_features)
    source_bias = model.source_bias()
    gate = source_gate(xyt, u, v, model, sigma_src)
    t = xyt[:, 2:3]
    q_val = model.Q(t)
    source_term = gate * (source_bias + plume) * q_val
    return bg, plume, q_val, gate, source_term


def predict_concentration(model, xyt, u, v, sigma_src):
    bg, plume, q_val, gate, source_term = field_components(model, xyt, u, v, sigma_src)

    if FIELD_MODE == "no_gate":
        return bg + plume * q_val
    if FIELD_MODE == "no_background":
        return source_term
    if FIELD_MODE == "minimal":
        return plume * q_val
    return bg + source_term
