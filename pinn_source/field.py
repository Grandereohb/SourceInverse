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
)


def _match_column(tensor):
    if tensor.dim() == 1:
        return tensor.view(-1, 1)
    return tensor


def source_gate(xyt, u, v, model, sigma_src):
    u = _match_column(u)
    v = _match_column(v)

    dx = xyt[:, 0:1] - model.xs
    dy = xyt[:, 1:2] - model.ys

    speed = torch.sqrt(u**2 + v**2 + 1e-12)
    ex = u / speed
    ey = v / speed

    along = dx * ex + dy * ey
    cross = -dx * ey + dy * ex

    sigma_core = max(float(sigma_src), 1e-4)
    sigma_core_gate = max(GATE_CORE_SCALE * sigma_core, 1e-4)
    sigma_cross = max(GATE_CROSS_SCALE * sigma_core, GATE_CROSS_MIN)
    gate_steepness = max(GATE_STEEPNESS_SCALE * sigma_core, GATE_STEEPNESS_MIN)
    decay_length = max(GATE_DECAY_SCALE * sigma_core, GATE_DECAY_MIN)

    source_core = torch.exp(-(dx**2 + dy**2) / (2.0 * sigma_core_gate**2))
    plume_tail = (
        torch.sigmoid(along / gate_steepness)
        * torch.exp(-(cross**2) / (2.0 * sigma_cross**2))
        * torch.exp(-torch.relu(along) / decay_length)
    )
    return torch.clamp(source_core + plume_tail, min=0.0)


def field_components(model, xyt, u, v, sigma_src):
    bg = model.background(xyt[:, 2:3])
    plume = model.plume_strength(xyt)
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
