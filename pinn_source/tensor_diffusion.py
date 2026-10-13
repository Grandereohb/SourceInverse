"""Positive lattice diffusion for a constant rotated tensor on a finite grid.

The generator matches K through its second moments, even when sqrt(2 K dt)
is smaller than a cell. Uniformization approximates its matrix exponential;
zero ghost cells kill outward jumps rather than returning or renormalizing mass.
This is a discrete diffusion semigroup, not an exact continuous Gaussian.
"""

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


@dataclass
class DiffusionPlan:
    kernel: torch.Tensor
    poisson_weights: tuple
    repetitions: int
    direction: tuple
    rates: torch.Tensor
    tensor: torch.Tensor
    tail_bound: float


def _positive_direction(a, c, b, max_direction=128):
    """Choose the shortest positive stencil with nonnegative axis rates."""
    if b == 0.:
        return 1, 1
    lower, upper = b/c, a/b
    # SPD implies a nonempty interval. Do not silently replace an unresolvable
    # tensor by a scalar or truncate negative transition probabilities.
    choices = []
    for n in range(1, max_direction + 1):
        m = max(1, math.ceil(lower*n - 1e-12))
        if m <= max_direction and m/n <= upper + 1e-12:
            choices.append((m*m+n*n, m+n, m, n))
    if not choices:
        raise ValueError("Tensor requires a diffusion stencil wider than 128 cells.")
    _, _, m, n = min(choices)
    return m, n


def build_diffusion_plan(x_grid, y_grid, d_parallel, d_perp, u, v, dt,
                         *, tail_tolerance=1e-10):
    """Build one differentiable plan; reuse it for all substeps of an interval."""
    dx, dy = x_grid[1]-x_grid[0], y_grid[1]-y_grid[0]
    prototype = x_grid
    cast = lambda value: torch.as_tensor(value, dtype=prototype.dtype, device=prototype.device).reshape(())
    dp, dq, u, v, duration = map(cast, (d_parallel, d_perp, u, v, dt))
    values = torch.stack((dx, dy, dp, dq, u, v, duration)).detach().cpu().double().tolist()
    if not all(math.isfinite(x) for x in values) or min(values[:4]) <= 0 or values[-1] < 0:
        raise ValueError("Diffusion requires finite positive spacing/D and nonnegative dt.")
    if not 0 < tail_tolerance < 1:
        raise ValueError("tail_tolerance must lie between zero and one.")
    speed2 = u*u + v*v
    if values[4]**2 + values[5]**2 <= 1e-24:
        kxx = kyy = (dp+dq)*.5
        kxy = dp*0.
    else:
        kxx = (dp*u*u + dq*v*v)/speed2
        kyy = (dp*v*v + dq*u*u)/speed2
        kxy = (dp-dq)*u*v/speed2
    tensor = torch.stack((kxx, kxy, kxy, kyy)).reshape(2, 2)
    a, c, b = kxx/(dx*dx), kyy/(dy*dy), kxy/(dx*dy)
    a0, c0, b0 = torch.stack((a, c, b)).detach().cpu().double().tolist()
    m, n = _positive_direction(a0, c0, abs(b0))
    sign = 1 if b0 >= 0 else -1
    wd = b.abs()/(m*n)
    wx, wy = a-wd*m*m, c-wd*n*n
    rates = torch.stack((wx, wy, wd))
    # Only floating-point cancellation at a stencil transition is tolerated.
    if float(rates.detach().min().cpu()) < -1e-6*max(a0, c0):
        raise ValueError("Negative diffusion rate: tensor stencil construction failed.")
    rates = rates.clamp_min(0.)
    rho = 2*rates.sum()
    mu_total = float((rho*duration).detach().cpu())
    repetitions = max(1, math.ceil(mu_total/8.))
    if repetitions > 10000:
        raise ValueError("Diffusion interval too stiff for uniformization; refine its grid/time plan.")
    mu = rho*duration/repetitions
    mu0 = mu_total/repetitions
    probability = math.exp(-mu0)
    cumulative = probability
    order = 0
    while 1.-cumulative > tail_tolerance/repetitions:
        order += 1
        probability *= mu0/order
        cumulative += probability
        if order > 128:
            raise RuntimeError("Poisson diffusion expansion failed to converge.")
    weight = torch.exp(-mu)
    weights = [weight]
    for k in range(1, order+1):
        weight = weight*mu/k
        weights.append(weight)
    radius_x, radius_y = max(1, m), max(1, n)
    masks = torch.zeros((3, 2*radius_y+1, 2*radius_x+1), dtype=prototype.dtype, device=prototype.device)
    for j, (sx, sy) in enumerate(((1, 0), (0, 1), (m, sign*n))):
        masks[j, radius_y+sy, radius_x+sx] = 1.
        masks[j, radius_y-sy, radius_x-sx] = 1.
    kernel = ((rates/rho)[:, None, None]*masks).sum(0)
    return DiffusionPlan(kernel, tuple(weights), repetitions, (m, sign*n),
                         rates, tensor, min(1., repetitions*max(0., 1.-cumulative)))


def apply_diffusion_plan(field, plan):
    """Linear, nonnegative, substochastic diffusion; leading dims are batches."""
    height, width = field.shape[-2:]
    value = field.reshape(-1, 1, height, width)
    kernel = plan.kernel[None, None]
    padding = (kernel.shape[-2]//2, kernel.shape[-1]//2)
    for _ in range(plan.repetitions):
        power = value
        result = plan.poisson_weights[0]*power
        for weight in plan.poisson_weights[1:]:
            power = F.conv2d(power, kernel, padding=padding)
            result = result + weight*power
        value = result
    return value.reshape(field.shape)
