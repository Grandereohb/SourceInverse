"""IA-only differentiable correlation time, retaining exact Q linearity."""

import math
import numpy as np
import torch
from numpy.polynomial.legendre import leggauss
from ia_metec_point_adapter import configure


def _effective(age, tau):
    x = age / tau
    series_x = torch.clamp(x, max=.1)
    # Horner form retains the short-time term and its derivative.
    coefficients = [(-1.) ** n / math.factorial(n) for n in range(2, 15)]
    value = torch.zeros_like(x)
    for coefficient in reversed(coefficients):
        value = coefficient + series_x * value
    return torch.where(x < .1, age * series_x * value, age + tau * torch.expm1(-x))


def _weight(low, width, tau):
    x = width / tau
    series_x = torch.clamp(x, max=.1)
    coefficients = [(-1.) ** n * (2. ** n - 2) / (math.factorial(n) * (n + 1))
                    for n in range(2, 15)]
    polynomial = torch.zeros_like(x)
    for coefficient in reversed(coefficients):
        polynomial = coefficient + series_x * polynomial
    b = torch.where(x < .1, width * series_x * series_x * polynomial,
                    width + 2 * tau * torch.expm1(-x) - .5 * tau * torch.expm1(-2 * x))
    c = torch.exp(-low / tau)
    d = -torch.expm1(-low / tau)
    return d * d * width + 2 * d * c * _effective(width, tau) + c * c * b


class CorrelatedResponse:
    """Five physical coordinates: x, y, Dperp, Dcontrast, tau_seconds.

    Tau has a physical one-sided derivative at zero. A cached moment table is
    independent of source location, D, and Q, and shared by their evaluations.
    """

    def __init__(self, base, data):
        self.base = base
        self.plan = base.plan
        self.xyt, self.basis = base.xyt, base.basis
        p = self.plan
        self.original = (p.parallel, p.perpendicular, p.age)
        edges = np.arange(len(data['concentration']) + 1) * 60.
        nodes = leggauss(p.time_order)[0]
        ends = np.concatenate([60 * minute + 30 * (nodes + 1) for minute in p.observation_minutes])
        self.ends = p.age.new_tensor(ends)
        self.edges = p.age.new_tensor(edges)
        wind = np.asarray(data['wind_xy'], float)
        speed = np.linalg.norm(wind, axis=1)
        direction = wind / np.where(speed > 0, speed, 1.)[:, None]
        direction[speed == 0] = [1., 0.]
        packed = np.column_stack((direction[:, 0] ** 2, direction[:, 0] * direction[:, 1], direction[:, 1] ** 2))
        self.parallel = p.age.new_tensor(packed)
        self.perpendicular = p.age.new_tensor([1., 0., 1.]) - self.parallel
        self.release = self.ends[:, None] - p.age
        # Padded entries have zero release weights, so assign harmless geometry.
        self.valid = p.weighted_basis.abs().sum(-1) > 0
        self.release = torch.where(self.valid, self.release, self.ends[:, None] / 2)
        indices = torch.searchsorted(self.edges, self.release.contiguous(), right=True) - 1
        self.release_parallel = self.parallel[indices]
        self.release_perpendicular = self.perpendicular[indices]
        end_indices = torch.searchsorted(self.edges, self.ends, right=False) - 1
        self.end_parallel = self.parallel[end_indices]
        self.end_perpendicular = self.perpendicular[end_indices]
        lag = torch.where(self.valid, end_indices[:, None] - indices, 0)
        self.lag = lag[..., None].expand(-1, -1, 3)
        segment_indices = end_indices[:, None] - torch.arange(int(lag.max()) + 1, device=p.device)
        segment_valid = segment_indices >= 0
        segment_indices = torch.clamp(segment_indices, min=0)
        segment_upper = torch.minimum(self.ends[:, None], self.edges[segment_indices + 1])
        self.full_width = torch.where(segment_valid, segment_upper - self.edges[segment_indices], 0.)
        self.full_low_age = self.ends[:, None] - segment_upper
        self.segment_parallel = self.parallel[segment_indices]
        self.segment_perpendicular = self.perpendicular[segment_indices]
        partial_upper = torch.minimum(self.ends[:, None], self.edges[indices + 1])
        self.partial_width = torch.where(self.valid, partial_upper - self.release, 0.)
        self.partial_low_age = self.ends[:, None] - partial_upper
        self.cached_tau = None
        self.cached = None
        self.calls = 0

    def _moments(self, tau):
        age = self.ends[:, None] - self.release
        initial = .5 * tau * (-torch.expm1(-age / tau)) ** 2
        # A complete wind interval is shared by all releases preceding it.
        # Prefix sums avoid allocating a group x release x history tensor.
        full_weight = _weight(self.full_low_age, self.full_width, tau)
        partial = initial + _weight(self.partial_low_age, self.partial_width, tau)
        def integrate(axes, release_axes):
            prefix = torch.cumsum(full_weight[..., None] * axes, dim=1)
            prefix = torch.cat((torch.zeros_like(prefix[:, :1]), prefix), dim=1)
            return torch.gather(prefix, 1, self.lag) + partial[..., None] * release_axes
        par = integrate(self.segment_parallel, self.release_parallel)
        per = integrate(self.segment_perpendicular, self.release_perpendicular)
        return par, per, _effective(age, tau)

    def _prepare(self, tau):
        key = float(tau)
        if key == self.cached_tau:
            return self.cached
        if key < 0 or not np.isfinite(key):
            raise ValueError('Finite nonnegative correlation time required')
        if key == 0:
            value = self.original
            derivative = (.5 * self.release_parallel - 1.5 * self.end_parallel[:, None],
                          .5 * self.release_perpendicular - 1.5 * self.end_perpendicular[:, None],
                          -torch.ones_like(self.original[2]))
        else:
            value, derivative = torch.func.jvp(self._moments, (tau,), (torch.ones_like(tau),))
        self.cached_tau, self.cached = key, (value, derivative)
        return self.cached

    def __call__(self, model, sigma, xyt, basis, *, physical_parameters, track_gradients=False):
        if track_gradients:
            # The outer solver uses the explicit Jacobian; don't silently promise autograd through caching.
            raise ValueError('Use matrix_and_jacobian for correlated response derivatives')
        theta = physical_parameters
        if theta.numel() != 5:
            raise ValueError('Five correlated-transport coordinates required')
        (par, per, age), _ = self._prepare(theta[4])
        self.plan.parallel, self.plan.perpendicular, self.plan.age = par, per, age
        try:
            return self.base(model, sigma, xyt, basis, physical_parameters=theta[:4])
        finally:
            self.plan.parallel, self.plan.perpendicular, self.plan.age = self.original

    @torch.no_grad()
    def matrix_and_jacobian(self, model, sigma, xyt, basis, physical_parameters):
        theta = physical_parameters
        (par, per, age), (dpar, dper, dage) = self._prepare(theta[4])
        p = self.plan
        p.parallel, p.perpendicular, p.age = par, per, age
        try:
            value, derivative = self.base.matrix_and_jacobian(model, sigma, xyt, basis, theta[:4])
            dp = (theta[2] + theta[3]) * p.d_scale
            dq = theta[2] * p.d_scale
            extra = []
            chunk = max(1, 2_000_000 // (len(p.xy) * p.releases))
            for first in range(0, p.groups, chunk):
                last = min(first + chunk, p.groups)
                valid = self.valid[first:last]
                covariance = 2 * (dp * par[first:last] + dq * per[first:last])
                dcov = 2 * (dp * dpar[first:last] + dq * dper[first:last])
                xx = torch.where(valid, covariance[..., 0], 1.)[:, None]
                xy = torch.where(valid, covariance[..., 1], 0.)[:, None]
                yy = torch.where(valid, covariance[..., 2], 1.)[:, None]
                dxx, dxy, dyy = [dcov[..., k][:, None] for k in range(3)]
                determinant = xx * yy - xy * xy
                delta = p.xy[None, :, None] - theta[:2] * p.length - p.displacement[first:last, None]
                dx, dy = delta[..., 0], delta[..., 1]
                alpha, beta = (yy * dx - xy * dy) / determinant, (xx * dy - xy * dx) / determinant
                horizontal = torch.exp(-.5 * (dx * alpha + dy * beta)) / (2 * np.pi * torch.sqrt(determinant))
                dh = .5 * horizontal * (alpha * alpha * dxx + 2 * alpha * beta * dxy + beta * beta * dyy -
                    (yy * dxx - 2 * xy * dxy + xx * dyy) / determinant)
                variance = 2 * dq * age[first:last, None]
                variance = torch.where(valid[:, None], variance, 1.)
                dv = 2 * dq * dage[first:last, None]
                z = p.z[None, :, None]
                near = torch.exp(-.5 * (z - p.height) ** 2 / variance) / torch.sqrt(2 * np.pi * variance)
                reflected = torch.exp(-.5 * (z + p.height) ** 2 / variance) / torch.sqrt(2 * np.pi * variance)
                vertical = near + reflected
                dvertical = dv * (near * (-.5 / variance + (z - p.height) ** 2 / (2 * variance ** 2)) +
                    reflected * (-.5 / variance + (z + p.height) ** 2 / (2 * variance ** 2)))
                extra.append(torch.einsum('gpr,grb->gpb', dh * vertical + horizontal * dvertical,
                                         p.weighted_basis[first:last]))
            fields = torch.cat(extra).reshape(p.minutes, p.time_order, p.beams, 1, -1)
            tau_derivative = torch.einsum('mtbqk,t,q->mbk', fields, p.time_weights, p.beam_weights)
            tau_derivative = tau_derivative.reshape(p.minutes * p.beams, -1)[self.base.indices]
            if not bool(torch.isfinite(tau_derivative).all()):
                raise FloatingPointError('Nonfinite correlation-time derivative')
            return value, torch.cat((derivative, tau_derivative[..., None]), -1)
        finally:
            p.parallel, p.perpendicular, p.age = self.original


def configure_variable_correlation(data, height_m, **kwargs):
    result = list(configure(data, height_m, **kwargs))
    result[3] = CorrelatedResponse(result[3], data)
    return tuple(result)
