"""Research-only integrated OU turbulent velocities with piecewise wind axes.

dX = u(t) dt + V dt; dV = -V/tau dt + sqrt(2 K(t))/tau dW.
At release, Cov(V)=K(release)/tau. D in K is the long-time diffusivity.
This homogeneous mean-wind model does not include an affine velocity gradient.
"""

import math
import numpy as np
from ia_metec_point_adapter import configure
from ia_physical_beam_adapter import _integral_at


def effective_age(age, tau):
    """Half the constant-K position variance per unit long-time diffusivity."""
    age = np.asarray(age, dtype=float)
    if not np.isfinite(tau) or tau < 0 or not np.isfinite(age).all() or np.any(age < 0):
        raise ValueError('Finite nonnegative ages and correlation time required')
    if tau == 0:
        return age.copy()
    x = age / tau
    small = x < .1
    # Direct subtraction loses the ballistic x^2 term at very young ages.
    series = sum((-1.) ** n * x[small] ** (n - 1) / math.factorial(n) for n in range(2, 15))
    result = np.array(age + tau * np.expm1(-x))
    result[small] = age[small] * series
    return result


def integrated_velocity_weight(low_age, high_age, tau):
    """Positive integral of (1-exp(-age/tau))^2 over an age interval."""
    low, high = np.broadcast_arrays(np.asarray(low_age, float), np.asarray(high_age, float))
    if np.any(low < 0) or np.any(high < low) or not np.isfinite([low, high]).all():
        raise ValueError('Ordered finite nonnegative integration ages required')
    if not np.isfinite(tau) or tau < 0:
        raise ValueError('Nonnegative finite correlation time required')
    h = high - low
    if tau == 0:
        return h
    x = h / tau
    small = x < .1
    a = effective_age(h, tau)
    b = np.array(h + 2 * tau * np.expm1(-x) - .5 * tau * np.expm1(-2 * x))
    series = sum((-1.) ** n * (2. ** n - 2) * x[small] ** n /
                 (math.factorial(n) * (n + 1)) for n in range(2, 15))
    b[small] = h[small] * series
    c = np.exp(-low / tau)
    d = -np.expm1(-low / tau)
    return d * d * h + 2 * d * c * a + c * c * b


class CorrelatedWindHistory:
    """Exact piecewise-axis integrated OU covariance, without inferred wind."""

    def __init__(self, edges, velocity, tau):
        self.edges = np.asarray(edges, float)
        self.velocity = np.asarray(velocity, float)
        if (self.edges.ndim != 1 or len(self.edges) < 2 or self.edges[0] != 0 or
                np.any(np.diff(self.edges) <= 0) or self.velocity.shape != (len(self.edges) - 1, 2) or
                not np.isfinite(self.edges).all() or not np.isfinite(self.velocity).all()):
            raise ValueError('Finite increasing edges and matching ENU velocities required')
        effective_age(np.array([0.]), tau)
        self.tau = float(tau)
        speed = np.linalg.norm(self.velocity, axis=1)
        direction = self.velocity / np.where(speed > 0, speed, 1.)[:, None]
        direction[speed == 0] = [1., 0.]
        self.parallel = direction[:, :, None] * direction[:, None, :]
        self.perpendicular = np.eye(2) - self.parallel

    def relative(self, end, releases):
        releases = np.asarray(releases, float)
        if (releases.ndim != 1 or not np.isfinite(releases).all() or not np.isfinite(end) or
                np.any(releases < 0) or np.any(releases >= end) or end > self.edges[-1]):
            raise ValueError('Releases must precede observation within the wind clock')
        age = end - releases
        release_index = np.searchsorted(self.edges, releases, side='right') - 1
        initial = (np.zeros_like(age) if self.tau == 0 else
                   .5 * self.tau * (-np.expm1(-age / self.tau)) ** 2)
        par = initial[:, None, None] * self.parallel[release_index]
        per = initial[:, None, None] * self.perpendicular[release_index]
        first = np.searchsorted(self.edges, releases.min(), side='right') - 1
        last = np.searchsorted(self.edges, end, side='left') - 1
        for index in range(first, last + 1):
            left = np.maximum(releases, self.edges[index])
            right = min(end, self.edges[index + 1])
            valid = right > left
            if not valid.any():
                continue
            weight = integrated_velocity_weight(end - right, end - left[valid], self.tau)
            par[valid] += weight[:, None, None] * self.parallel[index]
            per[valid] += weight[:, None, None] * self.perpendicular[index]
        displacement = (_integral_at(end, self.edges, self.velocity) -
                        _integral_at(releases, self.edges, self.velocity))
        maps = np.broadcast_to(np.eye(2), par.shape).copy()
        return maps, displacement, par[:, [0, 0, 1], [0, 1, 1]], per[:, [0, 0, 1], [0, 1, 1]]


def configure_correlated(data, height_m, tau_s, **kwargs):
    """Fixed-tau response, reusing the validated point kernel and Q interface."""
    if 'transport_history' in kwargs:
        raise ValueError('Correlated-velocity prototype cannot be combined with affine transport')
    edges = np.arange(len(data['concentration']) + 1) * 60.
    history = CorrelatedWindHistory(edges, data['wind_xy'], tau_s)
    result = configure(data, height_m, transport_history=history, **kwargs)
    plan = result[3].plan
    # The point adapter only uses plan.age for the vertical Gaussian variance.
    # Horizontal covariance and translation have already been integrated above.
    plan.age = plan.age.new_tensor(effective_age(plan.age.cpu().numpy(), tau_s))
    return result
