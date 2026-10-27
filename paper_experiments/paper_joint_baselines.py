"""Joint TRF and unmodified public gbest PSO on a shared transport response.

No source-strength elimination occurs in either optimizer. Bounds and starts
are inputs, never inferred from evaluation truth. Only fixed transport is cached.
"""

from dataclasses import dataclass
import time

import numpy as np
from scipy.optimize import least_squares
import torch


@dataclass(frozen=True)
class JointBounds:
    lower: np.ndarray
    upper: np.ndarray

    def __post_init__(self):
        lower = np.asarray(self.lower, dtype=float).copy()
        upper = np.asarray(self.upper, dtype=float).copy()
        if (lower.ndim != 1 or len(lower) < 5 or lower.shape != upper.shape
                or not np.isfinite(lower).all() or np.isnan(upper).any()
                or np.any(lower >= upper) or lower[2] <= 0
                or np.any(lower[3:] < 0)):
            raise ValueError('Bounds must describe x,y,D_perp,delta_D,nonnegative coefficients')
        lower.setflags(write=False)
        upper.setflags(write=False)
        object.__setattr__(self, 'lower', lower)
        object.__setattr__(self, 'upper', upper)

    def validate(self, theta):
        theta = np.asarray(theta, dtype=float)
        if (theta.shape != self.lower.shape or not np.isfinite(theta).all()
                or np.any(theta < self.lower) or np.any(theta > self.upper)):
            raise ValueError('Parameter vector is outside the declared shared bounds')
        return theta


def candidate_parameters(candidate):
    return np.r_[candidate.source_x, candidate.source_y,
                 candidate.diffusion_perp_norm,
                 candidate.diffusion_parallel_norm-candidate.diffusion_perp_norm,
                 candidate.coefficients]


class JointResidual:
    """Physical coordinates keep external optimization gradients connected."""

    def __init__(self, response, model, sigma, xyt, basis, observations, bounds):
        self.response, self.model = response, model
        self.sigma, self.xyt, self.basis = sigma, xyt, basis
        self.bounds = bounds
        self.observations = torch.as_tensor(observations, dtype=model.xs.dtype,
                                            device=model.xs.device).reshape(-1)
        if (not bool(torch.isfinite(self.observations).all())
                or len(bounds.lower) != 4+basis.shape[1]):
            raise ValueError('Finite observations and matching Q basis required')
        self.matrix_calls = self.jacobian_calls = self.residual_calls = 0

    def tensor_residual(self, theta):
        matrix = self.response(self.model, self.sigma, self.xyt, self.basis,
                               track_gradients=True, physical_parameters=theta[:4])
        self.matrix_calls += 1
        return matrix @ theta[4:] - self.observations

    def tensor(self, theta):
        return torch.as_tensor(theta, dtype=self.model.xs.dtype,
                               device=self.model.xs.device)

    def residual(self, theta):
        self.bounds.validate(theta)
        self.residual_calls += 1
        with torch.no_grad():
            value = self.tensor_residual(self.tensor(theta))
        result = value.detach().cpu().numpy().astype(float)
        if not np.isfinite(result).all():
            raise FloatingPointError('Nonfinite shared residual')
        return result

    def jacobian(self, theta):
        self.bounds.validate(theta)
        self.jacobian_calls += 1
        with torch.enable_grad():
            jac = torch.autograd.functional.jacobian(
                self.tensor_residual, self.tensor(theta), vectorize=True)
        result = jac.detach().cpu().numpy().astype(float)
        if not np.isfinite(result).all():
            raise FloatingPointError('Nonfinite shared Jacobian')
        return result

    def population_cost(self, population):
        values = np.asarray(population, dtype=float)
        for value in values:
            self.bounds.validate(value)
        if not hasattr(self.response, 'evaluate_population'):
            return np.asarray([np.sum(self.residual(value)**2) for value in values])
        with torch.no_grad():
            theta = self.tensor(values)
            matrices = self.response.evaluate_population(self.model, self.sigma,
                self.xyt, self.basis, theta[:, :4],
                compute_dtype=getattr(self, 'population_dtype', None),
                batch_size=getattr(self, 'population_batch_size', 4),
                max_kernel_elements=getattr(self, 'population_kernel_elements', 2_000_000))
            prediction = torch.einsum('pmq,pq->pm', matrices, theta[:, 4:].to(matrices.dtype))
            residual = prediction.to(self.observations.dtype)-self.observations
            cost = torch.sum(residual**2, dim=1).cpu().numpy()
        self.matrix_calls += len(values)
        self.residual_calls += len(values)
        if not np.isfinite(cost).all():
            raise FloatingPointError('Nonfinite population objective')
        return cost

    def counts(self):
        return dict(response_matrix_calls=self.matrix_calls,
                    jacobian_calls=self.jacobian_calls,
                    residual_calls=self.residual_calls)

    def result(self, theta, seconds, **metadata):
        residual = self.residual(theta)
        prediction = residual+self.observations.detach().cpu().numpy()
        return dict(source_norm=np.asarray(theta[:2]).tolist(),
                    diffusion_norm=[float(theta[2]+theta[3]), float(theta[2])],
                    coefficients=np.asarray(theta[4:]).tolist(),
                    prediction=prediction.tolist(), mse=float(np.mean(residual**2)),
                    sse=float(residual@residual), seconds=float(seconds),
                    counters=self.counts(), **metadata)


def fit_joint_nls(objective, starts, max_nfev=1000):
    """Bounded joint least squares, with no NNLS inside the iterations."""
    if not starts or max_nfev < 1:
        raise ValueError('At least one start and a positive budget required')
    started = time.perf_counter()
    best = None
    runs = []
    # Fixed coordinate scaling is shared across starts, not changed using truth.
    scale = np.maximum(np.max(np.abs(np.asarray(starts)), axis=0), 1e-6)
    scale[:2] = 1.
    scale[4:] = max(float(np.max(np.abs(np.asarray(starts)[:, 4:]))), 1e-6)
    for initial in starts:
        initial = objective.bounds.validate(initial)
        start_counts = objective.counts()
        fit = least_squares(objective.residual, initial, jac=objective.jacobian,
                            method='trf', loss='linear', x_scale=scale,
                            bounds=(objective.bounds.lower, objective.bounds.upper),
                            max_nfev=max_nfev, ftol=1e-8, xtol=1e-8, gtol=1e-8)
        score = float(fit.fun@fit.fun)
        runs.append(dict(success=bool(fit.success), status=int(fit.status),
                         message=str(fit.message), nfev=int(fit.nfev),
                         njev=int(fit.njev or 0), sse=score,
                         work={k: v-start_counts[k] for k, v in objective.counts().items()}))
        if best is None or score < best[0]:
            best = score, fit.x.copy()
    return objective.result(best[1], time.perf_counter()-started,
                            method='joint_nls_trf', optimizer_runs=runs,
                            converged=bool(runs[int(np.argmin([r['sse'] for r in runs]))]['success']),
                            coordinate_scale=scale.tolist())


def fit_standard_pso(objective, *, seed, particles, iterations, options=None, starts=None,
                     coordinate_scales=None, local_fraction=0., velocity_fraction=None,
                     initial_spread=.5, zero_initial_velocity=False,
                     use_population_evaluation=False):
    """Call PySwarms' fixed-coefficient GlobalBestPSO; no local refinement."""
    if not np.isfinite(objective.bounds.upper).all():
        raise ValueError('PSO requires an explicitly declared finite shared parameter box')
    if particles < 2 or iterations < 1:
        raise ValueError('PSO needs at least two particles and one iteration')
    from pyswarms.single import GlobalBestPSO

    options = dict(c1=1.49618, c2=1.49618, w=.7298) if options is None else dict(options)
    if set(options) != {'c1', 'c2', 'w'} or not all(np.isfinite(list(options.values()))):
        raise ValueError('Declare fixed c1,c2,w; adaptive schedules are not used')
    started = time.perf_counter()
    span = objective.bounds.upper-objective.bounds.lower
    scales = None if coordinate_scales is None else np.asarray(coordinate_scales, dtype=float)
    if scales is not None and (scales.shape != span.shape or not np.isfinite(scales).all()
                               or np.any(scales <= 0)):
        raise ValueError('Positive finite PSO coordinate scales required')
    if not 0 <= local_fraction <= 1 or (local_fraction and (scales is None or not starts)):
        raise ValueError('Local initialization requires scales, common starts and a valid fraction')
    if not np.isfinite(initial_spread) or initial_spread <= 0:
        raise ValueError('Positive finite initial spread required')
    if velocity_fraction is not None and (scales is None or not np.isfinite(velocity_fraction)
                                         or velocity_fraction <= 0):
        raise ValueError('Velocity limits require scales and a positive fraction')

    def population_cost(population):
        if use_population_evaluation:
            return objective.population_cost(objective.bounds.lower+population*span)
        costs = []
        for normalized in population:
            residual = objective.residual(objective.bounds.lower+normalized*span)
            costs.append(float(residual@residual))
        return np.asarray(costs)

    # The public implementation uses legacy global numpy RNG; restore caller state.
    rng_state = np.random.get_state()
    try:
        np.random.seed(seed)
        initial = None
        if starts is not None:
            if len(starts) > particles:
                raise ValueError('More common starts than PSO particles')
            initial = np.random.uniform(size=(particles, len(span)))
            for i, value in enumerate(starts):
                initial[i] = (objective.bounds.validate(value)-objective.bounds.lower)/span
            if local_fraction:
                local_count = max(len(starts), int(particles*local_fraction))
                for i in range(len(starts), local_count):
                    center = np.asarray(starts[i % len(starts)])
                    value = center+np.random.normal(size=len(span))*initial_spread*scales
                    initial[i] = np.clip((value-objective.bounds.lower)/span, 0., 1.)
        velocity_clamp = None
        if velocity_fraction is not None:
            maximum = np.minimum(float(velocity_fraction)*scales/span, .2)
            velocity_clamp = (-maximum, maximum)
        optimizer = GlobalBestPSO(n_particles=particles, dimensions=len(span),
            options=options, bounds=(np.zeros(len(span)), np.ones(len(span))),
            bh_strategy='reflective', init_pos=initial, velocity_clamp=velocity_clamp)
        if velocity_clamp is not None and starts:
            optimizer.swarm.velocity[:len(starts)] = 0.
        if zero_initial_velocity:
            optimizer.swarm.velocity[:] = 0.
        score, position = optimizer.optimize(population_cost, iters=iterations, verbose=False)
    finally:
        np.random.set_state(rng_state)
    return objective.result(objective.bounds.lower+position*span,
        time.perf_counter()-started, method='standard_gbest_pso', seed=int(seed),
        particles=int(particles), iterations=int(iterations), options=options,
        optimizer_sse=float(score), history=list(map(float, optimizer.cost_history)),
        coordinate_system='unit box, affine mapping of the declared shared bounds',
        common_initial_particles=0 if starts is None else len(starts),
        local_initial_fraction=float(local_fraction),
        physical_coordinate_scales=None if scales is None else scales.tolist(),
        velocity_fraction=velocity_fraction,
        initial_spread=float(initial_spread), zero_initial_velocity=bool(zero_initial_velocity),
        population_evaluation=bool(use_population_evaluation),
        population_dtype=str(getattr(objective, 'population_dtype', 'native')),
        implementation='pyswarms.single.GlobalBestPSO', convergence='fixed budget, not certified')
