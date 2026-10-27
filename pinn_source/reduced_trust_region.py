"""Bound-constrained variable projection with an active-set residual Jacobian.

The response operator accepts physical coordinates and optionally returns dA/dtheta.
Only Q coefficients are eliminated; all supplied observations enter the objective.
"""

import time

import numpy as np
from scipy.optimize import least_squares, lsq_linear


class ReducedResidual:
    def __init__(self, response, observations, q_upper_bound=np.inf, weights=None,
                 rcond=1e-12):
        self.response = response
        self.observed = np.asarray(observations, dtype=float).reshape(-1)
        self.weights = (np.ones_like(self.observed) if weights is None
                        else np.asarray(weights, dtype=float).reshape(-1))
        if (not self.observed.size or self.weights.shape != self.observed.shape
                or not np.isfinite(self.observed).all()
                or not np.isfinite(self.weights).all() or np.any(self.weights < 0)
                or not np.any(self.weights > 0)):
            raise ValueError('Finite observations and nonnegative nonzero weights required')
        self.upper = float(q_upper_bound)
        if np.isnan(self.upper) or self.upper <= 0 or not 0 < rcond < 1:
            raise ValueError('Positive Q upper bound and valid SVD cutoff required')
        self.rcond = float(rcond)
        self.response_calls = self.derivative_calls = self.inner_solves = 0
        self.theta = self.matrix = self.solution = self.residual_value = None
        self.cached_jacobian = None
        self.best = None
        self.trace = []

    def _matrix(self, matrix):
        matrix = np.asarray(matrix, dtype=float)
        if (matrix.ndim != 2 or matrix.shape[0] != self.observed.size
                or not matrix.shape[1] or not np.isfinite(matrix).all()):
            raise ValueError('Finite response matrix matching observations required')
        return matrix

    def residual(self, theta):
        theta = np.asarray(theta, dtype=float)
        if theta.ndim != 1 or not np.isfinite(theta).all():
            raise ValueError('Finite physical parameter vector required')
        if self.theta is not None and np.array_equal(theta, self.theta):
            return self.residual_value.copy()
        matrix = self._matrix(self.response(theta, False))
        self.response_calls += 1
        weighted = self.weights[:, None]*matrix
        scale = np.linalg.norm(weighted, axis=0)
        scale[scale == 0] = 1.
        normalized = weighted/scale
        # Column equilibration changes only coordinates, including the Q bounds.
        solution = lsq_linear(normalized, self.weights*self.observed,
                              bounds=(np.zeros(len(scale)), self.upper*scale),
                              method='bvls', tol=1e-12, max_iter=500)
        self.inner_solves += 1
        if not solution.success:
            raise RuntimeError('Inner bounded Q solve failed: '+str(solution.message))
        coefficients = solution.x/scale
        prediction = matrix@coefficients
        residual = self.weights*(prediction-self.observed)
        self.theta, self.matrix = theta.copy(), matrix
        self.solution = dict(coefficients=coefficients, prediction=prediction,
                             active_mask=solution.active_mask.copy(), column_scale=scale,
                             inner_optimality=float(solution.optimality),
                             response_rank=int(np.linalg.matrix_rank(weighted)),
                             response_condition=float(np.linalg.cond(weighted)))
        self.residual_value, self.cached_jacobian = residual, None
        sse = float(residual@residual)
        self.trace.append(dict(theta=theta.tolist(), sse=sse,
                               active_mask=solution.active_mask.tolist()))
        if self.best is None or sse < self.best['sse']:
            self.best = dict(theta=theta.copy(), sse=sse,
                             **{k: v.copy() if isinstance(v, np.ndarray) else v
                                for k, v in self.solution.items()})
        return residual.copy()

    def jacobian(self, theta):
        self.residual(theta)
        if self.cached_jacobian is not None:
            return self.cached_jacobian.copy()
        matrix, derivative = self.response(np.asarray(theta, dtype=float), True)
        self.response_calls += 1
        self.derivative_calls += 1
        matrix = self._matrix(matrix)
        np.testing.assert_allclose(matrix, self.matrix, rtol=1e-8, atol=1e-12)
        derivative = np.asarray(derivative, dtype=float)
        if derivative.shape != (*matrix.shape, len(theta)) or not np.isfinite(derivative).all():
            raise ValueError('Response Jacobian must have shape observations x Q x theta')
        derivative = derivative*self.weights[:, None, None]
        a = self.solution['coefficients']
        free = self.solution['active_mask'] == 0
        jacobian = np.einsum('mqk,q->mk', derivative, a)
        if np.any(free):
            scale = self.solution['column_scale'][free]
            af = self.weights[:, None]*matrix[:, free]/scale
            u, singular, vt = np.linalg.svd(af, full_matrices=False)
            valid = singular > self.rcond*singular[0] if singular.size and singular[0] > 0 else np.zeros_like(singular, dtype=bool)
            inverse = (vt[valid].T/singular[valid])@u[:, valid].T
            # Both terms are needed: projecting dA*a alone is only an approximation.
            jacobian -= af@(inverse@jacobian)
            da_transpose_r = np.einsum('mqk,m->qk',
                derivative[:, free, :]/scale[None, :, None], self.residual_value)
            jacobian -= inverse.T@da_transpose_r
        self.cached_jacobian = jacobian
        return jacobian.copy()


def fit_reduced_trust_region(response, observations, starts, lower, upper, *,
                             q_upper_bound=np.inf, weights=None, max_nfev=250,
                             coordinate_scale=None):
    """Scaled TRF on physical theta, re-solving bounded Q at every trial point."""
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    starts = np.asarray(starts, dtype=float)
    if (lower.ndim != 1 or starts.ndim != 2 or starts.shape[1:] != lower.shape
            or not len(starts) or upper.shape != lower.shape or np.any(lower >= upper)
            or np.isnan(lower).any() or np.isnan(upper).any()
            or not np.isfinite(starts).all() or np.any(starts < lower) or np.any(starts > upper)
            or max_nfev < 1):
        raise ValueError('Matching physical bounds, feasible starts and positive budget required')
    scale = (np.maximum(np.max(np.abs(starts), axis=0), 1e-3) if coordinate_scale is None
             else np.asarray(coordinate_scale, dtype=float))
    if scale.shape != lower.shape or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError('Positive finite physical coordinate scales required')
    if coordinate_scale is None:
        scale[:min(2, len(scale))] = 1.
    objective = ReducedResidual(response, observations, q_upper_bound, weights)
    begun = time.perf_counter()
    runs = []
    for start in starts:
        fit = least_squares(objective.residual, start, jac=objective.jacobian,
            bounds=(lower, upper), method='trf', loss='linear', x_scale=scale,
            max_nfev=int(max_nfev), ftol=1e-9, xtol=1e-9, gtol=1e-9)
        runs.append(dict(success=bool(fit.success), status=int(fit.status),
                         message=str(fit.message), nfev=int(fit.nfev),
                         njev=int(fit.njev or 0), optimality=float(fit.optimality),
                         sse=float(fit.fun@fit.fun)))
    result = dict(objective.best)
    result.update(seconds=time.perf_counter()-begun, optimizer_runs=runs,
        coordinate_scale=scale.tolist(), trace=objective.trace,
        response_matrix_calls=objective.response_calls, derivative_calls=objective.derivative_calls,
        inner_solve_count=objective.inner_solves,
        converged=bool(runs[int(np.argmin([r['sse'] for r in runs]))]['success']))
    return result
