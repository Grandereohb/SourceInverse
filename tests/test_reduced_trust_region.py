import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'pinn_source'))
from reduced_trust_region import ReducedResidual, fit_reduced_trust_region


def response(theta, jacobian):
    t = np.linspace(-1., 1., 21)
    matrix = np.column_stack((np.exp(theta[0]*t), 1+theta[1]*t*t))
    if not jacobian:
        return matrix
    derivative = np.zeros((*matrix.shape, 2))
    derivative[:, 0, 0] = t*matrix[:, 0]
    derivative[:, 1, 1] = t*t
    return matrix, derivative


class ReducedTrustRegionTests(unittest.TestCase):
    def test_full_residual_jacobian_with_nonzero_residual_and_weights(self):
        theta = np.array([.7, .4])
        observations = response(theta, False)@np.array([.8, .5])+ .03*np.cos(np.arange(21))
        obj = ReducedResidual(response, observations, weights=np.linspace(.5, 2., 21))
        jacobian = obj.jacobian(theta)
        self.assertGreater(np.linalg.norm(obj.residual(theta)), .01)
        for k in range(2):
            shift = np.eye(2)[k]*1e-6
            numerical = (obj.residual(theta+shift)-obj.residual(theta-shift))/2e-6
            np.testing.assert_allclose(jacobian[:, k], numerical, atol=1e-8, rtol=1e-5)

    def test_capped_coefficients_are_not_differentiated_as_free(self):
        theta = np.array([.7, .4])
        observed = response(theta, False)@np.array([3., .4])
        obj = ReducedResidual(response, observed, q_upper_bound=1.)
        jacobian = obj.jacobian(theta)
        self.assertTrue(np.any(obj.solution['active_mask'] != 0))
        for k in range(2):
            shift = np.eye(2)[k]*1e-6
            numerical = (obj.residual(theta+shift)-obj.residual(theta-shift))/2e-6
            np.testing.assert_allclose(jacobian[:, k], numerical, atol=1e-7, rtol=1e-5)

    def test_eliminated_q_recovery_and_decreasing_profile_objective(self):
        true = np.array([.7, .4])
        observed = response(true, False)@np.array([.8, .5])
        fit = fit_reduced_trust_region(response, observed, [[.5, .2]], [.1, .01], [2., 2.])
        self.assertLess(fit['sse'], 1e-12)
        np.testing.assert_allclose(fit['theta'], true, atol=1e-5)
        np.testing.assert_allclose(fit['coefficients'], [.8, .5], atol=1e-5)
        self.assertLess(fit['sse'], fit['trace'][0]['sse'])

    def test_column_equilibration_preserves_predictions(self):
        theta = np.array([.7, .4])
        observed = response(theta, False)@np.array([.8, .5])
        first = ReducedResidual(response, observed)
        second = ReducedResidual(lambda p, j: response(p, False)*[1e-8, 1e5], observed)
        np.testing.assert_allclose(first.residual(theta), second.residual(theta), atol=1e-12)

    def test_invalid_bounds_rejected(self):
        with self.assertRaises(ValueError):
            fit_reduced_trust_region(response, np.ones(21), [[2., 2.]], [0., 0.], [1., 1.])


if __name__ == '__main__':
    unittest.main()
