import unittest
import numpy as np
import torch
from ia_correlated_dispersion import configure_correlated
from ia_correlated_response import configure_variable_correlation
from ia_correlated_solver import fit_correlated, correlation_information


class CorrelatedSolverTests(unittest.TestCase):
    def test_known_small_noiseless_source(self):
        data = dict(concentration=np.zeros((3, 5)),
            wind_xy=np.array([[1., .2], [.5, .6], [1.2, -.3]]),
            stations_xy=np.array([[5., 3.], [10., -4.], [20., 6.], [30., -8.], [40., 5.]]),
            station_heights=np.full(5, 2.4))
        truth = np.array([0., 0., .03, .02, 3.])
        m, x, b, r, _ = configure_correlated(data, 2., 3., time_order=16, release_order=16)
        observed = (r(m, 0., x, b, physical_parameters=m.xs.new_tensor(truth[:4])) @
                    m.xs.new_full((5,), .001)).numpy()
        model, xyt, basis, response, _ = configure_variable_correlation(data, 2., time_order=16, release_order=16)
        fit = fit_correlated(model, xyt, basis, response, observed,
            [np.array([.01, -.01, .025, .03, 1.])],
            np.array([-.2, -.2, .001, 0., 0.]), np.array([.2, .2, .3, .3, np.inf]),
            .1, count=1, max_nfev=80)
        np.testing.assert_allclose(fit['theta'], truth, atol=1e-5)
        np.testing.assert_allclose(fit['coefficients'], .001, rtol=1e-5)
        self.assertTrue(fit['converged'])
        self.assertLess(fit['sse'], 1e-12)

    def test_exact_tau_diffusion_compensation_detected(self):
        matrix = np.ones((5, 1))
        derivative = np.zeros((5, 1, 5))
        derivative[:, 0, 2] = np.arange(5)
        derivative[:, 0, 4] = -2 * np.arange(5)
        info = correlation_information(matrix, derivative, np.array([1.]), np.array([0]))
        self.assertLess(info['tau_remaining_fraction'], 1e-12)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
