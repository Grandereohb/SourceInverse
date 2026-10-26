import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_reference_transport import puff_response
from paper_baseline_demos import conditional_nnls


class IndependentPuffReferenceTests(unittest.TestCase):
    def test_zero_wind_isotropic_constant_source_closed_form(self):
        times = np.array([0., 0.2, 1., 2.])
        d, sigma = 0.03, 0.15
        response = puff_response([[0., 0.]], times, [0., 1., 2.], [0., 0.], d, d,
                                 speed=0., decay=0., sigma=sigma, order=64)
        expected = np.log1p(2*d*times/sigma**2)/(4*np.pi*d)
        np.testing.assert_allclose(response.sum(2).ravel(), expected, rtol=1e-11, atol=1e-12)
        self.assertTrue((response >= 0).all())

    def test_veering_anisotropic_quadrature_converges_and_is_linear(self):
        kwargs = dict(stations=[[0.5, 0.2], [0.8, -0.2]], times=[0., 1., 3., 6.],
                      knots=[0., 1.5, 3., 4.5, 6.], source=[-0.3, -0.1],
                      d_parallel=0.08, d_perp=0.03, wind='veering')
        response = puff_response(**kwargs, order=32)
        fine = puff_response(**kwargs, order=64)
        np.testing.assert_allclose(response, fine, rtol=1e-8, atol=1e-10)
        a, b = np.array([0., 1., 2., 1., 0.]), np.array([1., 0., 0.5, 0., 1.])
        np.testing.assert_allclose(response@(a+b), response@a+response@b, atol=1e-12)

    def test_invalid_geometry_and_diffusion_rejected(self):
        with self.assertRaises(ValueError):
            puff_response([[0., 0.]], [1.], [0., 2.], [0., 0.], 0.01, 0.03)

    def test_conditional_source_coefficients_recovered_at_known_geometry(self):
        response = puff_response([[0.3, 0.], [0.8, 0.2]], [0.5, 1., 2., 3., 4., 5., 6.],
            [0., 1.5, 3., 4.5, 6.], [-0.3, -0.1], 0.08, 0.03).reshape(-1, 5)
        coefficients = np.array([0., 1., 0.3, 2., 0.])
        fitted, prediction, mse = conditional_nnls(response, response@coefficients)
        np.testing.assert_allclose(fitted, coefficients, rtol=1e-10, atol=1e-10)
        self.assertLess(mse, 1e-20)


if __name__ == '__main__':
    unittest.main()
