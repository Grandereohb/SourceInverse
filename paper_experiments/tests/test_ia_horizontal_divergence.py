import unittest
import numpy as np
from ia_horizontal_divergence import fit_horizontal_affine, incompressible_3d_gradient


class HorizontalDivergenceTests(unittest.TestCase):
    def test_known_affine_and_3d_trace(self):
        xy = np.array([[0., 0.], [100., 10.], [20., 80.]])
        a = np.array([[[.01, -.02], [.03, .005]], [[0., .04], [-.02, -.01]]])
        b = np.array([[2., 1.], [1., -3.]])
        wind = b[:, None] + np.einsum('tij,sj->tsi', a, xy)
        fit = fit_horizontal_affine(xy, wind)
        np.testing.assert_allclose(fit['gradient'], a, atol=1e-15)
        np.testing.assert_allclose(fit['intercept'], b, atol=1e-14)
        np.testing.assert_allclose(fit['divergence'], [.015, -.01], atol=1e-15)
        np.testing.assert_allclose(np.trace(incompressible_3d_gradient(a), axis1=-2, axis2=-1), 0.)
        self.assertEqual(fit['residual_degrees_of_freedom_per_time'], 0)

    def test_frame_and_unit_consistency(self):
        xy = np.array([[0., 0.], [100., 10.], [20., 80.]])
        wind = np.array([[[1., 2.], [2., 0.], [3., 1.]]])
        rotation = np.array([[.6, -.8], [.8, .6]])
        a = fit_horizontal_affine(xy, wind)
        transformed = fit_horizontal_affine(xy @ rotation.T + [500., -200.], wind @ rotation.T)
        np.testing.assert_allclose(transformed['gradient'], rotation @ a['gradient'] @ rotation.T, atol=1e-15)
        np.testing.assert_allclose(transformed['divergence'], a['divergence'], atol=1e-15)
        scaled = fit_horizontal_affine(xy / 1000., wind)
        np.testing.assert_allclose(scaled['gradient'], a['gradient'] * 1000, atol=1e-12)
        self.assertAlmostEqual(a['design_condition'], scaled['design_condition'])

    def test_collinear_rejected(self):
        with self.assertRaises(ValueError):
            fit_horizontal_affine([[0, 0], [1, 0], [2, 0]], np.ones((1, 3, 2)))


if __name__ == '__main__':
    unittest.main()
