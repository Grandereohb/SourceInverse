import unittest
import numpy as np
from scipy.linalg import expm
from scipy.integrate import quad
from ia_3d_affine_transport import horizontal_flow, general_step, ThreeDimensionalAffineHistory
from paper_sqrt_affine_wind import exact_step_factors


class ThreeDimensionalAffineTests(unittest.TestCase):
    def test_flow_translation_and_volume(self):
        a = np.array([[.01, .02], [-.01, .005]])
        b = np.array([2., -1.])
        for t in [0., .001, 10., 60.]:
            flow, shift, _, _, z, _ = general_step(a, b, np.array([t]), np.array([1., 0.]))
            augmented = np.zeros((3, 3))
            augmented[:2, :2], augmented[:2, 2] = a, b
            reference = expm(augmented * t)
            np.testing.assert_allclose(flow[0], reference[:2, :2], atol=1e-12)
            np.testing.assert_allclose(shift[0], reference[:2, 2], atol=1e-11)
            self.assertAlmostEqual(np.linalg.det(flow[0]) * z[0], 1., places=12)

    def test_traceless_limit(self):
        a = np.array([[.01, .02], [-.01, -.01]])
        b = np.array([2., -1.])
        times = np.array([.01, 10., 60.])
        direction = np.array([.6, .8])
        actual = general_step(a, b, times, direction)
        reference = exact_step_factors(a, b, times, direction)
        for index in [0, 1]:
            np.testing.assert_allclose(actual[index], reference[index], rtol=1e-11, atol=1e-11)
        for index in [2, 3]:
            np.testing.assert_allclose(actual[index] @ actual[index].swapaxes(-1, -2),
                reference[index] @ reference[index].swapaxes(-1, -2), rtol=1e-10, atol=1e-10)
        np.testing.assert_array_equal(actual[4], np.ones_like(times))
        np.testing.assert_array_equal(actual[5], times)

    def test_isotropic_expansion_vertical_mass(self):
        k = .01
        t = 20.
        flow, _, p, q, z, variance = general_step(np.eye(2) * k, np.zeros(2), np.array([t]), np.array([1., 0.]))
        np.testing.assert_allclose((p @ p.swapaxes(-1, -2) + q @ q.swapaxes(-1, -2))[0],
                                  np.eye(2) * np.expm1(2 * k * t) / (2 * k), atol=1e-11)
        np.testing.assert_allclose(variance, -np.expm1(-4 * k * t) / (4 * k))
        v = 2 * 3. * variance[0]
        h = 2. * z[0]
        density = lambda height: (np.exp(-(height - h) ** 2 / (2 * v)) +
                                  np.exp(-(height + h) ** 2 / (2 * v))) / np.sqrt(2 * np.pi * v)
        self.assertAlmostEqual(quad(density, 0, np.inf)[0], 1., places=12)

    def test_piecewise_order_and_positive_covariance(self):
        gradients = np.array([[[.01, .02], [-.01, .005]], [[-.005, .01], [.02, -.003]]])
        wind = np.array([[2., .5], [1., -1.]])
        history = ThreeDimensionalAffineHistory([0., 20., 40.], wind, gradients, wind)
        releases = np.array([0., 5., 20., 39.])
        maps, shift, p, q, z, variance = history.relative_state(40., releases)
        expected = horizontal_flow(gradients[1], 20.) @ horizontal_flow(gradients[0], 20.)
        np.testing.assert_allclose(maps[0], expected, atol=1e-12)
        np.testing.assert_allclose(np.linalg.det(maps) * z, np.ones(len(releases)), atol=1e-12)
        covariance = p @ p.swapaxes(-1, -2) + q @ q.swapaxes(-1, -2)
        self.assertTrue(np.all(np.linalg.eigvalsh(covariance) > 0))
        self.assertTrue(np.all(variance > 0))
        refined = ThreeDimensionalAffineHistory([0., 20., 40.], wind, gradients, wind, order=32).relative_state(40., releases)
        for a, b in zip([maps, shift, z, variance], [refined[0], refined[1], refined[4], refined[5]]):
            np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
