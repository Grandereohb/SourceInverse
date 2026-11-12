import unittest
import numpy as np
import torch
from scipy.integrate import quad
from scipy.linalg import expm
from ia_correlated_dispersion import (effective_age, integrated_velocity_weight,
    CorrelatedWindHistory, configure_correlated)
from ia_metec_point_adapter import configure


class CorrelatedDispersionTests(unittest.TestCase):
    def test_ballistic_and_diffusive_limits(self):
        tau = 5.
        young = np.array([1e-10, 1e-6, 1e-3])
        np.testing.assert_allclose(effective_age(young, tau), young ** 2 / (2 * tau), rtol=1e-4)
        old = np.array([1e6])
        np.testing.assert_allclose(effective_age(old, tau), old - tau, atol=1e-9)
        np.testing.assert_array_equal(effective_age(young, 0), young)
        self.assertAlmostEqual(float(effective_age(2., 3.)), float(effective_age(np.array([2.]), 3.)[0]))

    def test_interval_integrals_against_quadrature(self):
        for tau in [1e-6, .1, 10., 1e6]:
            for low, high in [(0., 1e-8), (0., 60.), (1., 60.), (1000., 1000.001)]:
                actual = integrated_velocity_weight(np.array([low]), np.array([high]), tau)[0]
                points = [value for value in [tau, 10 * tau, 100 * tau] if low < value < high]
                expected = quad(lambda x: (-np.expm1(-x / tau)) ** 2, low, high,
                                epsabs=1e-24, epsrel=1e-11, points=points)[0]
                np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-22)

    def test_constant_direction_covariance(self):
        history = CorrelatedWindHistory([0, 30, 60], [[2, 0], [2, 0]], 3.)
        releases = np.array([0., 15., 55., 59.9999])
        _, displacement, par, per = history.relative(60., releases)
        effective = effective_age(60. - releases, 3.)
        np.testing.assert_allclose(par, effective[:, None] * [[1., 0, 0]], atol=1e-13)
        np.testing.assert_allclose(per, effective[:, None] * [[0., 0, 1]], atol=1e-13)
        np.testing.assert_allclose(displacement[:, 0], 2 * (60 - releases))

    def test_turning_covariance_positive_and_rotational(self):
        edges = [0, 30, 60]
        wind = np.array([[2., 0], [0., 3.]])
        rotation = np.array([[.6, -.8], [.8, .6]])
        releases = np.array([0., 10., 31., 59.])
        def matrix(v):
            return np.stack([np.stack([v[:, 0], v[:, 1]], -1),
                             np.stack([v[:, 1], v[:, 2]], -1)], -2)
        _, shift, p, q = CorrelatedWindHistory(edges, wind, 5.).relative(60., releases)
        _, rotated_shift, rp, rq = CorrelatedWindHistory(edges, wind @ rotation.T, 5.).relative(60., releases)
        for original, rotated in [(p, rp), (q, rq)]:
            np.testing.assert_allclose(matrix(rotated), rotation @ matrix(original) @ rotation.T, atol=1e-13)
            self.assertTrue(np.all(np.linalg.eigvalsh(matrix(original)) >= -1e-14))
        np.testing.assert_allclose(rotated_shift, shift @ rotation.T, atol=1e-13)
        np.testing.assert_allclose(matrix(p + q), effective_age(60 - releases, 5.)[:, None, None] * np.eye(2), atol=1e-13)

    def test_reflected_gaussian_mass(self):
        for tau in [0., 1., 30.]:
            variance = 2 * 2. * effective_age(np.array([4.]), tau)[0]
            vertical = lambda z: (np.exp(-(z - 2.) ** 2 / (2 * variance)) +
                                  np.exp(-(z + 2.) ** 2 / (2 * variance))) / np.sqrt(2 * np.pi * variance)
            self.assertAlmostEqual(quad(vertical, 0, np.inf)[0], 1., places=10)

    def test_turning_covariance_against_state_space_exponential(self):
        # Independent 4-state (X,Y,Vx,Vy) Lyapunov propagation, not the age formula.
        wind = np.array([[2., 1.], [-1., 3.]])
        for tau in [.2, 5., 100.]:
            history = CorrelatedWindHistory([0., 2., 5.], wind, tau)
            covariance = np.zeros((4, 4))
            k = 7 * history.parallel + 2 * history.perpendicular
            covariance[2:, 2:] = k[0] / tau
            drift = np.block([[np.zeros((2, 2)), np.eye(2)],
                              [np.zeros((2, 2)), -np.eye(2) / tau]])
            for duration, tensor in zip([2., 3.], k):
                noise = np.zeros((4, 4))
                noise[2:, 2:] = 2 * tensor / tau ** 2
                augmented = np.block([[drift, noise], [np.zeros((4, 4)), -drift.T]])
                # Keep the reference's unstable Van Loan block from exp(t/tau) overflow/cancellation.
                steps = max(1, int(np.ceil(duration / tau)))
                propagator = expm(augmented * (duration / steps))
                f = propagator[:4, :4]
                for _ in range(steps):
                    covariance = f @ covariance @ f.T + propagator[:4, 4:] @ f.T
            _, _, p, q = history.relative(5., np.array([0.]))
            packed = (2 * (7 * p + 2 * q))[0]
            actual = np.array([[packed[0], packed[1]], [packed[1], packed[2]]])
            np.testing.assert_allclose(actual, covariance[:2, :2], rtol=1e-8, atol=1e-10)

    def evaluate(self, device):
        data = dict(concentration=np.zeros((2, 2)), wind_xy=np.array([[1., .2], [.5, .6]]),
                    stations_xy=np.array([[10., 0.], [20., 4.]]), station_heights=np.array([2.4, 3.]))
        model, xyt, basis, response, _ = configure_correlated(data, 2., 3., device=device)
        theta = model.xs.new_tensor([.01, -.02, .03, .02])
        value, derivative = response.matrix_and_jacobian(model, 0., xyt, basis, theta)
        for j in range(4):
            step = torch.zeros_like(theta)
            step[j] = 1e-6
            plus = response.matrix_and_jacobian(model, 0., xyt, basis, theta + step)[0]
            minus = response.matrix_and_jacobian(model, 0., xyt, basis, theta - step)[0]
            torch.testing.assert_close(derivative[..., j], (plus - minus) / 2e-6, rtol=1e-6, atol=1e-7)
        q1 = value.new_tensor([.001, .002, .003, .002, .001])
        q2 = q1.flip(0) * 2
        torch.testing.assert_close(value @ (q1 + q2), value @ q1 + value @ q2)
        zero = configure_correlated(data, 2., 0., device=device)
        original = configure(data, 2., device=device)
        def response_at(item):
            m, x, b, r, _ = item
            return r(m, 0., x, b, physical_parameters=theta)
        torch.testing.assert_close(response_at(zero), response_at(original), rtol=1e-10, atol=1e-10)
        self.assertTrue(bool(torch.isfinite(value).all() and torch.isfinite(derivative).all()))
        return value.cpu()

    def test_response_derivatives_q_linearity_zero_tau_limit(self):
        self.evaluate('cpu')

    def test_minute_average_against_independent_age_integral(self):
        data = dict(concentration=np.zeros((1, 1)), wind_xy=np.array([[1., 0.]]),
                    stations_xy=np.array([[20., 0.]]), station_heights=np.array([2.4]))
        model, xyt, basis, response, _ = configure_correlated(data, 2., 3.,
            time_order=32, release_order=32, device='cpu')
        theta = model.xs.new_tensor([0., 0., .03, 0.])  # D=5m2/s, Q=.001kg/s.
        actual = float((response(model, 0., xyt, basis, physical_parameters=theta) @
                        model.xs.new_full((5,), .001))[0])
        def integrand(age):
            variance = 10 * float(effective_age(age, 3.))
            if variance == 0:
                return 0.
            horizontal = np.exp(-(20 - age) ** 2 / (2 * variance)) / (2 * np.pi * variance)
            vertical = (np.exp(-.4 ** 2 / (2 * variance)) +
                        np.exp(-4.4 ** 2 / (2 * variance))) / np.sqrt(2 * np.pi * variance)
            return (60 - age) / 60 * horizontal * vertical * .001 * 1.524e6
        expected = quad(integrand, 0., 60., epsabs=1e-10, epsrel=1e-10)[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-6)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_consistency(self):
        torch.testing.assert_close(self.evaluate('cuda'), self.evaluate('cpu'), rtol=1e-10, atol=1e-10)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
