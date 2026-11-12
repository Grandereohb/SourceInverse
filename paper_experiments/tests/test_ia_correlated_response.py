import unittest
import numpy as np
import torch
from ia_correlated_dispersion import configure_correlated
from ia_correlated_response import configure_variable_correlation


class CorrelatedResponseTests(unittest.TestCase):
    def run_device(self, device):
        data = dict(concentration=np.zeros((2, 2)), wind_xy=np.array([[1., .2], [.5, .6]]),
                    stations_xy=np.array([[10., 0.], [20., 4.]]), station_heights=np.array([2.4, 3.]))
        model, xyt, basis, response, _ = configure_variable_correlation(data, 2., device=device)
        original_arrays = response.original
        for tau in [0., 1e-6, .3, 3., 300.]:
            theta = model.xs.new_tensor([.01, -.02, .03, .02, tau])
            value, derivative = response.matrix_and_jacobian(model, 0., xyt, basis, theta)
            fixed = configure_correlated(data, 2., tau, device=device)
            m, x, b, r, _ = fixed
            reference = r(m, 0., x, b, physical_parameters=theta[:4])
            torch.testing.assert_close(value, reference, rtol=1e-9, atol=1e-9)
            q = value.new_tensor([.001, .002, .003, .004, .005])
            torch.testing.assert_close(value @ (q + q), value @ q + value @ q)
            self.assertTrue(bool(torch.isfinite(derivative).all()))
            self.assertTrue(all(a is b for a, b in zip(original_arrays,
                (response.plan.parallel, response.plan.perpendicular, response.plan.age))))
            if tau >= .3:
                for coordinate in range(5):
                    step = torch.zeros_like(theta)
                    h = 1e-6 if coordinate < 4 else max(1e-4, tau * 1e-5)
                    step[coordinate] = h
                    plus = response(model, 0., xyt, basis, physical_parameters=theta + step)
                    minus = response(model, 0., xyt, basis, physical_parameters=theta - step)
                    torch.testing.assert_close(derivative[..., coordinate], (plus - minus) / (2 * h),
                                               rtol=2e-5, atol=2e-7)
        # Boundary derivative is one-sided, not a central difference through negative tau.
        theta = model.xs.new_tensor([.01, -.02, .03, .02, 0.])
        zero, derivative = response.matrix_and_jacobian(model, 0., xyt, basis, theta)
        theta[4] = 1e-5
        plus = response(model, 0., xyt, basis, physical_parameters=theta)
        torch.testing.assert_close(derivative[..., 4], (plus - zero) / 1e-5, rtol=1e-3, atol=2e-4)
        return zero.cpu()

    def test_cpu_full_jacobian_fixed_prototype_and_zero_boundary(self):
        self.run_device('cpu')

    def test_shared_history_with_late_query_truncation_and_missing_station(self):
        minute = np.arange(35)
        data = dict(concentration=np.zeros((35, 3)),
            wind_xy=np.column_stack((np.cos(minute * .2), np.sin(minute * .2))),
            stations_xy=np.array([[10., 1.], [20., 4.], [-10., 4.]]), station_heights=np.full(3, 2.4))
        data['concentration'][34, 1] = np.nan
        options = dict(device='cpu', time_order=4, release_order=4,
                       history_minutes=20, observation_minutes=np.array([34]))
        m, x, b, response, observed = configure_variable_correlation(data, 2., **options)
        self.assertEqual(len(observed), 2)
        for tau in [.1, 3., 100.]:
            theta = m.xs.new_tensor([.01, -.02, .03, .02, tau])
            actual = response.matrix_and_jacobian(m, 0., x, b, theta)[0]
            fm, fx, fb, fixed, _ = configure_correlated(data, 2., tau, **options)
            expected = fixed(fm, 0., fx, fb, physical_parameters=theta[:4])
            torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-9)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_consistency(self):
        torch.testing.assert_close(self.run_device('cuda'), self.run_device('cpu'), rtol=1e-9, atol=1e-9)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
