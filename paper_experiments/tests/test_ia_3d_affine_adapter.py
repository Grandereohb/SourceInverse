import unittest
import numpy as np
import torch
from ia_3d_affine_adapter import configure_3d_affine
from ia_metec_point_adapter import configure
from paper_sqrt_affine_wind import SquareRootAffineWindHistory


class ThreeDimensionalAdapterTests(unittest.TestCase):
    def data(self):
        return dict(concentration=np.array([[0., np.nan], [0., 0.]]),
            wind_xy=np.array([[1., .2], [.5, .6]]),
            stations_xy=np.array([[10., 0.], [20., 4.]]), station_heights=np.array([2.4, 3.]))

    def evaluate(self, device):
        data = self.data()
        velocity = data['wind_xy']
        gradient = np.array([[[.003, .001], [-.001, .002]], [[-.001, .002], [.001, -.002]]])
        m, x, b, r, _ = configure_3d_affine(data, 2., velocity, gradient, device=device)
        theta = m.xs.new_tensor([.01, -.02, .03, .02])
        value, derivative = r.matrix_and_jacobian(m, 0., x, b, theta)
        torch.testing.assert_close(value, r(m, 0., x, b, physical_parameters=theta), rtol=1e-10, atol=1e-10)
        for j in range(4):
            step = torch.zeros_like(theta)
            step[j] = 1e-6
            plus = r(m, 0., x, b, physical_parameters=theta + step)
            minus = r(m, 0., x, b, physical_parameters=theta - step)
            torch.testing.assert_close(derivative[..., j], (plus - minus) / 2e-6, rtol=1e-6, atol=1e-7)
        q1 = value.new_tensor([.001, .002, .003, .002, .001])
        q2 = q1 * 2
        torch.testing.assert_close(value @ (q1 + q2), value @ q1 + value @ q2)
        self.assertTrue(bool(torch.isfinite(derivative).all()))
        return value.cpu()

    def test_cpu_q_linearity_and_full_derivatives(self):
        self.evaluate('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_consistency(self):
        torch.testing.assert_close(self.evaluate('cuda'), self.evaluate('cpu'), rtol=1e-10, atol=1e-10)

    def test_traceless_history_recovers_previous_operator(self):
        data = self.data()
        gradients = np.array([[[.003, .001], [-.001, -.003]], [[-.001, .002], [.001, .001]]])
        current = configure_3d_affine(data, 2., data['wind_xy'], gradients)
        history = SquareRootAffineWindHistory([0., 60., 120.], data['wind_xy'], gradients, data['wind_xy'])
        old = configure(data, 2., transport_history=history)
        theta = current[0].xs.new_tensor([.01, -.02, .03, .02])
        def calculate(item):
            m, x, b, r, _ = item
            return r.matrix_and_jacobian(m, 0., x, b, theta)
        for actual, expected in zip(calculate(current), calculate(old)):
            torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-9)

    def test_minute_batches_preserve_full_horizontal_and_vertical_history(self):
        data = self.data()
        gradients = np.array([[[.003, .001], [-.001, .002]], [[-.001, .002], [.001, -.002]]])
        full = configure_3d_affine(data, 2., data['wind_xy'], gradients)
        theta = full[0].xs.new_tensor([.01, -.02, .03, .02])
        m, x, b, r, _ = full
        matrix, derivative = r.matrix_and_jacobian(m, 0., x, b, theta)
        parts, jacobians = [], []
        for minute in [0, 1]:
            m, x, b, r, _ = configure_3d_affine(data, 2., data['wind_xy'], gradients,
                observation_minutes=np.array([minute]))
            value, jacobian = r.matrix_and_jacobian(m, 0., x, b, theta)
            parts.append(value)
            jacobians.append(jacobian)
        torch.testing.assert_close(torch.cat(parts), matrix, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(torch.cat(jacobians), derivative, rtol=1e-10, atol=1e-10)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
