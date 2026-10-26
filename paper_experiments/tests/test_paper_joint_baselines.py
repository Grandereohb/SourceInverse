from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_joint_baselines import JointBounds, JointResidual, fit_joint_nls, fit_standard_pso
from paper_fast_parcels import CachedMeanResponse
import run_paper_parcel_pilot as parcel


class JointBaselineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_bounds_do_not_silently_clip(self):
        bounds = JointBounds(np.array([-1, -1, .001, 0, 0]), np.ones(5))
        with self.assertRaises(ValueError):
            bounds.validate([0, 0, .01, 0, -1e-5])
        with self.assertRaises(ValueError):
            JointBounds(np.zeros(5), np.ones(5))

    def test_joint_nls_does_not_eliminate_coefficients(self):
        model = SimpleNamespace(xs=torch.tensor(0., dtype=torch.float64))
        matrix = torch.eye(5, dtype=torch.float64)
        def response(*args, **kwargs):
            p = kwargs['physical_parameters']
            return matrix*(1+p[0]**2+p[1]**2+p[2]+p[3])
        bounds = JointBounds(np.r_[[-1, -1, .001, 0], np.zeros(5)],
                             np.r_[[1, 1, 1, 1], np.full(5, 10)])
        obj = JointResidual(response, model, 0, None, torch.eye(5), np.arange(1, 6), bounds)
        fit = fit_joint_nls(obj, [np.r_[[0, 0, .02, .01], np.ones(5)]], max_nfev=100)
        self.assertLess(fit['mse'], 1e-12)
        self.assertGreater(fit['counters']['jacobian_calls'], 0)
        self.assertEqual(len(fit['coefficients']), 5)

    def test_external_coordinates_match_existing_response_and_derivatives(self):
        model, xyt, basis = parcel.configure(dict(wind='veering'), [[.3, -.1], [.5, 1.]])
        model.double()
        response = CachedMeanResponse(stations=2)
        bounds = JointBounds(np.r_[[-.75, -.75, .00100005, 0], np.zeros(5)],
                             np.r_[[.75, .75, np.inf, np.inf], np.full(5, np.inf)])
        theta = np.r_[[.1, -.2, .02, .03], [.3, 1, 2, 1, .2]]
        obj = JointResidual(response, model, .15, xyt, basis, np.zeros(12), bounds)
        with parcel.experiment.ia.temporary_source_and_diffusion(model, .1, -.2, .05, .02):
            original = response(model, .15, xyt, basis).numpy()@theta[4:]
        np.testing.assert_allclose(obj.residual(theta), original, rtol=1e-10, atol=1e-12)
        jac = obj.jacobian(theta)
        for j in range(len(theta)):
            offset = np.zeros_like(theta)
            offset[j] = 1e-6
            numerical = (obj.residual(theta+offset)-obj.residual(theta-offset))/(2e-6)
            np.testing.assert_allclose(jac[:, j], numerical, rtol=1e-4, atol=1e-7)
        self.assertTrue(np.isfinite(jac).all())

    def test_pso_rejects_undeclared_amplitude_upper_bound(self):
        objective = SimpleNamespace(bounds=JointBounds(
            np.array([-1, -1, .001, 0, 0]), np.array([1, 1, 1, 1, np.inf])))
        with self.assertRaisesRegex(ValueError, 'finite shared'):
            fit_standard_pso(objective, seed=1, particles=4, iterations=2)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_external_coordinate_cuda_matches_cpu(self):
        values = []
        bounds = JointBounds(np.r_[[-.75, -.75, .00100005, 0], np.zeros(5)],
                             np.r_[[.75, .75, 1, 1], np.full(5, 10.)])
        theta = np.r_[[.1, -.2, .02, .03], [.3, 1, 2, 1, .2]]
        for device in ('cpu', 'cuda'):
            model, xyt, basis = parcel.configure(dict(wind='steady'), [[.3, -.1]], device)
            model.double()
            obj = JointResidual(CachedMeanResponse(stations=1), model, .15, xyt,
                                basis, np.zeros(6), bounds)
            values.append((obj.residual(theta), obj.jacobian(theta)))
        for cpu, gpu in zip(*values):
            np.testing.assert_allclose(cpu, gpu, rtol=2e-7, atol=1e-9)


if __name__ == '__main__':
    unittest.main()
