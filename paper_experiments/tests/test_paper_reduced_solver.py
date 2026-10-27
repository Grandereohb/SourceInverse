from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'deployment_runs/paper_baseline_dependencies'))
sys.path.insert(0, str(ROOT/'paper_experiments'))
from paper_joint_baselines import JointBounds, JointResidual, fit_standard_pso
from paper_reduced_solver import fit_profiled_trf, standard_pso_scales
from paper_fast_parcels import CachedMeanResponse
import run_paper_parcel_pilot as parcel


class ReducedAdapterTests(unittest.TestCase):
    def test_physical_adapter_reduces_a_real_transport_residual(self):
        torch.set_num_threads(1)
        model, xyt, basis = parcel.configure(dict(wind='veering'), [[.3, -.1], [.5, 1.]])
        model.double()
        bounds = JointBounds(np.r_[[-.75, -.75, .00100005, 0], np.zeros(5)],
                             np.r_[[.75, .75, 1, 1], np.full(5, 10.)])
        theta = np.r_[[.1, -.2, .02, .03], [.3, 1., 2., 1., .2]]
        obj = JointResidual(CachedMeanResponse(stations=2), model, .15, xyt,
                             basis, np.zeros(12), bounds)
        observed = obj.residual(theta)
        obj.observations = torch.tensor(observed, dtype=model.xs.dtype)
        start = theta.copy()
        start[:2] += [.01, .01]
        fit = fit_profiled_trf(obj, [start], max_nfev=60)
        self.assertLess(fit['sse'], np.sum(obj.residual(start)**2)*.01)
        self.assertTrue(np.isfinite(fit['prediction']).all())
        self.assertTrue((fit['coefficients'] >= 0).all())

    def test_scaled_standard_pso_improves_initial_value_and_restores_rng(self):
        bounds = JointBounds(np.array([-1., -1., .001, 0., 0.]), np.array([1., 1., 1., 1., 100.]))
        true = np.array([.1, -.2, .02, .01, 1.])
        start = np.array([0., 0., .04, .02, .8])
        scales = standard_pso_scales([start], bounds.lower, bounds.upper)
        def result(theta, seconds, **metadata):
            return dict(theta=theta, seconds=seconds, **metadata)
        obj = SimpleNamespace(bounds=bounds, residual=lambda theta: (theta-true)/scales, result=result)
        state = np.random.get_state()
        kwargs = dict(seed=1729, particles=24, iterations=100, starts=[start],
                      coordinate_scales=scales, local_fraction=.75, velocity_fraction=.2,
                      initial_spread=.1, zero_initial_velocity=True)
        fit = fit_standard_pso(obj, **kwargs)
        self.assertLess(fit['history'][-1], fit['history'][0]*.01)
        replay = fit_standard_pso(obj, **kwargs)
        np.testing.assert_array_equal(fit['theta'], replay['theta'])
        current = np.random.get_state()
        np.testing.assert_array_equal(current[1], state[1])
        self.assertEqual(current[2:], state[2:])
        bounds.validate(fit['theta'])


if __name__ == '__main__':
    unittest.main()
