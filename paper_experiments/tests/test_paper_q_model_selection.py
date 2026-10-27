from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_joint_baselines import JointBounds
from paper_q_model_selection import (nested_hat_transform, gaussian_bic,
                                     fit_profiled_q_models)


class QCapacityTests(unittest.TestCase):
    def test_nested_hats_are_exact_and_nonnegative(self):
        clock = np.linspace(0., 1., 101)
        fine = np.column_stack([np.interp(clock, np.linspace(0., 1., 5), np.eye(5)[j])
                                for j in range(5)])
        for count in (1, 3, 5):
            transform = nested_hat_transform(5, count)
            self.assertTrue((transform >= 0).all())
            np.testing.assert_allclose(transform.sum(1), 1.)
            expected = np.ones((len(clock), 1)) if count == 1 else np.column_stack([
                np.interp(clock, np.linspace(0., 1., count), np.eye(count)[j]) for j in range(count)])
            np.testing.assert_allclose(fine@transform, expected, atol=1e-15)
        with self.assertRaises(ValueError):
            nested_hat_transform(5, 4)

    def test_bic_balances_fit_and_capacity_without_truth(self):
        self.assertLess(gaussian_bic(1., 40, 5), gaussian_bic(1., 40, 9))
        self.assertGreater(gaussian_bic(2., 40, 5), gaussian_bic(.2, 40, 9))
        self.assertTrue(np.isfinite(gaussian_bic(0., 40, 5)))
        difference = gaussian_bic(1., 40, 5)-gaussian_bic(.8, 40, 9)
        scaled = gaussian_bic(1e6, 40, 5)-gaussian_bic(.8e6, 40, 9)
        self.assertAlmostEqual(difference, scaled)

    def test_selection_keeps_observations_and_lifts_coefficients(self):
        objective = SimpleNamespace(basis=torch.eye(5, dtype=torch.float64),
            tensor=lambda v: torch.as_tensor(v, dtype=torch.float64),
            bounds=JointBounds(np.r_[[-1., -1., .001, 0.], np.zeros(5)],
                np.r_[[1., 1., 1., 1.], np.full(5, 10.)]),
            observations=torch.ones(30), response=lambda *a, **kw: torch.ones(30, 5).double(),
            model=SimpleNamespace(xs=torch.tensor(0., dtype=torch.float64)), sigma=.1, xyt=None)
        seen = []
        def fit(obj, starts, budget):
            seen.append(obj.observations.numel())
            return dict(coefficients=np.ones(obj.basis.shape[1]), sse=1., source_norm=[0., 0.],
                converged=True, seconds=.01)
        with patch('paper_q_model_selection.fit_profiled_trf', side_effect=fit):
            result = fit_profiled_q_models(objective, [np.zeros(9)])
        self.assertEqual(seen, [30, 30, 30])
        self.assertEqual(result['q_basis_count'], 1)
        np.testing.assert_allclose(result['coefficients'], np.ones(5))

    def test_real_transport_constant_release_selection_and_replay(self):
        from paper_fast_parcels import CachedMeanResponse
        from paper_joint_baselines import JointResidual
        import run_paper_parcel_pilot as parcel

        torch.set_num_threads(1)
        model, xyt, basis = parcel.configure(dict(wind='veering'), [[.3, -.1], [.5, 1.]])
        model.double()
        bounds = JointBounds(np.r_[[-.75, -.75, .00100005, 0], np.zeros(5)],
                             np.r_[[.75, .75, 1, 1], np.full(5, 10.)])
        theta = np.r_[[.1, -.2, .02, .03], np.ones(5)]
        obj = JointResidual(CachedMeanResponse(stations=2), model, .15, xyt,
                            basis, np.zeros(12), bounds)
        obj.observations = obj.tensor(obj.residual(theta)+1e-5*np.cos(np.arange(12)))
        result = fit_profiled_q_models(obj, [theta], max_nfev=60)
        self.assertEqual(result['q_basis_count'], 1)
        theta[:4] = [*result['source_norm'], result['diffusion_norm'][1],
                     result['diffusion_norm'][0]-result['diffusion_norm'][1]]
        theta[4:] = result['coefficients']
        prediction = obj.residual(theta)+obj.observations.numpy()
        np.testing.assert_allclose(prediction, result['prediction'], atol=1e-10)


if __name__ == '__main__':
    unittest.main()
