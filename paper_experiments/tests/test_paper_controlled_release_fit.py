import sys
from pathlib import Path
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fit_paper_controlled_release import MappedResponse, mean_hat_q
from paper_physical_beam_adapter import PhysicalBeamPlan


class FitDriverTests(unittest.TestCase):
    def test_hat_time_mean(self):
        self.assertAlmostEqual(mean_hat_q([1., 3., 5.], 17.), 3.)
        self.assertAlmostEqual(mean_hat_q([.001], 61.), .001)
        with self.assertRaises(ValueError):
            mean_hat_q([], 61.)

    def test_constant_mapping_preserves_linearity_and_gradient(self):
        class Response:
            basis = torch.tensor([[1., 0.], [0., 1.]])

            def __call__(self, model, sigma, xyt, basis, track_gradients=False):
                return model*torch.tensor([[2., 3.], [4., 5.]])
        basis = torch.ones((2, 1))
        callback = MappedResponse(Response(), basis, True)
        model = torch.tensor(2., requires_grad=True)
        output = callback(model, 0., None, basis, True)
        np.testing.assert_allclose(output.detach(), [[10.], [18.]])
        output.sum().backward()
        self.assertEqual(model.grad.item(), 14.)
        with self.assertRaises(ValueError):
            callback(model, 0., None, torch.zeros_like(basis))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_unindexed_cuda_device_is_canonicalized(self):
        plan = PhysicalBeamPlan([0., 1.], [[1., 0.]], [[0., 0., 1.6]],
            [[1., 0., 1.6]], [288.15], [101325.], [0., 1.], np.eye(2),
            height_m=.3, beam_order=2, time_order=2, release_order=2, device='cuda')
        self.assertEqual(plan.device, torch.device('cuda', torch.cuda.current_device()))


if __name__ == '__main__':
    unittest.main()
