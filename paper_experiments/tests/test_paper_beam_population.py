from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_physical_beam_adapter import PhysicalBeamResponse
from paper_experiments.tests.test_paper_physical_beam_adapter import TinyModel, small_plan


class BeamPopulationTests(unittest.TestCase):
    def test_exact_and_single_precision_population(self):
        for device in ('cpu', 'cuda') if torch.cuda.is_available() else ('cpu',):
            model = TinyModel(device=device)
            plan = small_plan(model, beam_solver='analytic')
            xyt = torch.zeros((1, 3), dtype=torch.float64, device=device)
            basis = torch.eye(2, dtype=torch.float64, device=device)
            response = PhysicalBeamResponse(plan, xyt, basis)
            parameters = torch.tensor([[.2, .1, .3, .4], [.4, .3, .2, .2]],
                                      dtype=torch.float64, device=device)
            expected = torch.stack([response(model, 0., xyt, basis, physical_parameters=p)
                                    for p in parameters])
            batched = response.evaluate_population(model, 0., xyt, basis, parameters)
            torch.testing.assert_close(expected, batched, rtol=1e-10, atol=1e-10)
            fast = response.evaluate_population(model, 0., xyt, basis, parameters,
                                                 compute_dtype=torch.float32)
            np.testing.assert_allclose(fast.cpu(), expected.cpu(), rtol=1e-4, atol=1e-8)


if __name__ == '__main__':
    unittest.main()
