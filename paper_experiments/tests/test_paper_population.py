from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_fast_parcels import CachedMeanResponse
from paper_joint_baselines import JointResidual, JointBounds
import run_paper_parcel_pilot as parcel


class PopulationTests(unittest.TestCase):
    def test_population_cost_matches_serial_cpu_cuda_and_chunks(self):
        torch.set_num_threads(1)
        for device in ('cpu', 'cuda') if torch.cuda.is_available() else ('cpu',):
            model, xyt, basis = parcel.configure(dict(wind='veering'), [[.3, -.1], [.5, 1.]], device)
            model.double()
            bounds = JointBounds(np.r_[[-.75, -.75, .00100005, 0], np.zeros(5)],
                                 np.r_[[.75, .75, 1, 1], np.full(5, 10.)])
            response = CachedMeanResponse(stations=2)
            obj = JointResidual(response, model, .15, xyt, basis, np.zeros(12), bounds)
            values = np.array([np.r_[[x, -.2, .02, .03], [.3, 1., 2., 1., .2]]
                               for x in (.1, .2, .3, -.1, 0.)])
            expected = np.array([np.sum(obj.residual(v)**2) for v in values])
            np.testing.assert_allclose(obj.population_cost(values), expected, rtol=1e-12, atol=1e-12)
            single = torch.stack([response(model, .15, xyt, basis,
                physical_parameters=obj.tensor(v[:4])) for v in values])
            batched = response.evaluate_population(model, .15, xyt, basis,
                obj.tensor(values[:, :4]), batch_size=2, max_kernel_elements=1000)
            torch.testing.assert_close(single, batched, rtol=1e-12, atol=1e-12)
            obj.population_dtype = torch.float32
            np.testing.assert_allclose(obj.population_cost(values), expected, rtol=1e-4, atol=1e-8)
            obj.observations = obj.tensor(single[0].numpy()@values[0, 4:] if device == 'cpu'
                else single[0].cpu().numpy()@values[0, 4:])+1e-3
            near = np.array([np.sum(obj.residual(v)**2) for v in values])
            np.testing.assert_allclose(obj.population_cost(values), near, rtol=1e-4, atol=1e-8)


if __name__ == '__main__':
    unittest.main()
