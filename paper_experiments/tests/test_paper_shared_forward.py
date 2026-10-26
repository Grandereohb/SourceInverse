import gc
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_paper_shared_forward as shared
from paper_reference_transport import puff_response


class SharedForwardTests(unittest.TestCase):
    def test_common_response_matches_reference_and_is_q_linear(self):
        torch.set_num_threads(1)
        stations = np.array([[.3, -.1], [.5, 1.], [1., -1.]])
        for wind in ('steady', 'veering'):
            model, xyt, basis = shared.formal.parcel.configure(
                dict(wind=wind), stations, wind_interval_mean=True)
            response = shared.SharedCaseResponse(len(stations))
            with shared.formal.base.ia.temporary_source_and_diffusion(model, -.2, .1, .05, .02):
                matrix = response(model, .15, xyt, basis).cpu().numpy()
                # Three solvers now evaluate this exact matrix, not three kernels.
                np.testing.assert_array_equal(matrix, response(model, .15, xyt, basis).cpu().numpy())
            times = (np.arange(96)+.5)/16
            reference = puff_response(stations, times, np.linspace(0., 6., 5),
                [-.2, .1], .05, .02, wind=wind, sigma=.15, order=64)
            reference = reference.reshape(6, 16, 3, 5).mean(1).reshape(18, 5)
            self.assertLess(np.linalg.norm(matrix-reference)/np.linalg.norm(reference), .01)
            a, b = np.arange(1., 6.), np.arange(5., 0., -1.)
            np.testing.assert_allclose(matrix@(a+b), matrix@a+matrix@b, atol=1e-12)

    def test_short_cpu_flow_uses_shared_starts_and_preserves_all_observations(self):
        torch.set_num_threads(1)
        spec = dict(case_id='shared_smoke', seed=990221, cohort='nominal', network='ideal5',
                    wind='veering', release='pulse', repeat=0)
        network = dict(stations=[[.3,-.1],[.5,1.],[1.,-1.],[1.7,.9],[2.,-.9]],
                       length_m=1000., sigma=.15)
        metadata, data = shared.formal.make_input(spec, network)
        with tempfile.TemporaryDirectory() as directory:
            case = Path(directory)
            shared.formal.base.save_json(case/'input.json', metadata)
            np.savez(case/'input.npz', **data)
            result = shared.fit_case(case, dict(search_step_norm=.5, adam_steps=2,
                puff_powell_max_evaluations_per_start=5), 'cpu')
            self.assertEqual(set(result['methods']),
                {'ia_vpdi', 'matched_joint_adam', shared.METHOD})
            self.assertTrue(result['forward_consistency']['shared_response'])
            for value in result['methods'].values():
                self.assertTrue(np.isfinite(value['observed_mse']))
                self.assertEqual(len(value['prediction']), 30)
            self.assertEqual(result['methods'][shared.METHOD]['response_matrix_calls'], 12)
            saved = json.loads((case/'result.json').read_text(encoding='utf-8'))
            self.assertEqual(saved['comparison_version'], 'shared_forward_v3')
            gc.collect()

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_response_matches_cpu(self):
        matrices = []
        for device in ('cpu', 'cuda'):
            model, xyt, basis = shared.formal.parcel.configure(
                dict(wind='veering'), [[.3,-.1],[.5,1.]], device, wind_interval_mean=True)
            with shared.formal.base.ia.temporary_source_and_diffusion(model, -.2, .1, .05, .02):
                matrices.append(shared.SharedCaseResponse(2)(model, .15, xyt, basis).cpu().numpy())
        np.testing.assert_allclose(*matrices, rtol=2e-5, atol=2e-6)


if __name__ == '__main__':
    unittest.main()
