"""Production IA adapter checks, separate from the research forward operator."""

import unittest

import numpy as np
import torch

from tests.test_ia_vpdi import synthetic_case
from ia_vpdi import (refine_profiled_candidate, score_all_data_candidate,
                    run_ia_vpdi, build_ia_vpdi_diagnostics)
from variable_projection import build_response_matrix


class ProductionReducedTests(unittest.TestCase):
    def check_device(self, device, transport='grid'):
        torch.set_num_threads(1)
        model, xyt, basis, _, observed, _ = synthetic_case(device=device)
        if transport != 'grid':
            model.recurrent_transport_solver = transport
            model.recurrent_parcel_subdivisions = 8
            truth = build_response_matrix(model, .08, xyt, basis)
            observed = (truth @ truth.new_tensor([.5, 1.8, .7])).detach()
        start = score_all_data_candidate(model, .08, xyt, observed, basis,
                                         .14, -.07, .004, .002)
        flags = [p.requires_grad for p in (model.xs, model.ys, model.rawD_perp,
                                          model.rawD_parallel_delta)]
        fit = refine_profiled_candidate(model, .08, xyt, observed, basis, start,
            (-.8, .8), (-.8, .8), steps=30, outer_solver='trf')
        self.assertTrue(fit.success, fit.message)
        self.assertLess(fit.best.profile_score, start.profile_score*.05)
        self.assertEqual(fit.outer_solver, 'trf')
        prediction = build_response_matrix(model, .08, xyt, basis).detach().cpu().numpy()@fit.best.coefficients
        np.testing.assert_allclose(prediction, fit.best.prediction, rtol=1e-5, atol=1e-7)
        self.assertEqual(flags, [p.requires_grad for p in (model.xs, model.ys,
            model.rawD_perp, model.rawD_parallel_delta)])
        self.assertTrue(np.isfinite(fit.best.prediction).all())

    def test_cpu_production_adapter(self):
        self.check_device('cpu')

    def test_parcel_production_adapter(self):
        self.check_device('cpu', 'gaussian_parcels')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_production_adapter(self):
        self.check_device('cuda')

    def test_fixed_diffusion_respects_q_cap(self):
        model, xyt, basis, _, observed, _ = synthetic_case()
        start = score_all_data_candidate(model, .08, xyt, observed, basis,
                                         .1, -.1, .004, .002)
        fit = refine_profiled_candidate(model, .08, xyt, observed, basis, start,
            (-.8, .8), (-.8, .8), steps=4, q_upper_bound=.2,
            optimize_diffusion=False, outer_solver='trf')
        self.assertTrue(fit.success, fit.message)
        self.assertLessEqual(fit.best.coefficients.max(), .2+1e-10)
        self.assertAlmostEqual(fit.best.diffusion_perp_norm, .002)

    def test_small_search_exports_solver_and_optimizer_status(self):
        model, xyt, _, _, observed, _ = synthetic_case()
        result = run_ia_vpdi(model, .08, xyt, observed, (-.2, .2), (-.2, .2), .2,
            q_basis_count=3, top_k=1, diffusion_scale_seeds=(1.,),
            diffusion_ratio_seeds=(2.,), outer_steps=8, outer_solver='trf')
        report = build_ia_vpdi_diagnostics(result)['report']
        self.assertEqual(report['outer_solvers'], ['trf'])
        self.assertIn('converged', report['outer_optimizer_details'][0])
        self.assertTrue(np.isfinite(result.best.profile_score))


if __name__ == '__main__':
    unittest.main()
